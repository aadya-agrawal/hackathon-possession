"""Kit-colour audit of every tracked slot's team -> per-frame track team override (evidence/track_team_override.npz).

Label-free: tracked players are projected into the tactical video with the frame homography, matched to COCO person
detections (yolo26l, imgsz 1920, conf 0.2, torso crops) and the torso colour is classified with the kit-colour
model of tools/extract_team_at_ball.py (HistGradientBoosting trained on tracking-matched persons of 3000 other
frames; goalkeeper class -> team from the pitch side of the feet).
  1. grid    : persons on graded frames with frame % 25 == 0 (1 s). Per (frame, on-screen track) sample: one-to-one
               scale-aware person match, colour team when matched with probability >= 0.8. Base team = metadata
               team with the goalkeeper end rule (goalkeeper slot within 30 m of the other goal -> other team; home
               defends x < 52.5 in the halves where possession.ballfree.attack_directions finds home attacking +x,
               tools/derive_match_constants.py home_left_frames). Per track a two-state Viterbi (P(disagree) 0.05
               vs 0.85, switch log(1e-4)) gives anomalous stretches >= 5 s.
  2. select  : candidate stretches with >= 5 disagreeing samples, >= 85% disagreement and mean colour confidence
               >= 0.8; dense 0.2 s frames planned around the boundaries of clean candidates.
  3. dense   : persons on those dense frames.
  4. final   : boundaries refined with grid + dense samples (midpoint between last agreeing and first disagreeing
               sample, snapped to the track's segment start/end when one lies in between). Category: tracking_reset
               (a team has < 6 tracks on average; reported only), mass_mislabel (other tracks disagree >= 10%) or
               swap. Override (int8 (frames, 46), -1 = keep) = colour team on the graded tracked frames of swap and
               mass_mislabel stretches. 'provenance' (JSON string in the npz) lists the stretches; its label
               statistics (labels_there, labels_equal_*) read --labels (default submissions/possession_compliant.csv,
               read only) and do not affect the override.

Usage (run from the repository root; export TMPDIR=/dev/shm). The colour model is <TAB_WORK>/colour_model.pkl
written by tools/extract_team_at_ball.py assemble (trained here from <TAB_WORK>'s person cache if missing):
  python tools/audit_track_teams.py run --devices 5,6,7 --procs 2 --tab-work /dev/shm/tab_work \\
      --work /dev/shm/audit_work --out track_team_override.npz          (~2.5 min on 3 L40S after the colour model)
  stages: persons --set grid|dense --device D [--shard i --nshards n] ; select ; final --out OUT  (all with --work)
Outputs: OUT (override + provenance), OUT.json (per-track audit, stretches, category summary, goalkeeper positive
control), work dir caches (persons_*.npz, audit_samples.npz, task1_raw.json, stretch_candidates.json,
frames_dense.json). Determinism: as tools/extract_team_at_ball.py (person boxes bit-identical across L40S GPUs,
shards and concurrency with ultralytics 8.4.40 / torch 2.4.1+cu121).
"""
import argparse
import json
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive_match_constants as C  # noqa: E402
import extract_team_at_ball as X  # noqa: E402

REPO = X.REPO
N_FRAMES = X.N_FRAMES
CONF = 0.8
W, H = 1920, 1080
STEP = 25
SWITCH = np.log(1e-4)
GK_END_RADIUS = 30.0


# ---------------------------------------------------------------- persons
def grid_frames():
    return [int(f) for f in np.flatnonzero(X.graded_mask()) if f % STEP == 0]


def set_frames(work, kind):
    if kind == "grid":
        return grid_frames()
    return sorted(set(json.loads((Path(work) / "frames_dense.json").read_text())))


def stage_persons(args):
    frames = set_frames(args.work, args.set)
    todo = np.array_split(np.array(frames), args.nshards)[args.shard].tolist()
    model = X.load_yolo(args.person_weights)
    dev = X.device_arg(args.device)
    rows, t0 = {}, time.time()
    for f, image in X.Video(X.data_paths(args.data_dir)["video"]).frames(todo):
        rows[f] = X.detect_persons(image, model, dev)
    path = Path(args.work) / "persons_{}_{}_{}.npz".format(args.set, args.shard, args.nshards)
    X.save_person_cache(path, rows)
    np.save(str(path)[:-4] + "_done.npy", np.array(todo, dtype=np.int32))
    print("persons", args.set, args.shard, len(todo), "{:.0f}s".format(time.time() - t0), flush=True)


def load_persons(work, kind):
    """Persons of one set as flat arrays in (frame, x1) order, plus the sorted frames processed."""
    paths = sorted(Path(work).glob("persons_{}_*_*.npz".format(kind)))
    done = np.unique(np.concatenate([np.load(str(p)[:-4] + "_done.npy") for p in paths]))
    cache = {}
    for p in paths:
        with np.load(p) as z:
            d = {k: z[k] for k in ("frame", "box", "conf", "crop")}
        bounds = np.flatnonzero(np.diff(np.r_[-1, d["frame"], -2]))
        for a, e in zip(bounds[:-1], bounds[1:]):
            cache[int(d["frame"][a])] = (d["box"][a:e], d["conf"][a:e], d["crop"][a:e])
    frames = sorted(cache)
    out = {"frame": np.concatenate([np.full(len(cache[f][0]), f, dtype=np.int32) for f in frames]),
           "box": np.concatenate([cache[f][0] for f in frames]), "conf": np.concatenate([cache[f][1] for f in frames]),
           "crop": np.concatenate([cache[f][2] for f in frames])}
    return out, done


# ---------------------------------------------------------------- colour
def colour_model(args):
    path = Path(args.tab_work) / "colour_model.pkl"
    if not path.exists():
        pos, team, gk_slots = X.load_tracking(args.data_dir)
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        windows = X.match_constants(args.tab_work, devices, args.data_dir, Path(args.person_weights).parent)
        clf, info = X.train_colour(X.load_person_cache(args.tab_work), pos, team, X.load_homographies(args.data_dir),
                                   gk_slots, windows["masked_windows"])
        path.write_bytes(pickle.dumps((clf, info)))
    model = pickle.loads(path.read_bytes())
    return (model[0] if isinstance(model, tuple) else model), str(path)


def colour_classes(clf, persons):
    p = clf.predict_proba(X.colour_features(persons["crop"]))
    full = np.zeros((len(p), 5))
    full[:, clf.classes_] = p
    merged = np.c_[full[:, 0], full[:, 1], full[:, 2] + full[:, 3], full[:, 4]]
    return merged.argmax(1), merged.max(1)


def gk_corrected_teams(pos, team, gk_slots, home_left):
    """Metadata team per frame and slot with the goalkeeper end rule (home_left: home defends x < 52.5)."""
    teams = np.tile(team.astype(np.int8), (N_FRAMES, 1))
    home_goal_x = np.where(home_left, 0.0, 105.0)
    for slot in (gk_slots[0], gk_slots[1]):
        t = int(team[slot])
        own = home_goal_x if t == 0 else 105.0 - home_goal_x
        x = pos[:, slot, 0]
        sw = np.isfinite(x) & (np.abs(x - (105.0 - own)) < GK_END_RADIUS)
        teams[sw, slot] = 1 - t
    return teams


def build_samples(pos, mats, persons, done, clf):
    cls, conf = colour_classes(clf, persons)
    frames = persons["frame"]
    idx = {}
    b = np.flatnonzero(np.diff(np.r_[-1, frames, -2]))
    for a, e in zip(b[:-1], b[1:]):
        idx[int(frames[a])] = (a, e)
    rows = []
    for f in done:
        m = mats[f]
        if not np.isfinite(m).all():
            continue
        ok = np.flatnonzero(np.isfinite(pos[f]).all(1))
        if not len(ok):
            continue
        px = X.to_pixels(m, pos[f][ok])
        on = (px[:, 0] > 10) & (px[:, 0] < W - 10) & (px[:, 1] > 60) & (px[:, 1] < H - 5)
        a, e = idx.get(int(f), (0, 0))
        boxes = persons["box"][a:e]
        slot, ratio, iso = X.match_persons(boxes, px, ok)  # match against all tracks, report on-screen ones
        by_slot = {int(s): k for k, s in enumerate(slot) if s >= 0}
        if len(boxes):
            feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
            hgt = np.maximum(boxes[:, 3] - boxes[:, 1], 1)
            near = (np.linalg.norm(feet[:, None] - px[None], axis=2) / hgt[:, None]).min(0)
        else:
            near = np.full(len(ok), np.inf)
        for j in np.flatnonzero(on):
            s = int(ok[j])
            k = by_slot.get(s, -1)
            rows.append((f, s, k >= 0, ratio[k] if k >= 0 else np.inf, iso[k] if k >= 0 else False, near[j],
                         cls[a + k] if k >= 0 else -1, conf[a + k] if k >= 0 else 0.0, pos[f, s, 0]))
    r = list(zip(*rows))
    return {"frame": np.array(r[0], np.int32), "track": np.array(r[1], np.int16), "matched": np.array(r[2], bool),
            "ratio": np.array(r[3], np.float32), "isolated": np.array(r[4], bool), "near": np.array(r[5], np.float32),
            "cls": np.array(r[6], np.int8), "conf": np.array(r[7], np.float32), "x": np.array(r[8], np.float32)}


def colour_team(s, home_left):
    """0/1 team from colour (goalkeeper class -> side of pitch), -1 other, -2 uninformative."""
    gk_team = np.where((s["x"] < 52.5) == home_left[s["frame"]], 0, 1)
    t = np.select([s["cls"] == 0, s["cls"] == 1, s["cls"] == 2, s["cls"] == 3], [0, 1, gk_team, -1], -2)
    return np.where(s["matched"] & (s["conf"] >= CONF), t, -2)


# ---------------------------------------------------------------- stretches
def viterbi2(obs, p_bad0, p_bad1):
    """obs: 1 bad, 0 good, -1 missing. Returns state per sample (1 = anomalous)."""
    n = len(obs)
    e = np.zeros((n, 2))
    e[obs == 1] = np.log([p_bad0, p_bad1])
    e[obs == 0] = np.log([1 - p_bad0, 1 - p_bad1])
    score = np.array([0.0, SWITCH]) + e[0]
    back = np.zeros((n, 2), dtype=np.int8)
    for i in range(1, n):
        stay, move = score, score[::-1] + SWITCH
        back[i] = np.where(stay >= move, [0, 1], [1, 0])
        score = np.maximum(stay, move) + e[i]
    st = np.zeros(n, dtype=np.int8)
    st[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        st[i - 1] = back[i, st[i]]
    return st


def stretches(frames, state, obs, min_frames=125, max_gap=5 * STEP):
    """Runs of state 1 split where consecutive samples are > max_gap apart; keep spans >= min_frames."""
    out = []
    i, n = 0, len(frames)
    while i < n:
        if state[i] != 1:
            i += 1
            continue
        j = i
        while j + 1 < n and state[j + 1] == 1 and frames[j + 1] - frames[j] <= max_gap:
            j += 1
        span = frames[j] - frames[i] + STEP
        o = obs[i:j + 1]
        if span >= min_frames:
            out.append({"start": int(frames[i]), "stop": int(frames[j] + STEP), "seconds": span / 25.0,
                        "samples": int(j - i + 1), "bad": int((o == 1).sum()), "good": int((o == 0).sum())})
        i = j + 1
    return out


def audit_report(s, ct, team, corr, n_done):
    absent = ~(s["near"] <= 0.6)
    base_miss = float(absent.mean())
    report = {"frames_sampled": int(n_done), "samples": int(len(ct)), "colour_conf": CONF,
              "global_person_absent_rate": round(base_miss, 4), "tracks": {}}
    for k in range(len(team)):
        sel = np.flatnonzero(s["track"] == k)
        if not len(sel):
            report["tracks"][str(k)] = {"samples": 0}
            continue
        fr = s["frame"][sel]
        c = ct[sel]
        inf = c >= 0
        dis_meta = np.where(c >= 0, (c != team[k]).astype(int), -1)
        dis_corr = np.where(c >= 0, (c != corr[sel]).astype(int), -1)
        dis_meta[c == -1] = -1
        ab = absent[sel].astype(int)
        cls = s["cls"][sel][s["matched"][sel]]
        iso = inf & s["isolated"][sel]
        report["tracks"][str(k)] = {
            "meta_team": int(team[k]), "samples": int(len(sel)),
            "matched_share": round(float(s["matched"][sel].mean()), 4),
            "person_present_share": round(float(1 - absent[sel].mean()), 4),
            "colour_class_share_matched": {n: round(float((cls == i).mean()), 4) if len(cls) else None
                                           for i, n in enumerate(("home", "away", "goalkeeper", "other"))},
            "informative": int(inf.sum()),
            "agree_meta": round(float((c[inf] == team[k]).mean()), 4) if inf.any() else None,
            "agree_corrected": round(float((c[inf] == corr[sel][inf]).mean()), 4) if inf.any() else None,
            "agree_meta_isolated": round(float((c[iso] == team[k]).mean()), 4) if iso.any() else None,
            "swap_stretches_vs_meta": stretches(fr, viterbi2(dis_meta, 0.05, 0.85), dis_meta),
            "swap_stretches_vs_corrected": stretches(fr, viterbi2(dis_corr, 0.05, 0.85), dis_corr),
            "ghost_stretches": stretches(fr, viterbi2(ab, max(base_miss, 0.05), 0.9), ab),
        }
    return report


def select_candidates(s, raw, pos, team):
    """Classify corrected-team swap stretches; dense 0.2 s frames around boundaries of clean ones."""
    ct, corr = s["colour_team"], s["corrected"]
    inf = ct >= 0
    dis = inf & (ct != corr)
    _, inv = np.unique(s["frame"], return_inverse=True)
    n_inf, n_dis = np.bincount(inv, inf), np.bincount(inv, dis)
    ok = np.isfinite(pos[:, :, 0])
    hn, an = ok[:, team == 0].sum(1), ok[:, team == 1].sum(1)
    out, frames = [], set()
    for k, v in raw["tracks"].items():
        k = int(k)
        for st in v.get("swap_stretches_vs_corrected", []):
            sel = np.flatnonzero((s["track"] == k) & (s["frame"] >= st["start"]) & (s["frame"] < st["stop"]))
            fi = inv[sel]
            other = float(np.mean((n_dis[fi] - dis[sel]) / np.maximum(n_inf[fi] - inf[sel], 1)))
            n_i = int(inf[sel].sum())
            frac = float(dis[sel].sum() / max(n_i, 1))
            conf = float(s["conf"][sel][inf[sel]].mean()) if n_i else 0.0
            fr = np.arange(st["start"], st["stop"])
            counts = (float(hn[fr].mean()), float(an[fr].mean()))
            reasons = []
            if st["bad"] < 5:
                reasons.append("few_disagreeing_samples")
            if frac < 0.85:
                reasons.append("mixed")
            if conf < 0.8:
                reasons.append("low_colour_confidence")
            if other >= 0.10:
                reasons.append("frame_level_failure")
            if min(counts) < 8:
                reasons.append("tracking_reset")
            out.append(dict(st, track=k, meta_team=int(team[k]), informative=n_i, disagree_share=round(frac, 3),
                            mean_conf=round(conf, 3), other_tracks_disagree=round(other, 3),
                            team_counts=[round(c, 1) for c in counts], rejected=reasons))
            if not reasons:
                bad_frames = s["frame"][sel][dis[sel]]
                a, b = int(bad_frames.min()), int(bad_frames.max())
                frames.update(range(max(0, a - 75), a + 1, 5))
                frames.update(range(b, b + 76, 5))
    return out, sorted(frames)


def stage_select(args):
    work = Path(args.work)
    pos, team, gk_slots = X.load_tracking(args.data_dir)
    mats = X.load_homographies(args.data_dir)
    clf, _ = colour_model(args)
    persons, done = load_persons(work, "grid")
    s = build_samples(pos, mats, persons, done, clf)
    home_left = C.home_left_frames(pos, team)
    ct = colour_team(s, home_left)
    corr = gk_corrected_teams(pos, team, gk_slots, home_left)[s["frame"], s["track"]]
    np.savez(work / "audit_samples.npz", colour_team=ct, corrected=corr, **s)
    raw = audit_report(s, ct, team, corr, len(done))
    (work / "task1_raw.json").write_text(json.dumps(raw, indent=1))
    s.update(colour_team=ct, corrected=corr)
    cands, frames = select_candidates(s, raw, pos, team)
    (work / "stretch_candidates.json").write_text(json.dumps(cands, indent=1))
    (work / "frames_dense.json").write_text(json.dumps(frames))
    print(len(cands), "candidates;", sum(not c["rejected"] for c in cands), "clean;", len(frames), "dense frames",
          flush=True)


# ---------------------------------------------------------------- final
def stage_final(args):
    work = Path(args.work)
    pos, team, gk_slots = X.load_tracking(args.data_dir)
    ball = X.load_ball(args.data_dir)
    mats = X.load_homographies(args.data_dir)
    clf, model_path = colour_model(args)
    grid = dict(np.load(work / "audit_samples.npz"))
    persons, done = load_persons(work, "dense")
    dense = build_samples(pos, mats, persons, done, clf)
    home_left = C.home_left_frames(pos, team)
    dense["colour_team"] = colour_team(dense, home_left)
    gkt = gk_corrected_teams(pos, team, gk_slots, home_left)
    s = {k: np.concatenate([grid[k], dense[k]]) for k in ("frame", "track", "colour_team")}
    ok = np.isfinite(pos[:, :, 0])
    graded = X.graded_mask()
    labels = np.loadtxt(args.labels, delimiter=",", skiprows=1, dtype=str)[:, 1]
    # nearest tracked player to the tracked ball (label statistics only)
    dist = np.linalg.norm(pos - ball[:, None], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    nearest = dist.argmin(1)
    near_d = dist[np.arange(N_FRAMES), nearest]
    cands = json.loads((work / "stretch_candidates.json").read_text())
    override = np.full((N_FRAMES, len(team)), -1, dtype=np.int8)
    code = {0: "H", 1: "A"}
    final = []
    for c in cands:
        if c["bad"] < 5 or c["disagree_share"] < 0.85 or c["mean_conf"] < 0.8:
            continue
        k = c["track"]
        cat = "tracking_reset" if min(c["team_counts"]) < 6 else (
            "mass_mislabel" if c["other_tracks_disagree"] >= 0.10 else "swap")
        m = s["track"] == k
        fr, ctk = s["frame"][m], s["colour_team"][m]
        order = np.argsort(fr)
        fr, ctk = fr[order], ctk[order]
        inf = ctk >= 0
        fr, ctk = fr[inf], ctk[inf]
        dis = ctk != gkt[fr, k]
        inside = (fr >= c["start"]) & (fr < c["stop"]) & dis
        first, last = fr[inside].min(), fr[inside].max()
        prev_ok = fr[(fr < first) & ~dis]
        next_ok = fr[(fr > last) & ~dis]
        d = np.diff(np.r_[0, ok[:, k].astype(np.int8), 0])
        seg_s, seg_e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        lo = prev_ok.max() if len(prev_ok) else first - 25
        hi = next_ok.min() if len(next_ok) else last + 25
        snap_s = seg_s[(seg_s > lo) & (seg_s <= first)]
        snap_e = seg_e[(seg_e > last) & (seg_e <= hi)]
        start = int(snap_s.max()) if len(snap_s) else int((lo + first + 1) // 2)
        stop = int(snap_e.min()) if len(snap_e) else int((last + hi + 1) // 2)
        team_after = int(Counter(ctk[inside].tolist()).most_common(1)[0][0])
        span = np.arange(start, stop)
        tracked = span[ok[span, k] & graded[span]]
        at_ball = tracked[(nearest[tracked] == k) & (near_d[tracked] < 2.0)]
        lab = Counter(labels[at_ball].tolist())
        final.append({"track": k, "meta_team": code[int(team[k])], "colour_team": code[team_after], "category": cat,
                      "start": start, "stop": stop, "seconds": round((stop - start) / 25, 1),
                      "start_snapped_to_segment": bool(len(snap_s)), "stop_snapped_to_segment": bool(len(snap_e)),
                      "boundary_uncertainty_frames": [int(first - lo), int(hi - last)],
                      "samples_disagree": int(inside.sum()), "graded_tracked_frames": int(len(tracked)),
                      "frames_nearest_to_ball_lt2m": int(len(at_ball)), "labels_there": dict(lab),
                      "labels_equal_meta_team": int(lab.get(code[int(team[k])], 0)),
                      "labels_equal_colour_team": int(lab.get(code[team_after], 0)),
                      "team_counts": c["team_counts"], "other_tracks_disagree": c["other_tracks_disagree"]})
        if cat != "tracking_reset":
            override[tracked, k] = team_after
    raw = json.loads((work / "task1_raw.json").read_text())
    # positive control: goalkeeper episodes (corrected != metadata) seen on screen in the 1 s grid
    ct, fr, tk = grid["colour_team"], grid["frame"], grid["track"]
    pc = {}
    for slot in (gk_slots[0], gk_slots[1]):
        ep = (tk == slot) & (gkt[fr, slot] != team[slot])
        inf = ep & (ct >= 0)
        gk_frames = graded & (gkt[:, slot] != team[slot])
        cov = np.zeros(N_FRAMES, bool)
        for st in raw["tracks"][str(slot)]["swap_stretches_vs_meta"]:
            cov[st["start"]:st["stop"]] = True
        pc[slot] = {"episode_graded_frames": int(gk_frames.sum()), "grid_samples_on_screen": int(ep.sum()),
                    "informative": int(inf.sum()),
                    "colour_disagrees_with_metadata": round(float((ct[inf] != team[slot]).mean()), 4) if inf.any() else None,
                    "colour_agrees_with_corrected": round(float((ct[inf] == gkt[fr[inf], slot]).mean()), 4) if inf.any() else None,
                    "episode_frames_covered_by_detected_stretches": round(float((cov & gk_frames).sum() / max(gk_frames.sum(), 1)), 4),
                    "detected_stretches_vs_meta": len(raw["tracks"][str(slot)]["swap_stretches_vs_meta"])}
    # The committed npz must not depend on the current labels or local paths, so label statistics
    # stay in the JSON report only.
    label_keys = ("labels_there", "labels_equal_meta_team", "labels_equal_colour_team")
    provenance = {"source": "tools/audit_track_teams.py", "colour_model": Path(model_path).name,
                  "persons": "yolo26l 1920 conf0.2",
                  "categories_applied": ["swap", "mass_mislabel"],
                  "base": "metadata team with repo goalkeeper end rule (30 m)",
                  "stretches": [{k: v for k, v in f.items() if k not in label_keys} for f in final]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, override=override, provenance=np.array(json.dumps(provenance)))
    cat_sum = {}
    for cat in ("swap", "mass_mislabel", "tracking_reset"):
        sub = [f for f in final if f["category"] == cat]
        cat_sum[cat] = {"stretches": len(sub), "tracks": sorted({f["track"] for f in sub}),
                        "seconds": round(sum(f["seconds"] for f in sub), 1),
                        "graded_tracked_frames": sum(f["graded_tracked_frames"] for f in sub),
                        "frames_nearest_to_ball_lt2m": sum(f["frames_nearest_to_ball_lt2m"] for f in sub),
                        "labels_equal_meta_team": sum(f["labels_equal_meta_team"] for f in sub),
                        "labels_equal_colour_team": sum(f["labels_equal_colour_team"] for f in sub),
                        "labels_D": sum(f["labels_there"].get("D", 0) for f in sub)}
    per_track = {k: {kk: v.get(kk) for kk in ("meta_team", "samples", "matched_share", "person_present_share",
                                                "informative", "agree_meta", "agree_corrected", "agree_meta_isolated",
                                                "colour_class_share_matched")}
                 for k, v in raw["tracks"].items()}
    ghosts = {k: v["ghost_stretches"] for k, v in raw["tracks"].items() if v.get("ghost_stretches")}
    report = {"per_track": per_track, "stretches": final, "category_summary": cat_sum, "positive_control_gk": pc,
              "ghost_stretches": ghosts, "override_changed_track_frames": int((override >= 0).sum()),
              "dense_frames": int(len(done))}
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"category_summary": cat_sum, "override_changed_track_frames": report["override_changed_track_frames"]}),
          flush=True)


# ---------------------------------------------------------------- driver
def run_persons_sharded(args, kind):
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    n = len(devices) * args.procs
    t0, procs = time.time(), []
    for i in range(n):
        log = open(Path(args.work) / "log_persons_{}_{}.txt".format(kind, i), "w")
        cmd = [sys.executable, str(Path(__file__).resolve()), "persons", "--set", kind, "--work", args.work,
               "--tab-work", args.tab_work, "--person-weights", args.person_weights, "--device", devices[i % len(devices)],
               "--shard", str(i), "--nshards", str(n)] + (["--data-dir", args.data_dir] if args.data_dir else [])
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit("persons {} shards failed: {}".format(kind, codes))
    print("persons", kind, "done in {:.0f}s".format(time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("persons", "select", "final", "run"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--tab-work", required=True, help="tools/extract_team_at_ball.py work dir (colour model)")
    ap.add_argument("--data-dir", default=None, help="provided data bundle (default: games.DATA_DIR)")
    ap.add_argument("--person-weights", default=str(REPO / "cache/yolo26l.pt"))
    ap.add_argument("--labels", default=str(REPO / "submissions/possession_compliant.csv"),
                    help="label CSV for the provenance statistics only (read only)")
    ap.add_argument("--set", choices=("grid", "dense"), default="grid")
    ap.add_argument("--device", default="0")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    Path(args.work).mkdir(parents=True, exist_ok=True)
    if args.stage in ("final", "run") and not args.out:
        raise SystemExit("--out required")
    if args.stage in ("persons", "run") and X.sha256(args.person_weights) != X.EXPECTED_SHA256["person"]:
        raise SystemExit("person weights sha256 mismatch (run tools/fetch_video_weights.py)")
    if args.stage == "persons":
        stage_persons(args)
    elif args.stage == "select":
        stage_select(args)
    elif args.stage == "final":
        stage_final(args)
    else:
        colour_model(args)
        run_persons_sharded(args, "grid")
        stage_select(args)
        run_persons_sharded(args, "dense")
        stage_final(args)


if __name__ == "__main__":
    main()
