"""Derive the match-specific inputs of tools/extract_team_at_ball.py from the data bundle (no hand-set frames).

Everything below is computed from the provided data with fixed, match-independent rules and seeds; nothing is
hand-embedded. The derived values are cached in <work>/match_constants.json (and the intermediate files in
<work>/match_constants/) so the sharded extraction processes all read the same values.

  1. masked_windows : the 116 swap-test windows [start, stop, split] (split 'd' = first half = development,
                      't' = second half = test). They only exclude kit-model training frames (padded by 50) and
                      define --frames development/test. Teacher labels = possession CSV of
                      `python -m possession.cli --config configs/submission_contact_carry.json` (tracking-only
                      selection; expected sha256 TEACHER_SHA256, written to <work>/match_constants/teacher.csv).
                      observed = raw tactical ball present; live = graded (games.half_video_blocks) & teacher label
                      H/A. Real gaps = missing-ball runs >= 125 frames that are >= 50% live; one window per
                      observed live run >= 150 frames, length = min(real gap length drawn with
                      np.random.default_rng(2027), run - 20), kept when >= 125 and centred in the run.
  2. static_cells   : detector-clutter cells [x + 0.5, y + 0.5, frames] in pitch metres. Frames = the stratified
                      visibility-pilot sample (seed 20271115): 540 frames in raw-ball gaps >= 125 graded frames,
                      allocated ~ sqrt(cell frames) (x2 for broadcast-referenced live frames, >= 3 per cell) over
                      cells half x gap length band x edge/interior x broadcast state x players >= 20, >= 10 frames
                      apart, plus 300 control frames (tracked ball +-5 frames) over half x broadcast state x
                      players cells. Broadcast state = possession.broadcast.align_broadcast with the broadcast_*
                      alignment parameters of configs/submission_broadcast_stoppage.json (live_ref = live view with
                      a mapped broadcast ball).
                      Detectors (tiled, imgsz 1280, conf 0.01, 10 px merge, top 150; see PILOT_DETECTORS): COCO
                      sports ball (class 32) of yolo26l 2x2 / 3x3 tiles and yolo26x 2x2, SoccerNet ball of
                      yolo-sn-ball-opt 2x2 / 3x3. A 1 m cell (floor of the projected candidate, conf >= 0.05) is
                      static when its 3x3 neighbourhood fires on >= 4% of the sample frames across >= 3 different
                      10-minute blocks and the cell holds >= half the hits of its busiest neighbour.
  3. home_left      : per frame, True when the home team defends x < length/2 (goalkeeper side), from
                      possession.ballfree.attack_directions over the tracking of each half (games.half_video_blocks),
                      exactly as possession/cli.py configured_frame_teams; frames outside the halves take the
                      direction of the latest half that started (the first half before kick-off).

Verification only (never used as a source of values): EXPECTED_SHA256 holds the sha256 of the compact JSON of the
values that were previously embedded in tools/extract_team_at_ball.py; derive() reports whether they match (checked
16 Nov 2026 on NVIDIA L40S with the versions listed in tools/extract_team_at_ball.py: teacher CSV, pilot frames,
all 848 x 5 pilot detections and both derived values identical; ~1.5 min on 6 GPUs plus the teacher CLI run).

Usage (repository root, TMPDIR on local disk; detector GPUs as in tools/extract_team_at_ball.py):
  python tools/derive_match_constants.py derive --work W --devices 2,3 [--data-dir D]
  python tools/derive_match_constants.py pilot --work W --device 2 --shard i --nshards n     (one detector shard)
tools/extract_team_at_ball.py calls derive() itself when <work>/match_constants.json is missing.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import games  # noqa: E402

GAME = games.GAMES[games.GAME]
N_FRAMES = GAME["n_video_frames"]
PITCH = np.array([105.0, 68.0])  # homography maps pitch / (105, 68) to pixels
BLOCKS = tuple((first, last + 1) for _, first, last in games.half_video_blocks(games.GAME))
TEACHER_CONFIG = "configs/submission_contact_carry.json"
TEACHER_SHA256 = "44dd3c69e3c1f87e0dcff84cb57c8dd384af523b777d778bb4b1c43b27d8e73d"
ALIGNMENT_CONFIG = "configs/submission_broadcast_stoppage.json"
ALIGNMENT_KEYS = ("broadcast_minimum_players", "broadcast_offset_samples", "broadcast_search_stride_frames",
                  "broadcast_match_distance_metres", "broadcast_offset_tolerance_frames",
                  "broadcast_minimum_offset_votes", "broadcast_offset_smoothing_frames",
                  "broadcast_maximum_bracket_frames")
EXPECTED_SHA256 = {"masked_windows": "04e94802c2e4c3fed3df390a0f749e13fe661ebaf632f1e9bf42dde725d5d575",
                   "static_cells": "b3d1979b5bd2859852d1ada8905efb446abd7ae1d8214e12e8c4e0cb8cbea88f"}
# name -> (weights under cache/, class id, tile grid (nx, ny)); all imgsz 1280
PILOT_DETECTORS = {
    "L2": ("yolo26l.pt", 32, (2, 2)),
    "L3": ("yolo26l.pt", 32, (3, 3)),
    "X2": ("yolo26x.pt", 32, (2, 2)),
    "S2": ("yolo-sn-ball-opt.pt", 0, (2, 2)),
    "S3": ("yolo-sn-ball-opt.pt", 0, (3, 3)),
}
PILOT_WEIGHTS_SHA256 = {
    "yolo26l.pt": "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26",
    "yolo26x.pt": "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92",
    "yolo-sn-ball-opt.pt": "a3082fb435a8501ae17cfe4ac78e66ca7041205e115feda344d34d1693064f36",
}
PILOT_SEED, PILOT_GAP_SAMPLES, PILOT_CONTROL_SAMPLES, PILOT_MIN_GAP = 20271115, 540, 300, 125
WINDOW_SEED, WINDOW_MIN_GAP, WINDOW_MIN_RUN, WINDOW_MARGIN = 2027, 125, 150, 10


def compact_sha256(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def block_mask(n=N_FRAMES):
    m = np.zeros(n, dtype=bool)
    for a, e in BLOCKS:
        m[a:e] = True
    return m


def half_index(frames):
    """1-based half of each frame (latest half that started; frames before kick-off -> 1)."""
    starts = np.array([a for a, _ in BLOCKS])
    return np.maximum(np.searchsorted(starts, np.asarray(frames), side="right"), 1)


def episodes(mask, minimum):
    from possession.decoder import _episodes
    return _episodes(mask, minimum)


# ---------------------------------------------------------------- 3. orientation
def home_left_frames(pos, team, pitch=None):
    """(N,) bool: home defends the x < length/2 goal, from each half's goalkeeper-side formation."""
    from possession.ballfree import attack_directions
    pitch = GAME["pitch"] if pitch is None else pitch
    direction = attack_directions(pos[:, team == 0], pos[:, team == 1], BLOCKS, pitch)
    per_half = np.array([direction[a] for a, _ in BLOCKS])
    return per_half[half_index(np.arange(len(pos))) - 1] > 0


# ---------------------------------------------------------------- 1. masked windows
def teacher_labels(work, data_dir=None):
    out = Path(work) / "match_constants" / "teacher.csv"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name("teacher.part.csv")
        cmd = [sys.executable, "-m", "possession.cli", "--config", str(REPO / TEACHER_CONFIG), "--output", str(part)]
        if data_dir:
            cmd += ["--data-dir", str(data_dir)]
        subprocess.run(cmd, cwd=str(REPO), check=True, stdout=subprocess.DEVNULL)
        part.replace(out)
    import pandas as pd
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    return pd.read_csv(out)["label"].to_numpy(), digest


def masked_windows(ball, labels):
    """[[start, stop, 'd'|'t']] swap-test windows from observed-ball live runs (see module docstring)."""
    observed = np.isfinite(ball).all(axis=1)
    live = block_mask(len(ball)) & np.isin(labels, ["H", "A"])
    gaps = [(s, e) for s, e in episodes(~observed, WINDOW_MIN_GAP) if live[s:e].mean() >= 0.5]
    lengths = np.asarray([e - s for s, e in gaps])
    rng = np.random.default_rng(WINDOW_SEED)
    windows = []
    for start, stop in episodes(observed & live, WINDOW_MIN_RUN):
        length = int(min(rng.choice(lengths), stop - start - 2 * WINDOW_MARGIN))
        if length < WINDOW_MIN_GAP:
            continue
        first = start + (stop - start - length) // 2
        windows.append([int(first), int(first + length), "d" if half_index(first) == 1 else "t"])
    return windows


# ---------------------------------------------------------------- 2. static clutter cells
def broadcast_reference(tactical_xy, tactical_team, data_dir=None):
    """Per tactical frame: broadcast view state and whether a mapped broadcast ball exists."""
    from possession.broadcast import align_broadcast, load_player_array
    from possession.tracking import load_raw_ball, load_tracking_metadata
    from possession.video import load_homography_quality
    selection = json.loads((REPO / ALIGNMENT_CONFIG).read_text())
    config = {key: selection[key] for key in ALIGNMENT_KEYS}
    meta_path, raw_path, homography_path = games.require(["atd_meta", "atd_raw", "homography"], data_dir=data_dir,
                                                         source="broadcast")
    meta = load_tracking_metadata(meta_path)
    # float32 positions, as the tactical arrays of tools/extract_team_at_ball.py load_tracking
    broadcast_xy = load_player_array(raw_path, meta.team_of_track, GAME["pitch"]).astype(np.float32).astype(np.float64)
    _, valid = load_homography_quality(homography_path, len(broadcast_xy))
    mapped, state, _ = align_broadcast(tactical_xy, tactical_team, broadcast_xy, meta.team_of_track, valid, config)
    ball, _ = load_raw_ball(raw_path, len(broadcast_xy), GAME["pitch"])
    inverse = np.full(len(tactical_xy), -1, dtype=np.int64)
    ok = mapped >= 0
    inverse[mapped[ok]] = np.flatnonzero(ok)
    has_ball = np.zeros(len(tactical_xy), dtype=bool)
    has_ball[inverse >= 0] = np.isfinite(ball[inverse[inverse >= 0]]).all(axis=1)
    return state, has_ball


def pilot_frames(ball, players, state, has_ball):
    """Sorted frames of the stratified visibility-pilot sample (gap + control frames)."""
    from possession.broadcast import LIVE_VIEW
    rng = np.random.default_rng(PILOT_SEED)
    n = len(ball)
    observed = np.isfinite(ball).all(axis=1)
    graded = block_mask(n)
    half = half_index(np.arange(n))
    has_ref = has_ball & (state == LIVE_VIEW)
    bstate = lambda f: np.where(has_ref[f], "live_ref", np.where(state[f] == LIVE_VIEW, "live_noref", "not_live"))
    crowd = lambda f: np.where(players[f] >= 20, "p>=20", "p<20")
    gaps = episodes(~observed & graded, PILOT_MIN_GAP)
    gap_id = np.full(n, -1)
    for i, (a, e) in enumerate(gaps):
        gap_id[a:e] = i
    starts, stops = np.array([a for a, _ in gaps]), np.array([e for _, e in gaps])
    frames = np.flatnonzero(gap_id >= 0)
    gid = gap_id[frames]
    length = (stops - starts)[gid]
    band = np.where(length < 250, "5-10s", np.where(length < 750, "10-30s", ">30s"))
    edge = np.where((frames - starts[gid] < 25) | (stops[gid] - frames <= 25), "edge", "interior")
    cells = np.array(["h{}|{}|{}|{}|{}".format(*c) for c in zip(half[frames], band, edge, bstate(frames),
                                                                 crowd(frames))])
    uniq, counts = np.unique(cells, return_counts=True)
    weight = np.sqrt(counts) * np.array([2.0 if "live_ref" in u else 1.0 for u in uniq])
    alloc = np.minimum(np.maximum(3, np.round(PILOT_GAP_SAMPLES * weight / weight.sum())).astype(int), counts)
    chosen = set()
    for cell, k in zip(uniq, alloc):
        picked = []
        for f in rng.permutation(frames[cells == cell]):
            if all(abs(f - x) >= 10 for x in picked):
                picked.append(int(f))
            if len(picked) == k:
                break
        chosen.update(picked)
    window = np.convolve(observed.astype(int), np.ones(11, dtype=int), "same") == 11
    cand = np.flatnonzero(window & graded)
    ccells = np.array(["h{}|{}|{}".format(*c) for c in zip(half[cand], bstate(cand), crowd(cand))])
    cu = np.unique(ccells)
    per = int(np.ceil(PILOT_CONTROL_SAMPLES / len(cu)))
    for cell in cu:
        pool = cand[ccells == cell]
        chosen.update(int(f) for f in rng.choice(pool, size=min(per, len(pool)), replace=False))
    return sorted(chosen)


def static_cells(cands, frames, min_share=0.04):
    """Input-only clutter map: 1 m pitch cells where candidates recur across many sample frames and blocks."""
    hits = defaultdict(set)
    for f in frames:
        seen = set()
        for name in PILOT_DETECTORS:
            for conf, *_px, xy in cands[f][name]:
                if conf >= 0.05 and xy is not None and abs(xy[0]) < 300 and abs(xy[1]) < 300:
                    seen.add((int(np.floor(xy[0])), int(np.floor(xy[1]))))
        for cell in seen:
            hits[cell].add(f)
    hot = []
    for (cx, cy), fs in hits.items():
        neigh = [(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        pooled = set().union(*(hits.get(c, set()) for c in neigh))
        blocks = {int(f) // (25 * 600) for f in pooled}
        if (len(pooled) >= min_share * len(frames) and len(blocks) >= 3
                and len(fs) >= 0.5 * max(len(hits.get(c, ())) for c in neigh)):
            hot.append([cx + 0.5, cy + 0.5, len(pooled)])
    return sorted(hot)


def pilot_inputs(work, data_dir=None):
    """Pilot frames, cached in <work>/match_constants/pilot_frames.json."""
    path = Path(work) / "match_constants" / "pilot_frames.json"
    if not path.exists():
        import extract_team_at_ball as X
        path.parent.mkdir(parents=True, exist_ok=True)
        ball = X.load_ball(data_dir)
        pos, team, _ = X.load_tracking(data_dir)
        state, has_ball = broadcast_reference(pos, team, data_dir)
        frames = pilot_frames(ball, np.isfinite(pos).all(axis=2).sum(axis=1), state, has_ball)
        path.write_text(json.dumps(frames))
    return json.loads(path.read_text())


def stage_pilot(work, device, shard, nshards, data_dir=None, cache=None):
    """Run the five pilot detectors on one shard of the pilot frames."""
    import extract_team_at_ball as X
    frames = pilot_inputs(work, data_dir)
    todo = frames[shard * len(frames) // nshards:(shard + 1) * len(frames) // nshards]
    cache = Path(cache or REPO / "cache")
    for name, digest in PILOT_WEIGHTS_SHA256.items():
        if X.sha256(cache / name) != digest:
            raise SystemExit("{} sha256 mismatch (run tools/fetch_video_weights.py)".format(cache / name))
    mats = X.load_homographies(data_dir)
    models = {w: X.load_yolo(cache / w) for w in sorted({w for w, _, _ in PILOT_DETECTORS.values()})}
    dev = X.device_arg(device)
    out, t0 = {}, time.time()
    for f, image in X.Video(X.data_paths(data_dir)["video"]).frames(todo):
        out[str(f)] = {name: X.detect_balls(image, models[w], dev, mats[f], class_id=c, grid=g)
                       for name, (w, c, g) in PILOT_DETECTORS.items()}
    path = Path(work) / "match_constants" / "pilot_{}_{}.json".format(shard, nshards)
    path.write_text(json.dumps(out))
    print("pilot", shard, len(out), "{:.0f}s".format(time.time() - t0), flush=True)


def run_pilot(work, devices, data_dir=None, cache=None):
    frames = pilot_inputs(work, data_dir)
    root = Path(work) / "match_constants"
    cands = {}
    for p in sorted(root.glob("pilot_*_*.json")):
        cands.update({int(f): v for f, v in json.loads(p.read_text()).items()})
    if not all(f in cands for f in frames):
        for p in root.glob("pilot_*_*.json"):
            p.unlink()
        n = len(devices)
        procs = []
        for i, d in enumerate(devices):
            cmd = [sys.executable, str(Path(__file__).resolve()), "pilot", "--work", str(work), "--device", str(d),
                   "--shard", str(i), "--nshards", str(n)] + (["--data-dir", str(data_dir)] if data_dir else []) + (
                ["--cache", str(cache)] if cache else [])
            procs.append(subprocess.Popen(cmd, stdout=open(root / "log_pilot_{}.txt".format(i), "w"),
                                          stderr=subprocess.STDOUT))
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit("pilot detector shards failed: {} (see {}/log_pilot_*.txt)".format(codes, root))
        cands = {}
        for p in sorted(root.glob("pilot_*_*.json")):
            cands.update({int(f): v for f, v in json.loads(p.read_text()).items()})
    return cands, frames


# ---------------------------------------------------------------- driver
def derive(work, devices=("0",), data_dir=None, cache=None):
    """Derive (or load) <work>/match_constants.json: masked_windows, static_cells and provenance."""
    path = Path(work) / "match_constants.json"
    if path.exists():
        return json.loads(path.read_text())
    import extract_team_at_ball as X
    t0 = time.time()
    (Path(work) / "match_constants").mkdir(parents=True, exist_ok=True)
    labels, teacher_sha = teacher_labels(work, data_dir)
    windows = masked_windows(X.load_ball(data_dir), labels)
    cands, frames = run_pilot(work, [str(d) for d in devices], data_dir, cache)
    cells = static_cells(cands, frames)
    hashes = {"masked_windows": compact_sha256(windows), "static_cells": compact_sha256(cells)}
    out = {"masked_windows": windows, "static_cells": cells, "provenance": {
        "teacher_config": TEACHER_CONFIG, "teacher_csv_sha256": teacher_sha,
        "teacher_csv_matches_expected": teacher_sha == TEACHER_SHA256, "pilot_frames": len(frames),
        "sha256": hashes, "matches_previous_values": {k: hashes[k] == EXPECTED_SHA256[k] for k in hashes},
        "seconds": round(time.time() - t0, 1)}}
    for key, ok in out["provenance"]["matches_previous_values"].items():
        if not ok:
            print("WARNING: derived {} differ from the values behind the committed evidence (sha256 {} != {})".format(
                key, hashes[key], EXPECTED_SHA256[key]), file=sys.stderr, flush=True)
    path.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("derive", "pilot"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--data-dir", default=None, help="provided data bundle (default: games.DATA_DIR)")
    ap.add_argument("--cache", default=None, help="detector weights dir (default <repo>/cache)")
    ap.add_argument("--devices", default="0", help="derive: comma-separated GPU indices for the pilot detectors")
    ap.add_argument("--device", default="0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    Path(args.work).mkdir(parents=True, exist_ok=True)
    if args.stage == "pilot":
        stage_pilot(args.work, args.device, args.shard, args.nshards, args.data_dir, args.cache)
    else:
        out = derive(args.work, [d.strip() for d in args.devices.split(",") if d.strip()], args.data_dir, args.cache)
        print(json.dumps(out["provenance"]))


if __name__ == "__main__":
    main()
