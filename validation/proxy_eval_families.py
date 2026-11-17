"""Paired multi-family, multi-seed challenge-like DFL proxy (round-11 experiment E protocol).

Extends proxy_eval.py:
  * every config gets ITS OWN ball-free probability (own model; own out-of-fold weights on the
    development matches, required in --fold-dir), computed on the SAME degraded inputs in the same
    job, so config comparisons are paired;
  * corruption seed control CS: period seed = 1000*period + crc32(match)%997 + 7919*CS
    (CS 0 = the proxy_eval.py draws for mask/drift);
  * corruption families
      clean    undegraded inputs (run once; seeds irrelevant)
      mask     challenge-replayed missing-player counts (persistent dropout) + raw ball-missing mask
      drift    mask + smooth Gaussian ball offset sigma 1.5 m, tau 1.0 s (= proxy_eval.py SIGMA 1.5)
      artifact mask + parameters from --artifact (default: the frozen fitted parameters in
               artifact_protocol.json: ball low-pass 0.12 s + 0.3 m smooth per-player error), see degrade();
  * label-free touch statistics with the challenge definitions ("fit" mode: no decoding, no scores),
    used by fit_artifact.py to fit the artifact family to the challenge's own touch statistics
    (challenge_stats.py).

Usage:
  python validation/proxy_eval_families.py eval OUT_JSON FAMILY[,FAMILY] CS[,CS] CONFIG[,CONFIG...] \
      --challenge-inputs WORK/challenge_inputs.npz --dfl-root $DFL_ROOT --fold-dir $FOLD_DIR \
      [--matches J03WMX,...] [--artifact '{"ball_smooth_s":0.12,...}'] [--workers N] [--cache DIR]
  python validation/proxy_eval_families.py fit OUT_JSON CS GRID_JSON BASE_CONFIG [common options]
      GRID_JSON: list of artifact parameter dicts (or {"family": "clean|mask|drift"}), e.g. one of
      artifact_protocol.json "fit_grids" (use --grid-stage NAME with GRID_JSON = artifact_protocol.json).
Round-11 E: eval families clean,mask,drift,artifact, CS 0,1,2,3,4, configs
configs/submission_tracked_ballfree.json,configs/submission_ball_aware.json, FOLD_DIR = 1500-iteration
out-of-fold weights; fit CS 100 (stability 101) with configs/submission_tracked_ballfree.json as BASE.
CONFIG paths are relative to the evaluated repository (or absolute). Env fallbacks as proxy_eval.py (CHALLENGE_INPUTS, DFL_ROOT, FOLD_DIR, VALIDATION_CACHE, MTAH_REPO);
MATCHES, ARTIFACT (json) and WORKERS env are also honoured.
"""
import argparse
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import common
import proxy_eval as P
from proxy_eval import GRADED, next_controller, touches

DEV = common.DFL_DEV
BANDS = ((0, 0.75), (0.75, 1.5), (1.5, 2.5))
PROTOCOL = Path(__file__).resolve().parent / "artifact_protocol.json"
_CHALLENGE = {}


def challenge():
    if not _CHALLENGE:
        c = np.load(P.SETTINGS["challenge_inputs"])
        _CHALLENGE["count"] = np.isfinite(c["pos"][..., 0]).sum(1)
        _CHALLENGE["seen"] = np.isfinite(c["ball"]).all(1)
    return _CHALLENGE


def smooth_noise(rng, n, tau):
    from scipy.ndimage import gaussian_filter1d
    off = gaussian_filter1d(rng.normal(0, 1, (n, 2)), tau * 25, axis=0, mode="wrap")
    return off / off.std(axis=0)


def artifact_offset(n, p, rng):
    """Episodic false-track offsets: alternating good/bad episodes (exponential lengths, bad mean
    p['bad_s'] s, bad duty p['duty']); during a bad episode the ball is displaced by one constant vector
    ~ N(0, p['scale']^2 I) (an instantaneous jump in and out); optional smooth jitter p['jitter'] m (tau 0.2 s)."""
    duty, bad, scale = p.get("duty", 0), p.get("bad_s", 1) * 25, p.get("scale", 0)
    off = np.zeros((n, 2)); t = 0 if duty > 0 else n; state = bool(rng.random() < duty)
    good = bad * (1 - duty) / max(duty, 1e-9)
    while t < n:
        length = max(1, int(round(rng.exponential(bad if state else good))))
        if state:
            off[t:t + length] = rng.normal(0, scale, 2)
        t += length; state = not state
    if p.get("jitter", 0) > 0:
        off += smooth_noise(rng, n, 0.2) * p["jitter"]
    return off


def degrade(period, family, seed, sigma=1.5, tau=1.0, art=None):
    rng = np.random.default_rng(seed)
    ball = period["ball"].astype(float).copy(); n = len(ball)
    players = np.concatenate([period["home"], period["away"]], 1).astype(float)
    if family == "clean":
        return ball, players
    if family == "drift":
        ball += smooth_noise(rng, n, tau) * sigma
    c = challenge()
    a, b = GRADED[int(rng.integers(0, 2))]
    start = int(rng.integers(a, max(a + 1, b - n)))
    seg = slice(start, start + n)
    missing = np.clip(22 - c["count"][seg], 0, 22); ball_seen = c["seen"][seg]
    if len(missing) < n:
        missing = np.r_[missing, np.zeros(n - len(missing), int)]; ball_seen = np.r_[ball_seen, np.ones(n - len(ball_seen), bool)]
    present = np.isfinite(players[..., 0])
    current = np.array([], int); key = None
    for t in range(n):
        slots = np.flatnonzero(present[t]); k = min(missing[t], len(slots))
        if key != (k, tuple(slots)):
            current = rng.choice(slots, k, replace=False) if k else np.array([], int); key = (k, tuple(slots))
        players[t, current] = np.nan
    if family == "artifact":
        # separate streams: mask/dropout draws are identical to the mask-only family for the same seed
        if art.get("ball_smooth_s", 0) > 0:
            # tracker low-pass: Gaussian temporal smoothing of the ball track (sigma in seconds)
            from scipy.ndimage import gaussian_filter1d
            ok = np.isfinite(ball).all(1)
            if ok.any():
                filled = np.column_stack([np.interp(np.arange(n), np.flatnonzero(ok), ball[ok, i]) for i in (0, 1)])
                ball = np.where(ok[:, None], gaussian_filter1d(filled, art["ball_smooth_s"] * 25, axis=0, mode="nearest"), np.nan)
        if art.get("duty", 0) > 0 or art.get("jitter", 0) > 0:
            ball += artifact_offset(n, art, np.random.default_rng(seed + 10 ** 7))
        if art.get("player_sigma", 0) > 0:
            # player-relative offsets: independent smooth position error per player track
            prng = np.random.default_rng(seed + 2 * 10 ** 7); p = players.shape[1]
            players += np.stack([smooth_noise(prng, n, art.get("player_tau", 2.0)) for _ in range(p)], 1) * art["player_sigma"]
    ball[~ball_seen] = np.nan
    return ball, players


def prepare(period, base, pitch, ball):
    return P.repair(ball, base, pitch)


def touch_stats(rows, observed_minutes):
    """Label-free statistics with the round-4 challenge definitions; rows = (f, team, dist, next_team)."""
    rows = np.asarray(rows, float).reshape(-1, 4)
    d, nc = rows[:, 2], rows[:, 3]
    out = {"near_per_min": float((d <= 0.75).sum() / observed_minutes),
           "far_per_min": float(((d > 0.75) & (d <= 2.5)).sum() / observed_minutes),
           "found": float((nc >= 0).mean()) if len(rows) else None, "n": int(len(rows))}
    for lo, hi in BANDS:
        m = (d > lo) & (d <= hi) & (nc >= 0)
        out["agree_%g-%g" % (lo, hi)] = [int((rows[m, 1] == nc[m]).sum()), int(m.sum())]
    return out


def match_touches(match, family, cs, art, base_path):
    """Touch rows and observed minutes for one match under one corruption (no decoding)."""
    common.use_repo(P.SETTINGS["repo"])
    from possession.cli import load_config
    from possession.tracking import control_features_from_arrays
    base = load_config(base_path)
    pitch, periods = P.load_match(match)
    rows, om = [], 0.0
    for index, period in enumerate(periods):
        ball, players = degrade(period, family, 1000 * index + zlib.crc32(match.encode()) % 997 + 7919 * cs, art=art)
        tr = prepare(period, base, pitch, ball)
        k = period["home"].shape[1]
        team_of = np.r_[np.zeros(k, int), np.ones(players.shape[1] - k, int)]
        control = control_features_from_arrays(players[:, :k], players[:, k:], tr.xy, 25, base)
        live = np.isin(period["truth"], (0, 1))
        for f, team, dist in touches(control, tr, base):
            rows.append((f, team, dist, next_controller(ball, players, team_of, f), int(live[min(len(live) - 1, f + 12)])))
        om += np.isfinite(ball).all(1).sum() / 1500
    return np.array(rows, float).reshape(-1, 5), om


def fit_job(args):
    match, cs, g, base_path, tag = args
    family = g.get("family", "artifact")
    rows, om = match_touches(match, family, cs, g if family == "artifact" else None, base_path)
    return tag, match, rows.tolist(), om


def eval_job(args):
    match, family, cs, configs, art = args
    common.use_repo(P.SETTINGS["repo"])
    from possession.cli import load_config
    from possession.decoder import build_evidence, decode
    from possession.external_dfl_validation import score_dfl_labels
    from possession.external_transfer_check import restart_only_tags
    from possession.tagging import infer_restart_ownership, localize_restarts
    from possession.tracking import control_features_from_arrays
    from possession.ballfree import configured_home_probability
    pitch, periods = P.load_match(match)
    cfgs = [load_config(c) for c in configs]
    prob_cfgs, weights = [], []
    for cfg in cfgs:
        pc = dict(cfg)
        for kind, key in (("players", "ballfree_weights_path"), ("ball", "ballfree_ball_weights_path")):
            fold = "%s/%s_ex_%s.npz" % (P.SETTINGS["fold_dir"], kind, match)
            if match in DEV and key in pc:
                if not os.path.exists(fold):
                    raise FileNotFoundError(fold)
                pc[key] = fold
        prob_cfgs.append(pc); weights.append({k: v for k, v in pc.items() if "weights_path" in k})
    labels = {c: [] for c in configs}; lengths = []; rows = []; om = 0.0
    for index, period in enumerate(periods):
        ball, players = degrade(period, family, 1000 * index + zlib.crc32(match.encode()) % 997 + 7919 * cs, art=art)
        n = len(ball); lengths.append(n)
        base = cfgs[0]
        tr = prepare(period, base, pitch, ball)
        k = period["home"].shape[1]
        team_of = np.r_[np.zeros(k, int), np.ones(players.shape[1] - k, int)]
        control = control_features_from_arrays(players[:, :k], players[:, k:], tr.xy, 25, base)
        for path, config, pc in zip(configs, cfgs, prob_cfgs):
            prob = configured_home_probability(players, team_of, pitch, 25, [(0, n)], pc, trajectory=tr)
            tags = restart_only_tags([], np.full(n, index + 1, dtype=np.int8), 25, config, None, 2027)
            tags = infer_restart_ownership(localize_restarts(tags, tr, 25, config), tr, control, config)
            ev = build_evidence(control, tr, tags, pitch, config, fps=25, ballfree_home_probability=prob)
            labels[path].append(decode(ev, tags, config))
        live = np.isin(period["truth"], (0, 1))
        for f, team, dist in touches(control, tr, base):
            rows.append((f, team, dist, next_controller(ball, players, team_of, f), int(live[min(n - 1, f + 12)])))
        om += np.isfinite(ball).all(1).sum() / 1500
    truth = np.concatenate([p["truth"] for p in periods])
    out = {"match": match, "family": family, "cs": cs, "weights": weights, "configs": {},
           "touches": np.array(rows, float).reshape(-1, 5).tolist(), "observed_minutes": om}
    for path in configs:
        lab = np.concatenate(labels[path])
        score = score_dfl_labels(lab, truth, lengths)
        sp = score["live_spell_f1"]
        out["configs"][os.path.splitext(os.path.basename(path))[0]] = {
            "f1": float(np.mean([sp[str(v)]["f1"] for v in (.3, .4, .5, .6, .7)])),
            "per_iou": {str(v): sp[str(v)] for v in (.3, .4, .5, .6, .7)},
            "reference_live_spells": score["reference_live_spells"], "predicted_live_spells": score["predicted_live_spells"],
            "live_team_accuracy": score["live_team_accuracy"]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("eval", "fit"))
    parser.add_argument("out_json")
    parser.add_argument("args", nargs="+", help="eval: FAMILIES SEEDS CONFIGS | fit: CS GRID_JSON BASE_CONFIG")
    parser.add_argument("--matches", default=os.environ.get("MATCHES"), help="eval only; default the 4 dev matches")
    parser.add_argument("--artifact", default=os.environ.get("ARTIFACT"),
                        help="artifact parameters as JSON (default: frozen parameters in artifact_protocol.json)")
    parser.add_argument("--grid-stage", help="fit: take the grid from GRID_JSON['fit_grids'][STAGE] (artifact_protocol.json)")
    parser.add_argument("--fold-dir", help="out-of-fold weights (env FOLD_DIR); required for dev matches in eval")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "8")))
    P.add_common_args(parser)
    args = parser.parse_args()
    P.configure(args)
    if args.mode == "eval":
        families, seeds, configs = args.args[0].split(","), [int(x) for x in args.args[1].split(",")], args.args[2].split(",")
        matches = (args.matches or ",".join(DEV)).split(",")
        art = json.loads(args.artifact) if args.artifact else json.loads(PROTOCOL.read_text())["artifact"]
        jobs = [(m, fam, cs, configs, art if fam == "artifact" else None) for fam in families
                for cs in (seeds if fam != "clean" else [0]) for m in matches]
        results = []
        with ProcessPoolExecutor(min(args.workers, len(jobs))) as pool:
            for r in pool.map(eval_job, jobs):
                results.append(r)
                print(r["match"], r["family"], r["cs"], " ".join("%s %.4f" % (k, v["f1"]) for k, v in r["configs"].items()), flush=True)
        json.dump({"artifact": art if "artifact" in families else None, "results": results}, open(args.out_json, "w"))
    else:
        cs, grid_file, base_path = int(args.args[0]), args.args[1], args.args[2]
        grid = json.load(open(grid_file))
        if args.grid_stage:
            grid = grid["fit_grids"][args.grid_stage]
        jobs = [(m, cs, g, base_path, i) for i, g in enumerate(grid) for m in DEV]
        res = {}
        with ProcessPoolExecutor(args.workers) as pool:
            for tag, match, rows, om in pool.map(fit_job, jobs):
                res.setdefault(tag, {})[match] = {"rows": rows, "om": om}
        json.dump({"grid": grid, "results": res}, open(args.out_json, "w"))


if __name__ == "__main__":
    main()
