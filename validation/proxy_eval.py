"""Challenge-like DFL proxy (truth known) with touch-level diagnostics (round-10 protocol).

DFL matches carry provider possession flags, so possession can be scored on them. To make them
look like the challenge, each period is degraded with inputs replayed from one random graded
challenge segment so the corruptions co-occur as on the challenge:
  * smooth drifting ball offset (Gaussian-filtered noise, std SIGMA m, correlation TAU s; SIGMA 0 = off),
  * the challenge's per-frame missing-player counts (persistent dropout of random tracked players),
  * the challenge's raw ball-missing mask.
Per-period seed = 1000*period + crc32(match) % 997. Ball-free probabilities are computed on the
degraded inputs; for development matches with out-of-fold weights in --fold-dir
(<kind>_ex_<match>.npz, kind players|ball; train_folds.py) those replace the config's weights,
otherwise the config's own weights are used (held-out matches). Every config is decoded on the
same degraded inputs with the FIRST config's probability. Score: mean live-spell F1 over IoU
0.3..0.7 (possession.external_dfl_validation.score_dfl_labels).

Writes OUT_DIR/<match>.npz (truth, lengths, labels_<config stem>, touches, probability, source)
and prints "<match> <config> <f1>". NODEGRADE=1 (or --no-degrade) evaluates clean inputs.

Usage:
  python validation/proxy_eval.py OUT_DIR SIGMA TAU CONFIG[,CONFIG...] \
      --challenge-inputs WORK/challenge_inputs.npz --dfl-root $DFL_ROOT --fold-dir $FOLD_DIR \
      [--matches J03WMX,...] [--cache DIR] [--workers N] [--repo WORKTREE]
Round-10/11 numbers: SIGMA 0 or 1.5, TAU 1, dev matches J03WMX,J03WN1,J03WOH,J03WOY (default),
held-out J03WPY,J03WQQ,J03WR9. CONFIG paths are relative to the evaluated repository (or absolute). Env fallbacks: CHALLENGE_INPUTS, DFL_ROOT, FOLD_DIR,
VALIDATION_CACHE, MTAH_REPO, STRESS_MATCHES, NODEGRADE.
"""
import argparse
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common

GRADED = ((11185, 90652), (110808, 190198))
RADIUS = 2.5  # touches are collected up to this distance; bands are split afterwards
SETTINGS = {"nodegrade": bool(os.environ.get("NODEGRADE"))}


def set_challenge_inputs(path):
    SETTINGS["challenge_inputs"] = str(path)


def degrade(period, sigma, tau, seed):
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(seed)
    ball = period["ball"].astype(float).copy(); n = len(ball)
    players = np.concatenate([period["home"], period["away"]], 1).astype(float)
    if SETTINGS["nodegrade"]:
        return ball, players
    if sigma > 0:
        off = gaussian_filter1d(rng.normal(0, 1, (n, 2)), tau * 25, axis=0, mode="wrap")
        ball += off * sigma / off.std(axis=0)
    c = np.load(SETTINGS["challenge_inputs"])
    count = np.isfinite(c["pos"][..., 0]).sum(1); seen = np.isfinite(c["ball"]).all(1)
    a, b = GRADED[int(rng.integers(0, 2))]
    start = int(rng.integers(a, max(a + 1, b - n)))
    seg = slice(start, start + n)
    missing = np.clip(22 - count[seg], 0, 22); ball_seen = seen[seg]
    if len(missing) < n:
        missing = np.r_[missing, np.zeros(n - len(missing), int)]; ball_seen = np.r_[ball_seen, np.ones(n - len(ball_seen), bool)]
    present = np.isfinite(players[..., 0])
    current = np.array([], int); key = None
    for t in range(n):
        slots = np.flatnonzero(present[t]); k = min(missing[t], len(slots))
        # Missing players persist: re-draw only when the count or the on-pitch set changes.
        if key != (k, tuple(slots)):
            current = rng.choice(slots, k, replace=False) if k else np.array([], int); key = (k, tuple(slots))
        players[t, current] = np.nan
    ball[~ball_seen] = np.nan
    return ball, players


def touches(control, trajectory, config, radius=RADIUS):
    """Contact-carry touch onsets (frame, team, nearest distance of that team within the window)."""
    from possession.decoder import contact_carry_evidence, _episodes
    _, support, _ = contact_carry_evidence(control, trajectory, 25, {**config, "contact_carry_distance_metres": radius})
    reach = config["contact_carry_support_frames"]; w = config["contact_carry_window_frames"]
    nearest = control.nearest_distances[:, :, 0]
    nearest = np.where(np.isfinite(nearest), nearest, np.inf)
    out = []
    for team in (0, 1):
        for s, e in _episodes(support[:, team], 1):
            f = s + reach if s > 0 else e - reach - 1
            out.append((f, team, float(nearest[max(0, f - w):f + w + 1, team].min())))
    return sorted(out)


def next_controller(ball, players, team_of, frame, horizon=75):
    """Team of the nearest player (<= 2 m) when the observed ball next stays slow (< 3 m/s for 5 frames)."""
    n = len(ball)
    for t in range(frame + 5, min(n - 5, frame + horizon)):
        seg = ball[t:t + 6]
        if not np.isfinite(seg).all():
            continue
        if np.linalg.norm(np.diff(seg, axis=0), axis=1).max() * 25 < 3:
            d = np.linalg.norm(players[t] - ball[t], axis=1)
            if np.isfinite(d).any() and np.nanmin(d) <= 2.0:
                return int(team_of[np.nanargmin(d)])
            return -1
    return -1


def repair(ball, base, pitch):
    from possession.trajectory import repair_ball_trajectory
    return repair_ball_trajectory(ball, 25, max_gap_frames=base["max_interpolation_gap_frames"],
                                  max_speed=base["max_ball_speed_metres_per_second"],
                                  prediction_history_frames=base["prediction_history_frames"],
                                  prediction_horizon_frames=base["prediction_horizon_frames"],
                                  max_acceleration=base["max_ball_acceleration_metres_per_second_squared"],
                                  pitch=pitch, pitch_margin=base["prediction_pitch_margin_metres"])


def load_match(match):
    return common.load_dfl_cached(SETTINGS["dfl_root"], SETTINGS["cache"], match)


def job(args):
    match, configs, sigma, tau = args
    common.use_repo(SETTINGS["repo"])
    from possession.cli import load_config
    from possession.decoder import build_evidence, decode
    from possession.external_dfl_validation import score_dfl_labels
    from possession.external_transfer_check import restart_only_tags
    from possession.tagging import infer_restart_ownership, localize_restarts
    from possession.tracking import control_features_from_arrays
    from possession.ballfree import configured_home_probability
    pitch, periods = load_match(match)
    cfgs = [load_config(c) for c in configs]
    prob_config = dict(cfgs[0])
    fold_dir = SETTINGS["fold_dir"]
    for kind, key in (("players", "ballfree_weights_path"), ("ball", "ballfree_ball_weights_path")):
        fold = "%s/%s_ex_%s.npz" % (fold_dir, kind, match)
        if fold_dir is not None and os.path.exists(fold) and key in prob_config:
            prob_config[key] = fold
    probs, sources = [], []
    labels = {c: [] for c in configs}; lengths = []; touch_rows = []; finger = []
    for index, period in enumerate(periods):
        ball, players = degrade(period, sigma, tau, 1000 * index + zlib.crc32(match.encode()) % 997)
        n = len(ball); lengths.append(n)
        base = cfgs[0]
        tr = repair(ball, base, pitch)
        k = period["home"].shape[1]
        home, away = players[:, :k], players[:, k:]
        team_of = np.r_[np.zeros(k, int), np.ones(players.shape[1] - k, int)]
        control = control_features_from_arrays(home, away, tr.xy, 25, base)
        prob = configured_home_probability(players, team_of, pitch, 25, [(0, n)], prob_config, trajectory=tr)
        probs.append(prob); sources.append(tr.source)
        for path, config in zip(configs, cfgs):
            tags = restart_only_tags([], np.full(n, index + 1, dtype=np.int8), 25, config, None, 2027)
            tags = infer_restart_ownership(localize_restarts(tags, tr, 25, config), tr, control, config)
            ev = build_evidence(control, tr, tags, pitch, config, fps=25, ballfree_home_probability=prob)
            labels[path].append(decode(ev, tags, config))
        truth = period["truth"]
        for f, team, dist in touches(control, tr, base):
            after = truth[min(n - 1, f + 12)]
            touch_rows.append((index, f, team, dist, int(after), next_controller(ball, players, team_of, f)))
        live = np.isin(truth, (0, 1))
        finger.append((live.sum(), np.isfinite(ball).all(1).mean()))
    truth = np.concatenate([p["truth"] for p in periods])
    out = {}
    for path in configs:
        lab = np.concatenate(labels[path])
        score = score_dfl_labels(lab, truth, lengths)
        out[path] = (float(np.mean([score["live_spell_f1"][str(v)]["f1"] for v in (.3, .4, .5, .6, .7)])), lab)
    return match, out, truth, lengths, np.array(touch_rows), finger, np.concatenate(probs), np.concatenate(sources), \
        {k: v for k, v in prob_config.items() if "weights_path" in k}


def configure(args):
    """Fill SETTINGS from parsed CLI args (shared with proxy_eval_families / oracle_breakdown)."""
    SETTINGS.update(repo=common.repo_root(getattr(args, "repo", None)),
                    cache=common.env_path("VALIDATION_CACHE", args.cache),
                    dfl_root=common.env_path("DFL_ROOT", args.dfl_root),
                    fold_dir=common.env_path("FOLD_DIR", getattr(args, "fold_dir", None), required=False))
    set_challenge_inputs(common.env_path("CHALLENGE_INPUTS", args.challenge_inputs))


def add_common_args(parser):
    parser.add_argument("--challenge-inputs", help="challenge_inputs.npz (env CHALLENGE_INPUTS; build_challenge_inputs.py)")
    parser.add_argument("--dfl-root", help="DFL XML folder (env DFL_ROOT)")
    parser.add_argument("--cache", help="cache directory for parsed DFL matches (env VALIDATION_CACHE)")
    parser.add_argument("--repo", help="repository/worktree to evaluate (env MTAH_REPO; default this repo)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("sigma", type=float)
    parser.add_argument("tau", type=float)
    parser.add_argument("configs", help="comma-separated config paths (first one drives repair and probabilities)")
    parser.add_argument("--matches", default=os.environ.get("STRESS_MATCHES", ",".join(common.DFL_DEV)))
    parser.add_argument("--fold-dir", help="out-of-fold weights (env FOLD_DIR)")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--no-degrade", action="store_true", help="clean inputs (same as NODEGRADE=1)")
    add_common_args(parser)
    args = parser.parse_args()
    configure(args)
    SETTINGS["nodegrade"] = SETTINGS["nodegrade"] or args.no_degrade
    matches = args.matches.split(",")
    configs = args.configs.split(",")
    os.makedirs(args.out_dir, exist_ok=True)
    with ProcessPoolExecutor(args.workers or len(matches)) as pool:
        for match, out, truth, lengths, rows, finger, prob, source, weights in pool.map(
                job, [(m, configs, args.sigma, args.tau) for m in matches]):
            save = {"truth": truth, "lengths": lengths, "touches": rows,
                    "live_frames": sum(f[0] for f in finger), "ball_seen": np.mean([f[1] for f in finger]),
                    "probability": prob, "source": source, "weights": json.dumps(weights)}
            for path, (f1, lab) in out.items():
                name = os.path.splitext(os.path.basename(path))[0]
                save["labels_" + name] = lab
                print("%s %-16s %.4f" % (match, name, f1), flush=True)
            np.savez_compressed(os.path.join(args.out_dir, match + ".npz"), **save)


if __name__ == "__main__":
    main()
