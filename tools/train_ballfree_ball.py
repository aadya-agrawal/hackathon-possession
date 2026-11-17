"""Train the ball-aware possession cnn3 (players + noisy ball) -> models/ballfree_seq_ball_cnn3.npz.

Same architecture, sampling, clean-only normalisation and optimiser as tools/train_ballfree_augmented.py
(its train() is reused); each view gains the ball channels of possession.ballfree_seq (period_views with a
ball). Labels and positions come from external matches only (Metrica sample games 1-3, DFL J03WMX, J03WN1,
J03WOH, J03WOY). From the challenge match only two per-frame MASKS are used, never positions: the count of
tracked players and whether the raw tactical ball (data/tactical/BAR-ATH_tactical_atd.txt) is present, both
on graded frames.

Degraded copy c of period p of match m: rng = default_rng([crc32(m), p, c, 1]); a ball drift sigma ~ U(0,
--max-sigma) m with Gaussian-smoothed noise (tau 1 s, per-axis std normalised to sigma); independent random
6000-frame graded challenge segments give, frame-aligned, the missing-player counts (players removed as in
the augmented recipe) and the ball-missing mask (ball set to NaN there). The ball (clean or degraded) then
goes through possession.trajectory.repair_ball_trajectory with the submission parameters, as at inference.

Usage (repository root; export TMPDIR=/dev/shm):
  python tools/train_ballfree_ball.py --metrica-root /path/to/metrica-sample-data \\
      --dfl META_J03WMX.xml:POS_J03WMX.xml --dfl ...J03WN1... --dfl ...J03WOH... --dfl ...J03WOY... \\
      --device cuda:7 --out models/ballfree_seq_ball_cnn3.npz
Defaults: --mix-clean 0.5 --copies 4 --max-sigma 2 --iters 1500 --seed 0. Deterministic (fixed seeds,
deterministic torch kernels) on a given GPU model; CPU or other GPUs give slightly different weights.
"""
import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import games  # noqa: E402
from train_ballfree_augmented import BLOCK, challenge_missing, drop_players, graded_blocks  # noqa: E402

# Ball repair parameters of configs/submission_tracked_ballfree.json
REPAIR = dict(max_gap_frames=38, max_speed=65.0, prediction_history_frames=7, prediction_horizon_frames=10,
              max_acceleration=45.0, pitch_margin=2.0)
TAU_SECONDS = 1.0


def challenge_ball_seen(data_dir=None):
    """Per-frame mask: raw tactical ball present in the challenge match (mask only, positions discarded)."""
    from possession.tracking import load_raw_ball
    paths = games.asset_paths(games.GAME, data_dir)
    n = sum(1 for _ in open(paths["atd_raw"]))
    ball, _ = load_raw_ball(paths["atd_raw"], n, games.GAMES[games.GAME]["pitch"])
    return np.isfinite(ball).all(1)


def metrica_matches(root, protocol, games_=("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")):
    """external_ballfree_seq.metrica_matches with the ball: (name, pitch, [(home, away, ball, truth)])."""
    from possession.external_metrica_validation import _load_events, _load_game
    from possession.external_possession_events import possession_chain_references
    for game in games_:
        data = _load_game(Path(root), game)
        events = _load_events(data["event_path"], {
            "PASS", "CARRY", "RECOVERY", "BALL LOST", "BALL OUT", "SHOT", "SET PIECE", "FAULT RECEIVED",
        })
        truth, _ = possession_chain_references(events, data["frames"], data["periods"], data["fps"],
                                               protocol["references"])
        edges = np.r_[0, np.flatnonzero(np.diff(data["periods"])) + 1, len(truth)]
        yield game, (105.0, 68.0), [(data["home"][a:b], data["away"][a:b], data["ball"][a:b], truth[a:b])
                                    for a, b in zip(edges[:-1], edges[1:])]


def dfl_matches(pairs):
    from possession.external_dfl_validation import load_dfl_match
    for metadata, positions in pairs:
        pitch, periods = load_dfl_match(metadata, positions)
        yield (Path(positions).stem.rsplit("DFL-MAT-", 1)[-1], pitch,
               [(p["home"], p["away"], p["ball"], p["truth"]) for p in periods])


def repaired_ball(ball, pitch):
    from possession.trajectory import repair_ball_trajectory
    return repair_ball_trajectory(np.asarray(ball, float), 25, pitch=pitch, **REPAIR).xy


def degrade(home, away, ball, rng, counts, seen, graded, max_sigma):
    """Challenge-like copy: ball drift, frame-aligned challenge player dropout and ball-missing mask."""
    from scipy.ndimage import gaussian_filter1d
    home, away, ball = np.asarray(home, float), np.asarray(away, float), np.asarray(ball, float).copy()
    n, k = len(ball), home.shape[1]
    sigma = rng.uniform(0, max_sigma)
    off = gaussian_filter1d(rng.normal(0, 1, (n, 2)), TAU_SECONDS * 25, axis=0)
    ball += off * (sigma / off.std(axis=0))
    missing, ball_seen, total = [], [], 0
    while total < n:
        a, b = graded[int(rng.integers(0, 2))]
        s = int(rng.integers(a, b - BLOCK))
        missing.append(counts[s:s + BLOCK])
        ball_seen.append(seen[s:s + BLOCK])
        total += BLOCK
    missing, ball_seen = np.concatenate(missing)[:n], np.concatenate(ball_seen)[:n]
    players = drop_players(np.concatenate([home, away], 1), missing, rng)
    ball[~ball_seen] = np.nan
    return players[:, :k], players[:, k:], ball


def views_job(job):
    from possession.ballfree_seq import period_views
    name, pi, pitch, home, away, ball, copy, counts, seen, graded, max_sigma = job
    if copy is not None:
        rng = np.random.default_rng([zlib.crc32(name.encode()), pi, copy, 1])
        home, away, ball = degrade(home, away, ball, rng, counts, seen, graded, max_sigma)
    return period_views(home, away, pitch, 25, repaired_ball(ball, pitch))


def build_views(matches, copies, counts, seen, workers, max_sigma=2.0):
    """Per match: list of periods ([clean, copy 0..copies-1] views, int8 truth), as train() expects."""
    graded = graded_blocks()
    jobs, keys = [], []
    for mi, (name, pitch, periods) in enumerate(matches):
        for pi, (home, away, ball, _) in enumerate(periods):
            for copy in [None] + list(range(copies)):
                jobs.append((name, pi, pitch, home, away, ball, copy, counts, seen, graded, max_sigma))
                keys.append((mi, pi))
    with ProcessPoolExecutor(max(1, min(workers, len(jobs)))) as pool:
        results = list(pool.map(views_job, jobs))
    out = [[([], np.asarray(p[3], np.int8)) for p in periods] for _, _, periods in matches]
    for (mi, pi), views in zip(keys, results):
        out[mi][pi][0].append(views)
    return out


def export(state_dict, path, ball_channels):
    from possession.external_ballfree_seq import export as export_players
    export_players(state_dict, path)
    with np.load(path) as z:
        weights = {key: z[key] for key in z.files}
    np.savez_compressed(path, ball_channels=np.array(ball_channels, dtype=np.int64), **weights)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrica-root", required=True)
    parser.add_argument("--protocol", default="configs/metrica_core_development.json")
    parser.add_argument("--dfl", action="append", default=[], help="METADATA.xml:POSITIONS.xml")
    parser.add_argument("--data-dir", default=None, help="provided data bundle (default: games.DATA_DIR)")
    parser.add_argument("--out", default="models/ballfree_seq_ball_cnn3.npz")
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mix-clean", type=float, default=0.5)
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--max-sigma", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    t0 = time.time()
    out = os.path.abspath(args.out)
    protocol = json.loads(Path(args.protocol).read_text())
    pairs = [[os.path.abspath(p) for p in item.split(":", 1)] for item in args.dfl]
    metrica_root = os.path.abspath(args.metrica_root)
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else None
    os.chdir(REPO)
    matches = list(metrica_matches(metrica_root, protocol)) + list(dfl_matches(pairs))
    views = build_views(matches, args.copies, challenge_missing(data_dir), challenge_ball_seen(data_dir),
                        args.workers, args.max_sigma)
    t1 = time.time()
    import torch
    from train_ballfree_augmented import train
    from possession.ballfree_seq import period_views
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = train(views, args.iters, args.seed, device, args.mix_clean, args.copies)
    players_only = period_views(np.zeros((40, 11, 2)), np.zeros((40, 11, 2)), (105.0, 68.0))[0].shape[1]
    export(model.state_dict(), out, views[0][0][0][0][0].shape[1] - players_only)
    print(json.dumps({"weights": out, "matches": [m[0] for m in matches], "iters": args.iters, "seed": args.seed,
                      "mix_clean": args.mix_clean, "copies": args.copies, "max_sigma": args.max_sigma,
                      "device": device, "views_s": round(t1 - t0, 1),
                      "train_s": round(time.time() - t1, 1)}), flush=True)


if __name__ == "__main__":
    main()
