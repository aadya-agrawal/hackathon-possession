"""Retrain the ball-free cnn3 with challenge-style player-dropout augmentation -> models/ballfree_seq_cnn3_aug.npz.

Labels and positions come from external matches only (Metrica sample games 1-3, DFL J03WMX, J03WN1, J03WOH,
J03WOY); from the challenge match only the per-frame COUNT of tracked players (data/tactical/
BAR-ATH_tactical_atd.txt, graded frames) is used, to replay realistic dropout. Architecture, features,
clean-only normalisation, iterations, batch/chunk sampling and seeds are those of
possession/external_ballfree_seq.py (build_model, export, metrica_matches, dfl_matches are reused); the only
change is that each sampled training chunk is read from the clean view (p = --mix-clean) or from one of
--copies degraded view copies, drawn from a separate RNG (seed + 7919), so --mix-clean 1 reproduces
models/ballfree_seq_cnn3.npz bit for bit.
Degraded copy c of period p of match m: rng = default_rng([crc32(m), p, c, round(100 * noise)]); optional smooth
position noise (sigma --noise m, ~1 s correlation; 0 for the shipped weights); per-frame missing-player counts
max(0, 22 - tracked) taken from independent random 6000-frame graded challenge segments; per frame that many
random on-pitch players are removed, the removed set persisting until the count or the on-pitch set changes.

Usage (run from the repository root; export TMPDIR=/dev/shm):
  python tools/train_ballfree_augmented.py --metrica-root /path/to/metrica-sample-data \\
      --dfl /path/DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml:/path/DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml \\
      --dfl ...J03WN1 pair... --dfl ...J03WOH pair (DFL-COM-000002)... --dfl ...J03WOY pair (DFL-COM-000002)... \\
      --device cuda:5 --out models/ballfree_seq_cnn3_aug.npz
DFL pairs are used in argument order (the shipped weights: J03WMX, J03WN1, J03WOH, J03WOY); the match name used
for seeding is the DFL-MAT id in the positions file name. Defaults: --mix-clean 0.5 --copies 4 --noise 0
--iters 500 --seed 0. Degraded views are built in --workers processes (~3-4 min with 16), training ~1-2 min.
Determinism: with torch 2.4.1+cu121 on an NVIDIA L40S (driver 550.90.12, CUDA_DEVICE_ORDER=PCI_BUS_ID, physical
GPU index in --device) the weights are bit-identical across GPUs and runs (deterministic kernels, fixed seeds);
CPU training or other GPU models give slightly different floating-point weights.
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
import games  # noqa: E402
from possession.external_ballfree_seq import dfl_matches, metrica_matches  # noqa: E402

BLOCK = 6000  # an independent challenge segment of missing-player counts every 4 min
FULL_TEAMS = 22


def challenge_missing(data_dir=None):
    """Per-frame number of missing tracked players in the challenge match (counts only, no positions kept)."""
    from possession.broadcast import load_player_array
    from possession.tracking import load_tracking_metadata
    paths = games.asset_paths(games.GAME, data_dir)
    meta = load_tracking_metadata(paths["atd_meta"])
    pos = load_player_array(paths["atd_raw"], meta.team_of_track, games.GAMES[games.GAME]["pitch"])
    return np.clip(FULL_TEAMS - np.isfinite(pos[..., 0]).sum(1), 0, FULL_TEAMS)


def graded_blocks():
    return [(first, last + 1) for _, first, last in games.half_video_blocks(games.GAME)]


def drop_players(players, missing, rng):
    # Per frame remove `missing[t]` random on-pitch players; the removed set persists until the
    # count or the on-pitch set changes
    players = players.copy()
    present = np.isfinite(players[..., 0])
    current, key = np.array([], int), None
    for t in range(len(players)):
        slots = np.flatnonzero(present[t])
        k = min(int(missing[t]), len(slots))
        if key != (k, tuple(slots)):
            current = rng.choice(slots, k, replace=False) if k else np.array([], int)
            key = (k, tuple(slots))
        players[t, current] = np.nan
    return players


def train_missing(n, rng, counts, graded):
    # Concatenate independent random graded challenge segments of BLOCK frames
    out, total = [], 0
    while total < n:
        a, b = graded[int(rng.integers(0, 2))]
        s = int(rng.integers(a, b - BLOCK))
        out.append(counts[s:s + BLOCK])
        total += BLOCK
    return np.concatenate(out)[:n]


def add_noise(players, sigma, rng, tau=1.0):
    # Smooth per-player position noise (sigma metres, ~tau s correlation)
    from scipy.ndimage import gaussian_filter1d
    if sigma <= 0:
        return players
    off = gaussian_filter1d(rng.normal(0, 1, players.shape), tau * 25, axis=0)
    return players + off * (sigma / off.std())


def views_job(job):
    # One period: the clean view (copy None) or degraded copy `copy`
    from possession.ballfree_seq import period_views
    name, pi, pitch, home, away, copy, noise, counts, graded = job
    if copy is not None:
        home, away = np.asarray(home, float), np.asarray(away, float)
        k = home.shape[1]
        players = np.concatenate([home, away], 1)
        rng = np.random.default_rng([zlib.crc32(name.encode()), pi, copy, int(round(noise * 100))])
        players = drop_players(add_noise(players, noise, rng), train_missing(len(players), rng, counts, graded), rng)
        home, away = players[:, :k], players[:, k:]
    return period_views(home, away, pitch, 25)


def build_views(matches, copies, noise, counts, workers):
    """Per match: list of periods (views per variant [clean, copy 0..copies-1], int8 truth)."""
    graded = graded_blocks()
    jobs, keys = [], []
    for mi, (name, pitch, periods) in enumerate(matches):
        for pi, (home, away, _) in enumerate(periods):
            for copy in [None] + list(range(copies)):
                jobs.append((name, pi, pitch, home, away, copy, noise, counts if copy is not None else None, graded))
                keys.append((mi, pi))
    with ProcessPoolExecutor(max(1, min(workers, len(jobs)))) as pool:
        results = list(pool.map(views_job, jobs))
    out = [[([], np.asarray(t, np.int8)) for _, _, t in periods] for _, _, periods in matches]
    for (mi, pi), views in zip(keys, results):
        out[mi][pi][0].append(views)
    return out


def train(matches, iters, seed, device, mix_clean, copies):
    """external_ballfree_seq.train with a per-chunk choice between the clean and degraded view copies."""
    import torch
    import torch.nn as nn
    from possession import AWAY, HOME
    from possession.external_ballfree_seq import BATCH, CHUNK, STEP, build_model
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if mix_clean >= 1:
        matches = [[(views[:1], truth) for views, truth in ps] for ps in matches]
    # Normalisation from clean views only, in the shipped concatenation order
    allx = np.concatenate([np.concatenate([np.concatenate([v[0][0] for v, _ in ps]), np.concatenate([v[0][1] for v, _ in ps])])
                           for ps in matches])
    mean, std = allx.mean(0), allx.std(0) + 1e-3
    seqs = []
    for ps in matches:
        for views, truth in ps:
            b = len(truth)
            for o in range(STEP):
                t = truth[o:b:STEP]
                seqs.append(([(H[o:b:STEP], A[o:b:STEP]) for H, A in views], (t == HOME).astype(np.float32),
                             np.isin(t, (HOME, AWAY)).astype(np.float32)))
    lengths = np.array([len(s[1]) for s in seqs], float)
    rng = np.random.default_rng(seed)
    vrng = np.random.default_rng(seed + 7919)
    model = build_model(allx.shape[1], mean, std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=iters, pct_start=0.1)
    T = lambda x: torch.as_tensor(x, device=device)  # noqa: E731
    tensors = [([(T(H), T(A)) for H, A in v], T(y), T(m)) for v, y, m in seqs]
    for _ in range(iters):
        idx = rng.choice(len(seqs), BATCH, p=lengths / lengths.sum())
        starts = [rng.integers(0, max(1, len(seqs[i][1]) - CHUNK)) for i in idx]
        ver = [0] * BATCH if mix_clean >= 1 else \
            [0 if vrng.random() < mix_clean else 1 + int(vrng.integers(0, copies)) for _ in idx]
        H = torch.stack([tensors[i][0][v][0][s:s + CHUNK] for i, s, v in zip(idx, starts, ver)])
        A = torch.stack([tensors[i][0][v][1][s:s + CHUNK] for i, s, v in zip(idx, starts, ver)])
        y = torch.stack([tensors[i][1][s:s + CHUNK] for i, s in zip(idx, starts)])
        m = torch.stack([tensors[i][2][s:s + CHUNK] for i, s in zip(idx, starts)])
        loss = (nn.functional.binary_cross_entropy_with_logits(model(H, A), y, reduction="none") * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrica-root", required=True)
    parser.add_argument("--protocol", default="configs/metrica_core_development.json")
    parser.add_argument("--dfl", action="append", default=[], help="METADATA.xml:POSITIONS.xml")
    parser.add_argument("--data-dir", default=None, help="provided data bundle (default: games.DATA_DIR)")
    parser.add_argument("--out", default="models/ballfree_seq_cnn3_aug.npz")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mix-clean", type=float, default=0.5)
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=16, help="processes for the view features")
    parser.add_argument("--device", default=None, help="e.g. cuda:5 (physical index) or cpu (default: cuda)")
    args = parser.parse_args(argv)
    t0 = time.time()
    out = os.path.abspath(args.out)
    protocol = json.loads(Path(args.protocol).read_text())
    pairs = [[os.path.abspath(p) for p in item.split(":", 1)] for item in args.dfl]
    metrica_root = os.path.abspath(args.metrica_root)
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else None
    os.chdir(REPO)
    matches = list(metrica_matches(metrica_root, protocol))
    matches += [(name.rsplit("DFL-MAT-", 1)[-1], pitch, periods) for name, pitch, periods in dfl_matches(pairs)]
    counts = challenge_missing(data_dir) if args.mix_clean < 1 else None
    views = build_views(matches, args.copies if args.mix_clean < 1 else 0, args.noise, counts, args.workers)
    t1 = time.time()
    import torch
    from possession.external_ballfree_seq import export
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = train(views, args.iters, args.seed, device, args.mix_clean, args.copies)
    export(model.state_dict(), out)
    print(json.dumps({"weights": out, "matches": [m[0] for m in matches], "iters": args.iters, "seed": args.seed,
                      "mix_clean": args.mix_clean, "copies": args.copies, "noise": args.noise, "device": device,
                      "gpu": torch.cuda.get_device_name(torch.device(device)) if str(device).startswith("cuda") else None,
                      "views_s": round(t1 - t0, 1), "train_s": round(time.time() - t1, 1)}), flush=True)


if __name__ == "__main__":
    main()
