"""Out-of-fold ball-free sequence models used by fast_eval.py (Metrica) and the DFL proxies.

Builds the challenge-like training views once (repo tools/train_ballfree_ball.py recipe: clean +
4 degraded copies per period; degraded = ball drift + frame-aligned challenge segments of
tracked-player counts and raw-ball missing mask), then trains either the full model or a
leave-one-match-out model on the 7 external matches (Metrica Sample Games 1-3, DFL J03WMX, J03WN1,
J03WOH, J03WOY). KIND "ball" trains the ball-aware cnn3 (tools/train_ballfree_ball.export);
KIND "players" trains the players-only model on the same degraded copies with the ball channels
sliced off (the augmented recipe re-run in-fold), so OOF comparisons are apples to apples.
Output weights are named <kind>_ex_<match>.npz by convention (the evaluators' --fold-dir layout).

Usage (export TMPDIR to a local disk):
  python validation/train_folds.py views --work WORK --metrica-root $METRICA_ROOT --dfl-root $DFL_ROOT
  python validation/train_folds.py train NAME KIND [EXCLUDE] --work WORK --out FOLD_DIR \
      [--iters 500] [--seed 0] [--device cuda:0]
Round-11 FOLD_DIR ("models1500"): ball_ex_<m> with --iters 1500, players_ex_<m> with --iters 500,
seed 0, for every m of the 7 matches; e.g. "train ball_ex_J03WMX ball J03WMX --iters 1500".
Env fallbacks: METRICA_ROOT, DFL_ROOT, VALIDATION_CACHE (parsed DFL cache; default WORK/cache),
FOLD_DIR (--out), MTAH_DATA_DIR (challenge bundle for the degradation masks), ITERS, SEED.
"""
import argparse
import json
import os
import pickle
import sys
import time

import common

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8"); os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
TRAIN = common.METRICA_GAMES + common.DFL_DEV
PLAYER_CHANNELS = 32


def load_matches(repo, metrica_root, dfl_root, cache):
    import train_ballfree_ball as tb
    protocol = json.load(open(str(repo / "configs/metrica_core_development.json")))
    out = list(tb.metrica_matches(str(metrica_root), protocol))
    for m in TRAIN[3:]:
        pitch, periods = common.load_dfl_cached(dfl_root, cache, m)
        out.append((m, pitch, [(p["home"], p["away"], p["ball"], p["truth"]) for p in periods]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("views", "train"))
    parser.add_argument("name", nargs="?", help="train: output stem, e.g. ball_ex_J03WMX")
    parser.add_argument("kind", nargs="?", choices=("ball", "players"))
    parser.add_argument("exclude", nargs="?", help="train: match left out (omit for the full model)")
    parser.add_argument("--work", required=True, help="directory for the cached training views")
    parser.add_argument("--out", help="output folder for weights (env FOLD_DIR)")
    parser.add_argument("--metrica-root", help="env METRICA_ROOT")
    parser.add_argument("--dfl-root", help="env DFL_ROOT")
    parser.add_argument("--cache", help="parsed DFL cache (env VALIDATION_CACHE; default WORK/cache)")
    parser.add_argument("--iters", type=int, default=int(os.environ.get("ITERS", 500)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 0)))
    parser.add_argument("--device", default="cuda", help="torch device (round 11: cuda:7)")
    parser.add_argument("--view-workers", type=int, default=24)
    parser.add_argument("--repo", help="repository/worktree (env MTAH_REPO)")
    args = parser.parse_args()
    repo = common.repo_root(args.repo)
    # Resolve every user path before switching the working directory to the repository.
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    if args.mode == "views":
        cache = common.env_path("VALIDATION_CACHE", args.cache, required=False) or os.path.join(work, "cache")
        metrica_root = common.env_path("METRICA_ROOT", args.metrica_root)
        dfl_root = common.env_path("DFL_ROOT", args.dfl_root)
    elif not args.name or not args.kind:
        parser.error("train needs NAME and KIND")
    else:
        out = str(common.env_path("FOLD_DIR", args.out))
    sys.path.insert(0, str(repo / "tools"))
    common.use_repo(repo)
    if args.mode == "views":
        import train_ballfree_ball as tb
        from train_ballfree_augmented import challenge_missing
        t = time.time()
        matches = load_matches(repo, metrica_root, dfl_root, cache)
        views = tb.build_views(matches, 4, challenge_missing(), tb.challenge_ball_seen(), args.view_workers)
        for (name, _, _), v in zip(matches, views):
            pickle.dump(v, open("%s/views_%s.pkl" % (work, name), "wb"), protocol=4)
        print("views", round(time.time() - t), "s; channels", views[0][0][0][0][0].shape[1], flush=True)
    else:
        os.makedirs(out, exist_ok=True)
        from train_ballfree_augmented import train
        import train_ballfree_ball as tb
        from possession.external_ballfree_seq import export
        views = [pickle.load(open("%s/views_%s.pkl" % (work, m), "rb")) for m in TRAIN if m != args.exclude]
        if args.kind == "players":
            views = [[([(H[:, :PLAYER_CHANNELS], A[:, :PLAYER_CHANNELS]) for H, A in v], t) for v, t in ps] for ps in views]
        t = time.time()
        model = train(views, args.iters, args.seed, args.device, 0.5, 4)
        path = "%s/%s.npz" % (out, args.name)
        if args.kind == "ball":
            tb.export(model.state_dict(), path, views[0][0][0][0][0].shape[1] - PLAYER_CHANNELS)
        else:
            export(model.state_dict(), path)
        print(args.name, args.kind, "exclude", args.exclude, "train", round(time.time() - t), "s", flush=True)


if __name__ == "__main__":
    main()
