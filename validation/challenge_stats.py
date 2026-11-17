"""Label-free challenge touch statistics on the proxy code path (raw tactical ball, repair only).

Computes the targets that fit_artifact.py matches: contact-carry touch onsets in the graded
frames of the challenge match (raw tactical ball repaired with the config's repair parameters,
no video, no broadcast, no labels), their nearest-player distance band, and whether the team of
the next slow-ball controller agrees. Prints aggregate statistics only.

Usage: python validation/challenge_stats.py BASE_CONFIG --challenge-inputs WORK/challenge_inputs.npz
           [--round4-capture TOUCHES.npy] [--repo WORKTREE]
Round 11 used BASE_CONFIG = configs/submission_tracked_ballfree.json (result identical to the
round-4 capture). --round4-capture optionally re-aggregates an older (frame, team, dist, next) array.
"""
import argparse
import json

import numpy as np

import common
import proxy_eval as P
import proxy_eval_families as E


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_config")
    parser.add_argument("--challenge-inputs", help="challenge_inputs.npz (env CHALLENGE_INPUTS)")
    parser.add_argument("--round4-capture", help="optional earlier touch capture (.npy rows frame, team, dist, next)")
    parser.add_argument("--repo", help="repository/worktree (env MTAH_REPO)")
    args = parser.parse_args()
    inputs = common.env_path("CHALLENGE_INPUTS", args.challenge_inputs)
    common.use_repo(common.repo_root(args.repo))
    from possession.cli import load_config
    from possession.tracking import control_features_from_arrays
    base = load_config(args.base_config)
    c = np.load(inputs)
    ball, pos, team = c["ball"].astype(float), c["pos"].astype(float), c["team"].astype(int)
    g = np.zeros(len(ball), bool)
    for a, b in P.GRADED:
        g[a:b] = True
    tr = P.repair(ball, base, (105.0, 68.0))
    control = control_features_from_arrays(pos[:, team == 0], pos[:, team == 1], tr.xy, 25, base)
    rows = [(f, t, d, P.next_controller(ball, pos, team, f)) for f, t, d in P.touches(control, tr, base) if g[f]]
    om = (np.isfinite(ball).all(1) & g).sum() / 1500
    out = {"proxy_code_path": E.touch_stats(rows, om)}
    if args.round4_capture:
        old = np.load(args.round4_capture)
        old = old[g[old[:, 0].astype(int)]]
        out["round4_capture_file"] = E.touch_stats(old, om)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
