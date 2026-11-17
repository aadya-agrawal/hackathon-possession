"""Score masked-window swap runs against their unmasked teacher with the official spell F1.

For one split: "whole" = official one-to-one possession-spell F1 (mean over IoU 0.3..0.7,
evaluation.possession_metrics.spell_scores) over the two graded runs; "local" = the same restricted
to the windows +-10 s (250 frames, merged, clipped to the graded runs); "masked agreement" = frame
label agreement inside the masked windows. Only our own outputs are compared (no challenge labels).

Usage:
  python validation/score_swap.py --windows WORK/windows.json --split development \
      --teacher OUT/base_teacher.csv OUT/base_development.csv [RUN.csv ...] [--out scores.json]
Round 11 current default (configs/submission_ball_aware.json, frozen windows): development
0.9292 whole / 0.8105 local / 0.9118 masked agreement; test 0.9474 / 0.8582 / 0.9138.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import common
from swap_windows import HALVES, load_windows

RUNS = [(first, stop) for _, first, stop in HALVES]


def local_runs(windows, n, margin=250):
    loc = []
    for w in windows:
        a, b = max(w["start"] - margin, 0), min(w["stop"] + margin, n)
        if loc and a <= loc[-1][1]:
            loc[-1] = (loc[-1][0], b)
        else:
            loc.append((a, b))
    return [(max(a, s), min(b, e)) for a, b in loc for s, e in RUNS if min(b, e) > max(a, s)]


def score(teacher_csv, run_csv, windows):
    import pandas as pd
    from evaluation.possession_metrics import label_codes, spell_scores
    T = label_codes(pd.read_csv(teacher_csv)["label"]); n = len(T)
    L = label_codes(pd.read_csv(run_csv)["label"])
    m = np.zeros(n, bool)
    for w in windows:
        m[w["start"]:w["stop"]] = True
    return {"whole": round(spell_scores(T, L, RUNS)["mean_f1"], 4),
            "local": round(spell_scores(T, L, local_runs(windows, n))["mean_f1"], 4),
            "masked_agreement": round(float((L[m] == T[m]).mean()), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="masked run CSVs")
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split", required=True, choices=("development", "test"))
    parser.add_argument("--teacher", required=True, help="unmasked run CSV of the same config")
    parser.add_argument("--out", help="optional JSON output")
    parser.add_argument("--repo", help="repository providing evaluation/ (env MTAH_REPO)")
    args = parser.parse_args()
    windows = load_windows(args.windows, args.split)
    teacher = str(Path(args.teacher).resolve())
    runs = [str(Path(r).resolve()) for r in args.runs]
    sys.path.insert(0, str(common.repo_root(args.repo)))
    result = {}
    for run in runs:
        result[Path(run).stem] = s = score(teacher, run, windows)
        print("%-24s whole %.4f | local %.4f | masked agreement %.4f" % (Path(run).stem, s["whole"], s["local"], s["masked_agreement"]))
    if args.out:
        Path(args.out).write_text(json.dumps({"split": args.split, "teacher": teacher, "scores": result}, indent=1))


if __name__ == "__main__":
    main()
