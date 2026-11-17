"""Masked-window definition for the label-free swap test on the challenge match.

One masked window per tracking-observed live run (>= 150 frames, observed ball and our own
reference submission says H/A), length drawn (rng seed 2027) from the real live-gap length
distribution (unobserved-ball runs >= 125 frames that are >= 50% live in the reference labels)
and capped to leave 10 observed frames each side; windows shorter than 125 frames are dropped.
Half 1 windows form the development split, half 2 a frozen test split. Real live gaps are listed
separately. No challenge labels are used: "live" comes from the reference submission CSV, i.e.
our own pipeline output.

The published round-11 numbers (development 0.9292/0.8105 whole/local for
configs/submission_ball_aware.json) used the frozen windows FROZEN_WINDOWS_SHA256, which are
reproduced exactly with --labels = the contact-carry selection output
(python -m possession.cli --config configs/submission_contact_carry.json --output REF.csv;
CSV sha256 44dd3c69e3c1f87e0dcff84cb57c8dd384af523b777d778bb4b1c43b27d8e73d).
--expect-sha checks the result.

Usage:
  python validation/swap_windows.py --inputs WORK/challenge_inputs.npz \
      --labels submissions/<reference>.csv --out WORK/windows.json [--expect-sha]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
HALVES = ((1, 11185, 90652), (2, 110808, 190198))
FROZEN_WINDOWS_SHA256 = "6cc1ac33fc5ba48e4be70794b0dde7467d6e8dea48cc0758b4ad34d66cf16453"


def build_windows(inputs_path, labels_path):
    import pandas as pd
    from possession.decoder import _episodes
    inputs = np.load(inputs_path)
    ball = inputs["ball"].astype(float)
    positions = inputs["pos"].astype(float)
    labels = pd.read_csv(labels_path)["label"].to_numpy()
    observed = np.isfinite(ball).all(axis=1)
    graded = np.zeros(len(ball), dtype=bool)
    for _, first, stop in HALVES:
        graded[first:stop] = True
    live = graded & np.isin(labels, ["H", "A"])
    gaps = [(int(s), int(e)) for s, e in _episodes(~observed, 125) if live[s:e].mean() >= 0.5]
    lengths = np.asarray([e - s for s, e in gaps])
    rng = np.random.default_rng(2027)
    windows = []
    for start, stop in _episodes(observed & live, 150):
        length = int(min(rng.choice(lengths), stop - start - 20))
        if length < 125:
            continue
        first = start + (stop - start - length) // 2
        half = next(h for h, a, z in HALVES if a <= first < z)
        nearest = float(np.nanmin(np.linalg.norm(positions[first - 1] - ball[first - 1], axis=1)))
        windows.append({"start": first, "stop": first + length, "half": half,
                        "split": "development" if half == 1 else "test",
                        "onset_nearest_player_m": nearest})
    real = [{"start": s, "stop": e, "half": next(h for h, a, z in HALVES if a <= s < z)} for s, e in gaps]
    return {"masked_windows": windows, "real_gaps": real}


def load_windows(path, split):
    return [w for w in json.loads(Path(path).read_text())["masked_windows"] if w["split"] == split]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True, help="challenge_inputs.npz (build_challenge_inputs.py)")
    parser.add_argument("--labels", required=True, help="reference submission CSV defining live frames")
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect-sha", action="store_true", help="fail unless the frozen round-11 windows are reproduced")
    args = parser.parse_args()
    out = build_windows(args.inputs, args.labels)
    text = json.dumps(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    windows = out["masked_windows"]
    for split in ("development", "test"):
        part = [w for w in windows if w["split"] == split]
        print(split, "windows", len(part), "frames", sum(w["stop"] - w["start"] for w in part),
              "onset player <=1 m", round(np.mean([w["onset_nearest_player_m"] <= 1 for w in part]), 2))
    print("real gaps", len(out["real_gaps"]), "frames", sum(g["stop"] - g["start"] for g in out["real_gaps"]))
    print("windows sha256", sha, "(frozen round-11:", FROZEN_WINDOWS_SHA256, ")")
    if args.expect_sha and sha != FROZEN_WINDOWS_SHA256:
        sys.exit("windows differ from the frozen round-11 set: wrong --labels reference?")


if __name__ == "__main__":
    main()
