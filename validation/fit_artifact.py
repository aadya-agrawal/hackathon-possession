"""Score artifact-family grid points against the label-free challenge touch statistics (frozen loss).

Input: one or more "fit" outputs of proxy_eval_families.py. For every grid point the touch rows of
the 4 development matches are aggregated (round-4 definitions: near/far touch rates per observed
minute, next-controller found rate, team agreement per distance band) and compared with the
challenge's own label-free statistics (TARGET, from challenge_stats.py on the rollback config).
Loss = sum of squared standardized errors: rates / 10% of target, found / 0.03, band agreement /
binomial SE of the challenge counts (190 / 87 / 60 touches). Prints the 25 best points and the
reference families (clean / mask / drift).

Usage: python validation/fit_artifact.py FIT_JSON [FIT_JSON ...]
"""
import argparse
import json

import numpy as np

TARGET = {"near_per_min": 5.5087, "far_per_min": 4.9197, "found": 0.5145,
          "agree_0-0.75": 165 / 190, "agree_0.75-1.5": 71 / 87, "agree_1.5-2.5": 38 / 60}
SCALE = {"near_per_min": 0.1 * 5.5087, "far_per_min": 0.1 * 4.9197, "found": 0.03,
         "agree_0-0.75": (165 / 190 * 25 / 190 / 190) ** .5, "agree_0.75-1.5": (71 / 87 * 16 / 87 / 87) ** .5,
         "agree_1.5-2.5": (38 / 60 * 22 / 60 / 60) ** .5}
BANDS = ((0, 0.75), (0.75, 1.5), (1.5, 2.5))


def stats(per_match):
    """per_match: {match: {"rows": [[f, team, dist, nc, live]], "om": minutes}} -> statistics (round-4 aggregation)."""
    rates, found, agree = [], [], {b: [0, 0] for b in BANDS}
    for m, v in per_match.items():
        r = np.asarray(v["rows"], float).reshape(-1, 5); d, nc = r[:, 2], r[:, 3]
        rates.append(((d <= 0.75).sum() / v["om"], ((d > 0.75) & (d <= 2.5)).sum() / v["om"]))
        found.append((nc >= 0).mean())
        for lo, hi in BANDS:
            k = (d > lo) & (d <= hi) & (nc >= 0)
            agree[(lo, hi)][0] += int((r[k, 1] == nc[k]).sum()); agree[(lo, hi)][1] += int(k.sum())
    out = {"near_per_min": float(np.mean([x[0] for x in rates])), "far_per_min": float(np.mean([x[1] for x in rates])),
           "found": float(np.mean(found))}
    for (lo, hi), (a, n) in agree.items():
        out["agree_%g-%g" % (lo, hi)] = a / n if n else float("nan")
    return out


def loss(s):
    return float(sum(((s[k] - TARGET[k]) / SCALE[k]) ** 2 for k in TARGET))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fit_json", nargs="+")
    args = parser.parse_args()
    rows = []
    for path in args.fit_json:
        d = json.load(open(path))
        for tag, per in d["results"].items():
            s = stats(per); rows.append((loss(s), d["grid"][int(tag)], s))
    rows.sort(key=lambda x: x[0])
    for l, g, s in rows[:25] + [r for r in rows if "family" in r[1]]:
        print("%8.2f %-60s %s" % (l, json.dumps(g), " ".join("%s=%.3f" % (k.replace("_per_min", ""), v) for k, v in s.items())))


if __name__ == "__main__":
    main()
