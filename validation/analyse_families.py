"""Tables for proxy_eval_families.py eval outputs: paired deltas (candidate - baseline) per
family / match / seed, spell counts, and touch-statistic fidelity of each family to the challenge.

Per family: mean F1 of both configs, mean delta, per-match delta (mean and seed range), per-seed
mean delta and its spread, matches / match-seed pairs improved, delta excluding J03WOY, matched /
spurious / missed counts summed over IoU 0.3..0.7, and the fit_artifact.py touch statistics and
loss (lower = closer to the challenge).

Usage: python validation/analyse_families.py EVAL_JSON [--baseline submission_tracked_ballfree]
           [--candidate submission_ball_aware]      (names = config file stems)
"""
import argparse
import json

import numpy as np

from fit_artifact import loss, stats


def tables(path, A="submission_tracked_ballfree", B="submission_ball_aware"):
    res = json.load(open(path))["results"]
    out = {}
    for fam in ("clean", "mask", "drift", "artifact"):
        rs = [r for r in res if r["family"] == fam]
        if not rs:
            continue
        matches = sorted({r["match"] for r in rs}); seeds = sorted({r["cs"] for r in rs})
        D = np.array([[next(r for r in rs if r["match"] == m and r["cs"] == s)["configs"][B]["f1"] -
                       next(r for r in rs if r["match"] == m and r["cs"] == s)["configs"][A]["f1"] for s in seeds] for m in matches])
        F = {k: np.array([[next(r for r in rs if r["match"] == m and r["cs"] == s)["configs"][k]["f1"] for s in seeds] for m in matches]) for k in (A, B)}
        counts = {}
        for k, name in ((A, "baseline"), (B, "candidate")):
            c = {x: int(sum(r["configs"][k]["per_iou"][i][x] for r in rs for i in r["configs"][k]["per_iou"])) for x in ("matched", "spurious", "missed")}
            c["predicted_live_spells"] = int(sum(r["configs"][k]["predicted_live_spells"] for r in rs))
            c["reference_live_spells"] = int(sum(r["configs"][k]["reference_live_spells"] for r in rs))
            counts[name] = c
        seed_means = D.mean(0)
        # matched-pair bootstrap over matches x seeds is weak with 4 matches; report seed spread and match signs
        per_touch = {}
        for s in seeds:
            per_touch[s] = stats({r["match"]: {"rows": [x for x in r["touches"]], "om": r["observed_minutes"]} for r in rs if r["cs"] == s})
        mean_stats = {k: float(np.mean([per_touch[s][k] for s in seeds])) for k in per_touch[seeds[0]]}
        out[fam] = {
            "configs": {"baseline": A, "candidate": B},
            "mean_f1": {"baseline": round(float(F[A].mean()), 4), "candidate": round(float(F[B].mean()), 4)},
            "mean_delta": round(float(D.mean()), 4),
            "per_match_delta_mean_over_seeds": {m: round(float(D[i].mean()), 4) for i, m in enumerate(matches)},
            "per_match_delta_seed_range": {m: [round(float(D[i].min()), 4), round(float(D[i].max()), 4)] for i, m in enumerate(matches)},
            "per_seed_mean_delta": {str(s): round(float(seed_means[j]), 4) for j, s in enumerate(seeds)},
            "seed_spread_of_mean_delta": {"min": round(float(seed_means.min()), 4), "max": round(float(seed_means.max()), 4),
                                          "sd": round(float(seed_means.std(ddof=1)), 4) if len(seeds) > 1 else 0.0},
            "matches_improved": int((D.mean(1) > 0).sum()), "n_matches": len(matches),
            "match_seed_pairs_improved": [int((D > 0).sum()), int(D.size)],
            "mean_delta_excluding_J03WOY": round(float(D[[i for i, m in enumerate(matches) if m != "J03WOY"]].mean()), 4) if "J03WOY" in matches and len(matches) > 1 else None,
            "counts_summed_over_iou_0.3-0.7_matches_seeds": counts,
            "touch_stats_mean_over_seeds": {k: round(v, 4) for k, v in mean_stats.items()},
            "touch_fit_loss_mean_over_seeds": round(float(np.mean([loss(per_touch[s]) for s in seeds])), 2)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_json")
    parser.add_argument("--baseline", default="submission_tracked_ballfree", help="baseline config stem")
    parser.add_argument("--candidate", default="submission_ball_aware", help="candidate config stem")
    args = parser.parse_args()
    print(json.dumps(tables(args.eval_json, args.baseline, args.candidate), indent=1))


if __name__ == "__main__":
    main()
