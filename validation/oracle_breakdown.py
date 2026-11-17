"""Oracle headroom on the challenge-like DFL proxy (truth known): which error class costs most.

Reads a proxy_eval.py output directory and, per match, replaces one error class of a config's
labels with the truth and reports the spell-F1 gain (official spell_scores, IoU 0.3..0.7, per
DFL period):
  oracle dead/live            fix dead/live errors (true dead -> D, missed live -> true team)
  oracle team, ball unseen    fix team on live frames where the degraded ball is missing
  oracle team, ball seen      fix team on live frames where the degraded ball is observed
  oracle team (all live)      fix team on all frames both truth and prediction call live
plus error-rate fractions. The degraded ball-seen mask is regenerated with proxy_eval.degrade and
the same seeds, so SIGMA/TAU must match the proxy run. Prints a table (external DFL data only).

Usage: python validation/oracle_breakdown.py PROXY_DIR SIGMA TAU CONFIG_STEM \
           --challenge-inputs WORK/challenge_inputs.npz --dfl-root $DFL_ROOT [--cache DIR]
"""
import argparse
import glob
import os
import zlib

import numpy as np

import proxy_eval as P


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("proxy_dir")
    parser.add_argument("sigma", type=float)
    parser.add_argument("tau", type=float)
    parser.add_argument("name", help="config stem used in the proxy run (labels_<name>)")
    P.add_common_args(parser)
    args = parser.parse_args()
    P.configure(args)
    proxy_dir = os.path.abspath(args.proxy_dir)
    import common
    common.use_repo(P.SETTINGS["repo"])
    from evaluation.possession_metrics import spell_scores
    rows = []
    for f in sorted(glob.glob(os.path.join(proxy_dir, "J03*.npz"))):
        m = os.path.basename(f)[:-4]; z = np.load(f)
        lab, tru, L = z["labels_" + args.name].astype(np.int8), z["truth"].astype(np.int8), list(z["lengths"])
        _, periods = P.load_match(m)
        seen = np.concatenate([np.isfinite(P.degrade(p, args.sigma, args.tau, 1000 * i + zlib.crc32(m.encode()) % 997)[0]).all(1)
                               for i, p in enumerate(periods)])
        e = np.cumsum([0] + L); runs = list(zip(e[:-1], e[1:]))
        f1 = lambda x: spell_scores(tru, x, runs)["mean_f1"]
        live_t, live_p = tru < 2, lab < 2
        v = {"current": f1(lab)}
        x = lab.copy(); x[~live_t] = 2; fix = live_t & ~live_p; x[fix] = tru[fix]; v["oracle dead/live"] = f1(x)
        x = lab.copy(); m_ = live_t & live_p & ~seen; x[m_] = tru[m_]; v["oracle team, ball unseen"] = f1(x)
        x = lab.copy(); m_ = live_t & live_p & seen; x[m_] = tru[m_]; v["oracle team, ball seen"] = f1(x)
        x = lab.copy(); m_ = live_t & live_p; x[m_] = tru[m_]; v["oracle team (all live)"] = f1(x)
        fr = {"live team wrong, ball seen": float(np.mean((lab != tru)[live_t & live_p & seen])),
              "live team wrong, ball unseen": float(np.mean((lab != tru)[live_t & live_p & ~seen])),
              "false D (true live)": float(np.mean(~live_p[live_t])), "missed D (true dead)": float(np.mean(live_p[~live_t])),
              "ball unseen share": float(np.mean(~seen))}
        rows.append((m, v, fr))
    if not rows:
        raise SystemExit("no J03*.npz in " + args.proxy_dir)
    keys = list(rows[0][1]); fk = list(rows[0][2])
    print("%-26s %s   mean   gain" % ("", " ".join("%7s" % r[0][-3:] for r in rows)))
    for k in keys:
        vals = [r[1][k] for r in rows]; base = np.mean([r[1]["current"] for r in rows])
        print("%-26s %s  %.4f  %+.4f" % (k, " ".join("%7.4f" % x for x in vals), np.mean(vals), np.mean(vals) - base))
    for k in fk:
        print("%-26s %s  %.3f" % (k, " ".join("%7.3f" % r[2][k] for r in rows), np.mean([r[2][k] for r in rows])))


if __name__ == "__main__":
    main()
