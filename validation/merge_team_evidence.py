"""Merge tactical and broadcast team-at-ball evidence for the swap-test team-pass variant.

Tactical frames first; broadcast adds new frames; frames where both decide and disagree are
dropped. Inputs/outputs are .npz files with int64 "frame" and int8 "team" arrays (the format of
video_team_evidence_path). Prints counts only.

Usage: python validation/merge_team_evidence.py TACTICAL.npz BROADCAST.npz OUT.npz
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tactical")
    parser.add_argument("broadcast")
    parser.add_argument("out")
    args = parser.parse_args()
    tac, bc = np.load(args.tactical), np.load(args.broadcast)
    tf, tt = tac["frame"].astype(np.int64), tac["team"].astype(np.int8)
    bf, bt = bc["frame"].astype(np.int64), bc["team"].astype(np.int8)
    both = np.intersect1d(tf, bf)
    tmap = dict(zip(tf.tolist(), tt.tolist())); bmap = dict(zip(bf.tolist(), bt.tolist()))
    conflict = {f for f in both.tolist() if tmap[f] != bmap[f]}
    frames = sorted((set(tmap) | set(bmap)) - conflict)
    teams = [tmap.get(f, bmap.get(f)) for f in frames]
    np.savez_compressed(args.out, frame=np.array(frames, np.int64), team=np.array(teams, np.int8))
    print("tactical %d broadcast %d both %d conflicts %d merged %d" % (len(tf), len(bf), len(both), len(conflict), len(frames)))


if __name__ == "__main__":
    main()
