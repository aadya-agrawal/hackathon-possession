"""Build the label-free challenge input cache (challenge_inputs.npz) used by the validation tools.

The proxy evaluators replay the challenge's per-frame missing-player counts and raw ball-missing
mask on external DFL matches, and the swap-test window builder needs the raw tactical ball and
player positions. All of them read one cached array file built here from the gated data bundle
(tactical ATD tracking + metadata only; no labels, no video).

Arrays (tactical video frame index, 199,319 frames, NaN = not tracked):
  pos  (frames, slots, 2) float32  player positions in metres (repo pitch size)
  spd  (frames, slots)    float32  provider player speed where the player is tracked
  ball (frames, 2)        float32  raw tactical ball position in metres
  team (slots,)           int8     0 home / 1 away per track slot (metadata team order)

Usage (repository root or anywhere; data bundle located via MTAH_DATA_DIR or ./data):
  python validation/build_challenge_inputs.py --out WORK/challenge_inputs.npz [--check OLD.npz]

--check compares against an existing file and prints only aggregate agreement (no rows).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def build(data_dir=None):
    from games import GAME, GAMES, require
    from possession.broadcast import load_player_array
    from possession.tracking import load_atd_arrays, load_raw_ball, load_tracking_metadata
    game = GAMES[GAME]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], data_dir=data_dir)
    metadata = load_tracking_metadata(meta_path)
    pitch = game["pitch"]
    pos = load_player_array(raw_path, metadata.team_of_track, pitch).astype(np.float32)
    atd = load_atd_arrays(raw_path)
    if atd is None or atd.players.shape[:2] != pos.shape[:2]:
        raise ValueError("ATD file is not strictly well-formed; cannot read player speeds")
    spd = np.where(np.isfinite(pos[..., 0]), atd.players[..., 2], np.nan).astype(np.float32)
    ball, _ = load_raw_ball(raw_path, game["n_video_frames"], pitch)
    return {"pos": pos, "spd": spd, "ball": ball.astype(np.float32),
            "team": np.asarray(metadata.team_of_track, dtype=np.int8)}


def compare(new, old_path):
    old = np.load(old_path)
    for key, value in new.items():
        ref = old[key]
        if ref.shape != value.shape:
            print(key, "shape differs", ref.shape, value.shape)
            continue
        same = (ref == value) | (np.isnan(ref) & np.isnan(value)) if value.dtype.kind == "f" else ref == value
        print(key, "identical" if same.all() else "differs on {:.6f} of entries".format(1 - same.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--data-dir", help="data bundle root (default MTAH_DATA_DIR or <repo>/data)")
    parser.add_argument("--check", help="existing challenge_inputs.npz to compare against (aggregate report only)")
    args = parser.parse_args()
    arrays = build(args.data_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print("wrote", args.out, {k: v.shape for k, v in arrays.items()})
    if args.check:
        compare(arrays, args.check)


if __name__ == "__main__":
    main()
