"""Masked-window swap test runner: run the full challenge pipeline with tracking-observed ball
frames hidden inside the swap windows of one split (swap_windows.py), or unmasked (the teacher).

The masked run is scored against the SAME config's unmasked run (score_swap.py): how well the
pipeline reconstructs its own output when the ball is hidden in live play where it was actually
observed. Label-free: only our own outputs are compared. The patched loader blanks the raw tactical
ball inside the windows; everything else (players, tags, broadcast, video evidence) is unchanged.
The config's working_submission_selection / compliance_provenance blocks are dropped (diagnostic
only; the config file itself is not modified).

Usage:
  python validation/swap_run.py OUT_DIR CONFIG NAME [SPLIT] --windows WORK/windows.json \
      [--data-dir BUNDLE] [--repo WORKTREE] [--video-ball-observations FILLS.json]
      [--contact-include-video-ball] [--team-evidence TEAM.npz --team-mode MODE]
  -> OUT_DIR/NAME_<SPLIT>.csv (SPLIT development|test) or OUT_DIR/NAME_teacher.csv (no SPLIT)
Round 11 (current default): for S in development test: swap_run.py OUT configs/submission_ball_aware.json base S;
plus the unmasked teacher (no SPLIT); then score_swap.py.
Earlier variants (session gapfill_current.py / gapfill_teampass.py) are options of this runner:
  nofill = SPLIT only; fill = SPLIT + --video-ball-observations fills_<SPLIT>.json;
  filltouch = fill + --contact-include-video-ball; real/realtouch = no SPLIT + fills_real.json;
  team-pass = configs/submission_touch_radius.json + --team-evidence team_<SPLIT>.npz --team-mode MODE
  (merge_team_evidence.py combines tactical and broadcast team-at-ball evidence).
Env fallbacks: MTAH_REPO, MTAH_DATA_DIR (data bundle, read by games.py).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import common
from swap_windows import load_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("config")
    parser.add_argument("name")
    parser.add_argument("split", nargs="?", choices=("development", "test"))
    parser.add_argument("--windows", help="windows.json from swap_windows.py (required with SPLIT)")
    parser.add_argument("--data-dir", help="challenge data bundle (default MTAH_DATA_DIR or <repo>/data)")
    parser.add_argument("--repo", help="repository/worktree whose pipeline is run (env MTAH_REPO)")
    parser.add_argument("--video-ball-observations", help="override video_ball_observations_path (fills json)")
    parser.add_argument("--contact-include-video-ball", action="store_true")
    parser.add_argument("--team-evidence", help="video_team_evidence_path (.npz frame/team)")
    parser.add_argument("--team-mode", help="video_team_mode")
    args = parser.parse_args()
    if args.split and not args.windows:
        parser.error("--windows is required with SPLIT")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve() if Path(args.config).exists() else None
    windows = load_windows(Path(args.windows).resolve(), args.split) if args.split else None
    team_evidence = str(Path(args.team_evidence).resolve()) if args.team_evidence else None
    fills = str(Path(args.video_ball_observations).resolve()) if args.video_ball_observations else None
    common.use_repo(common.repo_root(args.repo))
    import possession.cli as cli
    from games import GAME, GAMES
    config = json.loads((config_path or Path(args.config)).read_text())
    config.pop("working_submission_selection", None)
    config.pop("compliance_provenance", None)
    if fills:
        config["video_ball_observations_path"] = fills
    if args.contact_include_video_ball:
        config["contact_carry_include_video_ball"] = True
    if team_evidence or args.team_mode:
        config.update(video_team_evidence_path=team_evidence, video_team_mode=args.team_mode)
    if windows is not None:
        n_frames = GAMES[GAME]["n_video_frames"]
        original = cli.load_raw_ball

        def masked(path, frames, pitch):
            xy, extra = original(path, frames, pitch)
            if frames == n_frames:
                xy = xy.copy()
                for window in windows:
                    xy[window["start"]:window["stop"]] = np.nan
            return xy, extra
        cli.load_raw_ball = masked
    name = "{}_{}".format(args.name, args.split or "teacher")
    tmp = out_dir / ".{}.config.{}.json".format(name, os.getpid())
    tmp.write_text(json.dumps(config))
    try:
        cli.run_pipeline(args.data_dir, str(out_dir / (name + ".csv")), str(tmp))
    finally:
        tmp.unlink()


if __name__ == "__main__":
    sys.exit(main())
