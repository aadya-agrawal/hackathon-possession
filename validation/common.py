"""Shared helpers for the validation tools: repository location, external-data roots and a small
RAM-friendly pickle cache. Not a CLI.

Environment variables (all optional when the matching CLI argument is given):
  MTAH_REPO          repository (or worktree) whose possession/ code and configs are evaluated;
                     default: the repository containing this file
  METRICA_ROOT       Metrica Sports sample-data checkout (Sample_Game_1..3)
  DFL_ROOT           folder with the DFL open-data XMLs (DFL_02_01_matchinformation_*, DFL_04_03_positions_*)
  VALIDATION_CACHE   cache directory for parsed external data and baseline results (required by
                     fast_eval / proxy tools, default WORK/cache in train_folds; prefer a local RAM disk)
  FOLD_DIR           out-of-fold ball-free model weights (<kind>_ex_<match>.npz, train_folds.py)
  MTAH_DATA_DIR      challenge data bundle (read by games.py)
"""
import hashlib
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parent
KEEP_PER_NAME = 2
# DFL open-data (CC-BY) matches: four development matches and three held-out matches.
DFL_COMPETITION = {"J03WMX": "000001", "J03WN1": "000001", "J03WOH": "000002", "J03WOY": "000002",
                   "J03WPY": "000002", "J03WQQ": "000002", "J03WR9": "000002"}
DFL_DEV = ("J03WMX", "J03WN1", "J03WOH", "J03WOY")
DFL_HELD_OUT = ("J03WPY", "J03WQQ", "J03WR9")
METRICA_GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")


def env_path(name, value=None, required=True):
    """CLI value, else environment variable `name`; error with the variable name when required."""
    value = value or os.environ.get(name)
    if value is None:
        if required:
            raise SystemExit("set {} (environment) or pass the matching CLI option".format(name))
        return None
    return Path(value).expanduser().resolve()


def repo_root(value=None):
    return env_path("MTAH_REPO", value, required=False) or DEFAULT_REPO


def use_repo(repo):
    """Import possession/ and evaluation/ from `repo` and run relative config paths from there."""
    repo = str(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def cached(cache_dir, name, key, build):
    """Pickle cache keyed by (name, key); keeps only the newest KEEP_PER_NAME entries per name."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "{}-{}.pkl".format(name, key)
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    value = build()
    tmp = path.with_suffix(".tmp{}".format(os.getpid()))
    with tmp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    stale = sorted(cache_dir.glob("{}-*.pkl".format(name)), key=lambda f: f.stat().st_mtime, reverse=True)
    for old in stale[KEEP_PER_NAME:]:
        old.unlink(missing_ok=True)
    return value


def dfl_paths(dfl_root, match, competition=None):
    competition = competition or DFL_COMPETITION[match]
    root = Path(dfl_root)
    return (root / "DFL_02_01_matchinformation_DFL-COM-{}_DFL-MAT-{}.xml".format(competition, match),
            root / "DFL_04_03_positions_raw_observed_DFL-COM-{}_DFL-MAT-{}.xml".format(competition, match))


def load_dfl_cached(dfl_root, cache_dir, match, competition=None):
    """(pitch, periods) from possession.external_dfl_validation.load_dfl_match, cached on the XML stat."""
    from possession.external_dfl_validation import load_dfl_match
    metadata, positions = dfl_paths(dfl_root, match, competition)
    stamp = "{}:{}:{}".format(positions, positions.stat().st_size, positions.stat().st_mtime_ns)
    return cached(cache_dir, "dfl-parsed-" + match, sha256(stamp.encode())[:24],
                  lambda: load_dfl_match(metadata, positions))
