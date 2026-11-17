"""Download the video detector weights used by tools/extract_team_at_ball.py, tools/audit_track_teams.py and
tools/derive_match_constants.py.

  python tools/fetch_video_weights.py [--cache cache] [--force]

Files (saved under --cache, default <repo>/cache/, sha256 verified; an existing file with the right hash is kept):
  yolo-sn-ball-opt.pt  ball detector from SoccerNet-v3D (https://github.com/mguti97/SoccerNet-v3D, release v1.0.0).
                       Licence: GPL-2.0 (SoccerNet-v3D repository).
  yolo26l.pt           Ultralytics YOLO26-L COCO detector, person class 0 (https://github.com/ultralytics/assets,
                       release v8.4.0, the asset tag ultralytics 8.4.x downloads from). Licence: AGPL-3.0 (Ultralytics).
  yolo26x.pt           Ultralytics YOLO26-X COCO detector (same release); only one of the five pilot detectors that
                       derive the static detector-clutter cells (tools/derive_match_constants.py). Licence: AGPL-3.0.
Only the standard library is needed. Exits non-zero if a download fails or a hash does not match (the partial
download is removed, an existing mismatching file is left untouched unless --force).
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = {
    "yolo-sn-ball-opt.pt": ("https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/yolo-sn-ball-opt.pt",
                            "a3082fb435a8501ae17cfe4ac78e66ca7041205e115feda344d34d1693064f36"),
    "yolo26l.pt": ("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
                   "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26"),
    "yolo26x.pt": ("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
                   "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, url, expected, cache, force):
    target = cache / name
    if target.exists():
        digest = sha256(target)
        if digest == expected:
            print("ok (cached)", target, digest)
            return True
        if not force:
            print("MISMATCH", target, digest, "expected", expected, "(use --force to replace)")
            return False
    part = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(part, "wb") as out:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                out.write(chunk)
        digest = sha256(part)
        if digest != expected:
            print("MISMATCH", url, digest, "expected", expected)
            return False
        os.replace(part, target)
        print("ok (downloaded)", target, digest)
        return True
    finally:
        if part.exists():
            part.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default=str(REPO / "cache"))
    parser.add_argument("--force", action="store_true", help="replace existing files whose hash does not match")
    args = parser.parse_args(argv)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    ok = [fetch(name, url, expected, cache, args.force) for name, (url, expected) in WEIGHTS.items()]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
