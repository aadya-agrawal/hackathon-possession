"""Render a short clip of the tactical video with the predicted possession drawn on top.

Visual inspection aid only: it reads a submission CSV (and optionally a ground-truth CSV for a
side-by-side check of alignment) and the tactical video. It never feeds anything back into the
pipeline, and the licensed video is not redistributed -- keep the output clip out of the repository.

    python tools/render_possession_video.py \
        --csv Westside.csv --start 11185 --seconds 30 --out clip.mp4 \
        --gt evaluation/gt/BAR-ATH_possession_gt_public.csv

Needs ffmpeg (with drawtext/drawbox) on PATH.
"""

import argparse
import csv
import os
import subprocess
import sys
from itertools import groupby
from typing import Dict, List, NamedTuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games import GAMES, GAME, asset_paths  # noqa: E402

FONT = "/System/Library/Fonts/Helvetica.ttc"
COLOURS = {"H": "0x2563eb", "A": "0xdc2626", "D": "0x6b7280", "X": "0x374151"}
WIDTH, HEIGHT = 1280, 720
BANNER_W, BANNER_H, BANNER_Y = 640, 48, 12  # centred pill; leaves the broadcast clock visible
STRIP_H = 14
STRIP_GAP = 6
MARGIN = 24


class Segment(NamedTuple):
    label: str
    first: int  # clip-relative frame, inclusive
    last: int  # clip-relative frame, inclusive


def read_labels(path: str) -> List[str]:
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return [row[1].strip().upper() for row in reader]


def segments_in_window(labels: List[str], start: int, count: int) -> List[Segment]:
    window = labels[start:start + count]
    out, cursor = [], 0
    for label, run in groupby(window):
        length = len(list(run))
        out.append(Segment(label, cursor, cursor + length - 1))
        cursor += length
    return out


def banner_text(label: str, names: Dict[str, str]) -> str:
    return {
        "H": "{} IN POSSESSION".format(names["H"].upper()),
        "A": "{} IN POSSESSION".format(names["A"].upper()),
        "D": "BALL DEAD",
        "X": "NO ANSWER",
    }[label]


def banner_filters(segments: List[Segment], names: Dict[str, str]) -> List[str]:
    filters = []
    for seg in segments:
        gate = "enable='between(n,{},{})'".format(seg.first, seg.last)
        filters.append("drawbox=x='(iw-{})/2':y={}:w={}:h={}:color={}@0.88:t=fill:{}".format(
            BANNER_W, BANNER_Y, BANNER_W, BANNER_H, COLOURS[seg.label], gate))
        filters.append(
            "drawtext=fontfile={}:text='{}':fontsize=28:fontcolor=white:x=(w-text_w)/2:"
            "y={}+({}-text_h)/2:{}".format(
                FONT, banner_text(seg.label, names), BANNER_Y, BANNER_H, gate))
    return filters


def strip_filters(segments: List[Segment], count: int, y: int, caption: str) -> List[str]:
    usable = WIDTH - 2 * MARGIN
    filters = [
        "drawbox=x={}:y={}:w={}:h={}:color=black@0.55:t=fill".format(
            MARGIN - 6, y - 4, usable + 12, STRIP_H + 8),
        "drawtext=fontfile={}:text='{}':fontsize=14:fontcolor=white:x={}:y={}".format(
            FONT, caption, MARGIN, y - 22),
    ]
    for seg in segments:
        x = MARGIN + int(usable * seg.first / count)
        w = max(1, int(usable * (seg.last - seg.first + 1) / count))
        filters.append("drawbox=x={}:y={}:w={}:h={}:color={}:t=fill".format(
            x, y, w, STRIP_H, COLOURS[seg.label]))
    return filters


def cursor_graph(background: str, duration: float, fps: int, top: int, bottom: int) -> str:
    # drawbox evaluates its position once, so the moving cursor is a 3 px white source that is
    # overlaid with a per-frame x expression instead.
    usable = WIDTH - 2 * MARGIN
    return (
        "[0:v]{bg}[bg];color=c=white:s=3x{h}:r={fps}:d={dur:.3f}[cur];"
        "[bg][cur]overlay=x='{m}+{u}*t/{dur:.3f}':y={y}:eval=frame:shortest=1[out]"
    ).format(bg=background, h=bottom - top, fps=fps, dur=duration, m=MARGIN, u=usable, y=top)


def build_filter(pred: List[Segment], truth: Optional[List[Segment]], count: int,
                 fps: int, names: Dict[str, str]) -> str:
    base = HEIGHT - MARGIN - STRIP_H
    filters = ["scale={}:{}".format(WIDTH, HEIGHT)]
    filters += banner_filters(pred, names)
    filters += strip_filters(pred, count, base, "prediction")
    top = base - 6
    if truth is not None:
        gt_y = base - STRIP_H - STRIP_GAP - 20
        filters += strip_filters(truth, count, gt_y, "public ground truth")
        top = gt_y - 6
    return cursor_graph(",".join(filters), count / fps, fps, top, base + STRIP_H + 6)


def render(args: argparse.Namespace) -> None:
    fps = GAMES[GAME]["fps"]
    count = int(args.seconds * fps)
    pred = segments_in_window(read_labels(args.csv), args.start, count)
    truth = None
    if args.gt:
        truth = segments_in_window(read_labels(args.gt), args.start, count)
        if all(seg.label == "X" for seg in truth):
            truth = None
    video = args.video or asset_paths()["video"]
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", "{:.3f}".format(args.start / fps), "-i", video,
        "-frames:v", str(count),
        "-filter_complex",
        build_filter(pred, truth, count, fps, {"H": args.home, "A": args.away}),
        "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        args.out,
    ]
    subprocess.run(command, check=True)
    print("wrote {} ({} frames, {} predicted segments)".format(args.out, count, len(pred)))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", required=True, help="submission CSV (frame,label)")
    parser.add_argument("--gt", help="optional ground-truth CSV drawn as a second strip")
    parser.add_argument("--start", type=int, required=True, help="first video frame")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--video", help="tactical video; defaults to the data bundle")
    parser.add_argument("--home", default="Barcelona")
    parser.add_argument("--away", default="Athletic Club")
    render(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
