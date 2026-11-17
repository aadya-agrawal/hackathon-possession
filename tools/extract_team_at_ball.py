"""Image-space TEAM AT THE BALL from the tactical video - deterministic extractor for evidence/team_at_ball_gaps.npz.

Default frame set 'real' = graded frames (both halves, games.half_video_blocks) where the raw tactical ball in
data/tactical/BAR-ATH_tactical_atd.txt is missing; the committed evidence/team_at_ball_gaps.npz is the output of
this script on that set. 'development' / 'test' reproduce the swap-test evidence on the masked windows.
Inputs (all read from the provided data bundle, games.asset_paths): tactical video, ATD metadata + raw tracking
(possession.tracking / possession.broadcast loaders, pitch from games.GAMES), homography CSV; detector weights
from cache/ (tools/fetch_video_weights.py downloads them; sha256 checked before any detection).

Pipeline (all thresholds frozen; the tracked BALL only selects the target frames, decisions use the tracked
PLAYERS + metadata team array). The match-specific inputs - swap-test masked windows, static detector-clutter cells
and the per-half playing direction - are derived from the data bundle by tools/derive_match_constants.py (see its
docstring; teacher CSV of configs/submission_contact_carry.json, a seeded pilot sample with five tiled ball
detectors, possession.ballfree.attack_directions) and cached in <work>/match_constants.json; nothing is hand-set.
  0. derive  : <work>/match_constants.json (masked_windows, static_cells), unless already present.
  1. balls   : S3 ball detector = <ball weights> (class 0), 3x3 tiles (10% overlap) of the 1920x1080 frame, imgsz 1280,
               conf 0.01, fp32, one predict call per frame (batch = 9 tiles); candidates merged at 10 px (keep higher
               score), top 150; each candidate [score(4dp), x(1dp), y(1dp), w, h, pitch xy (3dp) via inverse homography].
               Run on the target frames grown by +-4 (temporal persistence needs neighbours).
  2. persons : COCO person detector = <person weights> (class 0), full frame, imgsz 1920, conf 0.2, fp32, one frame per
               predict call; per box a 16x8 torso crop (rows 12-55% of box height, cols 20-80% of width, INTER_AREA).
               Run on (a) the 3000 colour-training frames, (b) target frames with a 'dense' ball (see 4).
  3. colour  : kit model. Training frames = sorted(np.random.default_rng(11).choice(P, 3000, replace=False)) where
               P = graded frames (11185-90651, 110808-190197) outside the 116 swap-test masked windows (derived
               masked_windows, each padded by 50 frames). Labels from person feet matched (greedy, <= 0.5 box heights)
               to tracked players projected with the frame homography: isolated matches (<= 0.35 h, no rival within
               0.9 h) -> home / away / home GK (slot 23) / away GK (slot 45); unmatched > 1.5 h from every projection
               -> other. Features: grass-masked
               HSV 9x3x3 histogram, Lab mean/std, grass share, unmasked 18-hue x 2 sat x 2 val histogram, Lab mean.
               HistGradientBoostingClassifier(max_iter=300, lr=0.08, max_leaf_nodes=31, l2=1.0, random_state=0) on all
               labelled persons in (frame, x1) order. Prediction: GK classes merged; GK team from pitch side of the feet
               (home GK defends x < 52.5 in the halves where attack_directions finds home attacking +x).
  4. ball rule per target frame t:
               'fill'  : conservative rule - best S3 candidate with score >= 0.3 outside the derived static-clutter
                         cells (+-1.5 m), unique (no other non-static >= 0.3 candidate > 10 px away), pitch xy
                         within 1.5 m of a tracked player, and 4/4 persistence (t-4, t-2, t+2, t+4 each have a >= 0.1 candidate within 8 + 12|o| px);
                         then a 40 m/s speed filter over the sorted fill observations of the whole target set
                         (iteratively drop the observation in most violations, ties -> lower score).
               'dense' : otherwise the best non-static >= 0.3 candidate if unique (same uniqueness test).
  5. team    : persons whose feet region [x1 - 0.2h, x2 + 0.2h] x [y2 - 0.15h, y2 + 0.05h] is within K = 0.25 h of the ball
               centre. Abstain if none, if the ball is above y2 - 0.35 h for all of them (airborne), if all are 'other'
               (p >= 0.9), if confident (p >= TAU = 0.9) candidates of both teams (duel), if a candidate is uncertain,
               or if the colour team conflicts with the team of the tracked players matched to the candidates.
               confidence = min colour probability of the deciding candidates.

Usage (run from the repository root; export TMPDIR=/dev/shm; stages can be sharded over processes/GPUs, results are
per frame and merged in sorted frame order, so shard counts / GPUs / concurrency do not change the output):
  # one command: shards balls then persons over 3 GPUs x 5 processes, then assembles on CPU (~66 min wall on 3 L40S, shared host)
  python tools/extract_team_at_ball.py run --devices 5,6,7 --procs 5 --work /dev/shm/tab_work --out team_at_ball_gaps.npz
  # or stage by stage (persons only after all ball shards finished)
  python tools/extract_team_at_ball.py balls    --device 5 [--shard i --nshards n] --work W
  python tools/extract_team_at_ball.py persons  --device 5 [--shard i --nshards n] --work W
  python tools/extract_team_at_ball.py assemble --work W --out team_at_ball_gaps.npz
  python tools/extract_team_at_ball.py all      --device 5 --work W --out team_at_ball_gaps.npz   (single process)
The work dir caches the derived match constants (match_constants.json, match_constants/: teacher CSV, pilot frames
and pilot detections; derived with the --device / --devices GPUs and the weights next to --ball-weights, incl.
yolo26x.pt), per-frame detections (balls_*.json, persons_*.npz) and the colour model (colour_model.pkl, a
(classifier, info) pickle that tools/audit_track_teams.py reuses); frames already cached are not re-detected.
Determinism: verified with ultralytics 8.4.40, torch 2.4.1+cu121 (cuDNN 9.1), opencv 4.11.0, numpy 1.26.4, sklearn
1.6.1 on NVIDIA L40S (driver 550.90.12) - outputs are bit-identical across GPUs, shard counts and process
concurrency. CPU inference is NOT bit-identical (S3 scores differ by <= 6e-4, some person boxes differ): on 300 real
frames CPU decided 47 vs 52 frames on GPU with 0 team disagreements on the common frames.
--frames SPEC: real (default) | development | test | windows (both) | path to .npy/.txt frame list.
Other options: --data-dir (provided data bundle, default games.DATA_DIR), --ball-weights / --person-weights
(default cache/yolo-sn-ball-opt.pt, cache/yolo26l.pt), --work (cache dir), --device: GPU index (physical, PCI bus
order) or 'cpu'.
Licences of the weights: yolo-sn-ball-opt.pt from SoccerNet-v3D (GPL-2.0), yolo26l.pt from Ultralytics (AGPL-3.0).
"""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import games  # noqa: E402
import derive_match_constants as C  # noqa: E402

GAME = games.GAMES[games.GAME]
N_FRAMES = GAME["n_video_frames"]
PITCH = np.array([105.0, 68.0])  # homography maps pitch / (105, 68) to pixels
# graded video frames [start, stop): 11185-90651 and 110808-190197
GRADED = tuple((first, last + 1) for _, first, last in games.half_video_blocks(games.GAME))
K, TAU, OTHER_TAU, AIR_FRAC = 0.25, 0.9, 0.9, 0.35
EXPECTED_SHA256 = {"ball": "a3082fb435a8501ae17cfe4ac78e66ca7041205e115feda344d34d1693064f36",
                   "person": "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26"}


# ---------------------------------------------------------------- inputs
def graded_mask():
    g = np.zeros(N_FRAMES, dtype=bool)
    for a, e in GRADED:
        g[a:e] = True
    return g


def match_constants(work, devices=("0",), data_dir=None, cache=None):
    """Swap-test masked windows and static clutter cells, derived from the data bundle (tools/derive_match_constants.py)."""
    return C.derive(work, devices, data_dir, cache)


def window_mask(windows, splits=("d", "t"), pad=0):
    m = np.zeros(N_FRAMES, dtype=bool)
    for a, e, s in windows:
        if s in splits:
            m[max(0, a - pad):min(N_FRAMES, e + pad)] = True
    return m


def data_paths(data_dir=None):
    return games.asset_paths(games.GAME, data_dir)


def load_ball(data_dir=None):
    """Raw tactical ball (N, 2) metres, NaN where the tracker has no ball."""
    from possession.tracking import load_raw_ball
    ball, _ = load_raw_ball(data_paths(data_dir)["atd_raw"], N_FRAMES, GAME["pitch"])
    return ball.astype(np.float64)


def load_tracking(data_dir=None):
    """Players (N, 46, 2) metres (float32 precision), metadata team per slot, goalkeeper slot per team."""
    from possession.broadcast import load_player_array
    from possession.tracking import load_tracking_metadata
    paths = data_paths(data_dir)
    meta = load_tracking_metadata(paths["atd_meta"])
    pos = load_player_array(paths["atd_raw"], meta.team_of_track, GAME["pitch"]).astype(np.float32)
    team = meta.team_of_track.astype(int)
    gk_slots = {int(team[s]): int(s) for s in meta.goalkeeper_slots}
    if sorted(gk_slots) != [0, 1]:
        raise ValueError("expected one goalkeeper slot per team, got {}".format(meta.goalkeeper_slots))
    return pos.astype(np.float64), team, gk_slots


def load_homographies(data_dir=None):
    from possession.video import load_homography_matrices
    return load_homography_matrices(data_paths(data_dir)["homography"], N_FRAMES)


def target_frames(spec, ball, windows):
    if spec in ("development", "test", "windows"):
        return np.flatnonzero(window_mask(windows, {"development": ("d",), "test": ("t",), "windows": ("d", "t")}[spec]))
    if spec == "real":
        return np.flatnonzero(graded_mask() & ~np.isfinite(ball).all(1))
    p = Path(spec)
    f = np.load(p) if p.suffix == ".npy" else np.loadtxt(p, dtype=np.int64, ndmin=1)
    return np.unique(np.asarray(f, dtype=np.int64))


def grow(frames, margin=4):
    m = np.zeros(N_FRAMES, dtype=bool)
    m[frames] = True
    g = m.copy()
    for o in range(1, margin + 1):
        g[o:] |= m[:-o]
        g[:-o] |= m[o:]
    return np.flatnonzero(g)


def colour_training_frames(windows):
    pool = graded_mask() & ~window_mask(windows, pad=50)
    rng = np.random.default_rng(11)
    return np.array(sorted(rng.choice(np.flatnonzero(pool), size=3000, replace=False).tolist()), dtype=np.int64)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- geometry
def project(matrix, x, y):
    den = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return np.asarray([(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / den,
                       (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / den])


def to_pixels(matrix, xy):
    n = xy / PITCH
    h = np.c_[n, np.ones(len(n))] @ matrix.T
    return h[:, :2] / h[:, 2:3]


def to_pitch(matrix, px):
    inv = np.linalg.inv(matrix)
    h = np.c_[px, np.ones(len(px))] @ inv.T
    return h[:, :2] / h[:, 2:3] * PITCH


def match_persons(boxes, players_px, slots):
    n = len(boxes)
    slot = np.full(n, -1)
    ratio = np.full(n, np.inf)
    isolated = np.zeros(n, dtype=bool)
    if n == 0 or len(players_px) == 0:
        return slot, ratio, isolated
    feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    d = np.linalg.norm(feet[:, None] - players_px[None], axis=2) / h[:, None]
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    used_b, used_p = set(), set()
    for b, p in order:
        if d[b, p] > 0.5:
            break
        if b in used_b or p in used_p:
            continue
        used_b.add(b)
        used_p.add(p)
        slot[b] = slots[p]
        ratio[b] = d[b, p]
        second = np.sort(d[b])[1] if d.shape[1] > 1 else np.inf
        other_box = np.min(np.delete(d[:, p], b)) if n > 1 else np.inf
        isolated[b] = d[b, p] <= 0.35 and second > 0.9 and other_box > 0.9
    return slot, ratio, isolated


# ---------------------------------------------------------------- detection
class Video:
    """Sequential decoding with seeks only across gaps > 50 frames (decoded pixels are identical either way)."""

    def __init__(self, path):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        self.next = None

    def frames(self, wanted):
        for f in wanted:
            if self.next is None or f < self.next or f - self.next > 50:
                self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, int(f))
                self.next = int(f)
            while self.next < f:
                self.cap.grab()
                self.next += 1
            ok, image = self.cap.read()
            if not ok:
                raise ValueError("cannot decode frame {}".format(f))
            self.next = int(f) + 1
            yield int(f), image


def device_arg(device):
    return "cpu" if str(device) == "cpu" else int(device)


def load_yolo(path):
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    return YOLO(str(path))


def tiles(width, height, nx, ny, overlap=0.1):
    tw = int(np.ceil(width / (nx - (nx - 1) * overlap)))
    th = int(np.ceil(height / (ny - (ny - 1) * overlap)))
    xs = np.linspace(0, width - tw, nx).round().astype(int)
    ys = np.linspace(0, height - th, ny).round().astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def detect_balls(image, model, device, matrix, class_id=0, grid=(3, 3)):
    height, width = image.shape[:2]
    regions = tiles(width, height, *grid)
    results = model.predict([image[y0:y1, x0:x1] for x0, y0, x1, y1 in regions], classes=[class_id], conf=0.01,
                            imgsz=1280, device=device, batch=len(regions), verbose=False)
    raw = []
    for (x0, y0, _, _), result in zip(regions, results):
        if result.boxes is None:
            continue
        for (cx, cy, w, h), score in zip(result.boxes.xywh.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            raw.append((float(score), cx + x0, cy + y0, float(w), float(h)))
    raw.sort(reverse=True)
    merged = []
    for score, x, y, w, h in raw:
        if all(np.hypot(x - m[1], y - m[2]) > 10.0 for m in merged):
            merged.append((score, x, y, w, h))
    inv = np.linalg.inv(matrix) if np.isfinite(matrix).all() else None
    out = []
    for score, x, y, w, h in merged[:150]:
        xy = None if inv is None else (project(inv, x, y) * PITCH).round(3).tolist()
        out.append([round(score, 4), round(float(x), 1), round(float(y), 1), round(w, 1), round(h, 1), xy])
    return out


def torso_crop(image, box):
    import cv2
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    ya, yb = int(round(y1 + 0.12 * h)), int(round(y1 + 0.55 * h))
    xa, xb = int(round(x1 + 0.2 * w)), int(round(x2 - 0.2 * w))
    ya, xa = max(ya, 0), max(xa, 0)
    yb, xb = min(yb, image.shape[0]), min(xb, image.shape[1])
    if yb - ya < 2 or xb - xa < 2:
        return np.zeros((16, 8, 3), dtype=np.uint8)
    return cv2.resize(image[ya:yb, xa:xb], (8, 16), interpolation=cv2.INTER_AREA)


def detect_persons(image, model, device):
    r = model.predict(image, classes=[0], conf=0.2, imgsz=1920, device=device, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    confs = r.boxes.conf.cpu().numpy().astype(np.float32)
    order = np.argsort(boxes[:, 0], kind="stable")  # (frame, x1) order, ties keep detector order
    boxes, confs = boxes[order], confs[order]
    crops = np.array([torso_crop(image, b) for b in boxes], dtype=np.uint8).reshape(-1, 16, 8, 3)
    return boxes, confs, crops


def shard_of(frames, shard, nshards, chunk=500):
    frames = np.asarray(sorted(frames))
    chunks = [frames[i:i + chunk] for i in range(0, len(frames), chunk)]
    return np.concatenate(chunks[shard::nshards]) if chunks[shard::nshards] else np.array([], dtype=np.int64)


# ---------------------------------------------------------------- caches
def load_ball_cache(work):
    cands = {}
    for p in sorted(Path(work).glob("balls_*.json")):
        cands.update({int(f): v for f, v in json.loads(p.read_text()).items()})
    return cands


def load_person_cache(work):
    out = {}
    for p in sorted(Path(work).glob("persons_*.npz")):
        with np.load(p) as z:
            d = {k: z[k] for k in ("frame", "box", "conf", "crop")}
        bounds = np.flatnonzero(np.diff(np.r_[-1, d["frame"], -2]))
        for a, e in zip(bounds[:-1], bounds[1:]):
            out[int(d["frame"][a])] = (d["box"][a:e], d["conf"][a:e], d["crop"][a:e])
    return out


def save_person_cache(path, rows):
    frames = sorted(rows)
    cat = lambda i, shape, dt: (np.concatenate([rows[f][i] for f in frames]).astype(dt) if frames else np.zeros(shape, dt))
    np.savez(path, frame=np.concatenate([np.full(len(rows[f][0]), f, dtype=np.int32) for f in frames]) if frames
             else np.zeros(0, np.int32), box=cat(0, (0, 4), np.float32), conf=cat(1, (0,), np.float32),
             crop=cat(2, (0, 16, 8, 3), np.uint8))


# ---------------------------------------------------------------- ball rules
def is_static(xy, cells):
    return xy is not None and any(abs(xy[0] - x) <= 1.5 and abs(xy[1] - y) <= 1.5 for x, y, _ in cells)


def dense_best(cs, cells):
    if not cs:
        return None
    flagged = [(c, is_static(c[5], cells)) for c in cs]
    pool = [c for c, st in flagged if c[0] >= 0.3 and not st]
    if not pool:
        return None
    best = max(pool, key=lambda c: c[0])
    if any(c[0] >= 0.3 and not st and np.hypot(c[1] - best[1], c[2] - best[2]) > 10 for c, st in flagged):
        return None
    return best


def fill_select(t, cands, players, cells):
    cs = cands.get(t)
    if not cs:
        return None
    flagged = [(c, is_static(c[5], cells)) for c in cs]
    pool = [c for c, st in flagged if c[0] >= 0.3 and not st]
    if not pool:
        return None
    best = max(pool, key=lambda c: c[0])
    bpx = np.array(best[1:3])
    if any(c[0] >= 0.3 and not st and np.linalg.norm(np.array(c[1:3]) - bpx) > 10 for c, st in flagged):
        return None
    if best[5] is None or not len(players):
        return None
    if np.min(np.linalg.norm(players - np.array(best[5]), axis=1)) > 1.5:
        return None
    for o in (-4, -2, 2, 4):
        nb = cands.get(t + o)
        if nb is None or not any(c[0] >= 0.1 and np.linalg.norm(np.array(c[1:3]) - bpx) <= 8 + 12 * abs(o) for c in nb):
            return None
    return best


def speed_filter(obs, fps=25.0, max_speed=40.0):
    obs = list(obs)
    while True:
        bad = np.zeros(len(obs), dtype=int)
        for i in range(len(obs) - 1):
            a, b = obs[i], obs[i + 1]
            dt = (b["frame"] - a["frame"]) / fps
            if np.hypot(b["pitch_x"] - a["pitch_x"], b["pitch_y"] - a["pitch_y"]) / dt > max_speed:
                bad[i] += 1
                bad[i + 1] += 1
        if not bad.any():
            return obs
        worst = max(np.flatnonzero(bad == bad.max()), key=lambda i: -obs[i]["confidence"])
        obs.pop(int(worst))


# ---------------------------------------------------------------- colour model
def colour_features(crops):
    import cv2
    n = len(crops)
    flat = crops.reshape(-1, 1, 3)
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV).reshape(n, -1, 3).astype(np.float32)
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).reshape(n, -1, 3).astype(np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (hue >= 30) & (hue <= 90) & (sat >= 50) & (val >= 40)
    keep = ~grass
    w = keep / np.maximum(keep.sum(1, keepdims=True), 1)
    hb = np.minimum((hue / 180 * 9).astype(int), 8)
    sb = np.minimum((sat / 256 * 3).astype(int), 2)
    vb = np.minimum((val / 256 * 3).astype(int), 2)
    idx = hb * 9 + sb * 3 + vb
    hist = np.zeros((n, 81), dtype=np.float32)
    for k in range(81):
        hist[:, k] = (w * (idx == k)).sum(1)
    mean = (lab * w[..., None]).sum(1)
    std = np.sqrt(np.maximum((((lab - mean[:, None]) ** 2) * w[..., None]).sum(1), 0))
    idx2 = np.minimum((hue / 180 * 18).astype(int), 17) * 4 + (sat >= 110) * 2 + (val >= 150)
    hist2 = np.zeros((n, 72), dtype=np.float32)
    for k in range(72):
        hist2[:, k] = (idx2 == k).mean(1)
    return np.c_[hist, mean / 255.0, std / 255.0, grass.mean(1, keepdims=True), hist2,
                 lab.mean(1) / 255.0].astype(np.float32)


def stack_persons(person_cache, frames):
    frames = [f for f in frames if f in person_cache]
    fr = np.concatenate([np.full(len(person_cache[f][0]), f, dtype=np.int64) for f in frames])
    box = np.concatenate([person_cache[f][0] for f in frames])
    crop = np.concatenate([person_cache[f][2] for f in frames])
    return fr, box, crop


def feet_pitch_x(frames, boxes, mats):
    out = np.full(len(frames), np.nan)
    bounds = np.flatnonzero(np.diff(np.r_[-1, frames, -2]))
    for a, e in zip(bounds[:-1], bounds[1:]):
        m = mats[frames[a]]
        if np.isfinite(m).all():
            b = boxes[a:e]
            out[a:e] = to_pitch(m, np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]])[:, 0]
    return out


def train_colour(person_cache, pos, team, mats, gk_slots, windows):
    from sklearn.ensemble import HistGradientBoostingClassifier
    frames, boxes, crops = stack_persons(person_cache, colour_training_frames(windows).tolist())
    n = len(frames)
    slot = np.full(n, -1)
    iso = np.zeros(n, dtype=bool)
    nearest = np.full(n, np.inf)
    bounds = np.flatnonzero(np.diff(np.r_[-1, frames, -2]))
    for a, e in zip(bounds[:-1], bounds[1:]):
        f = frames[a]
        m = mats[f]
        ok = np.isfinite(pos[f]).all(1)
        if not np.isfinite(m).all() or not ok.any():
            continue
        px = to_pixels(m, pos[f][ok])
        b = boxes[a:e].astype(np.float32)
        s, _, i = match_persons(b, px, np.flatnonzero(ok))
        slot[a:e], iso[a:e] = s, i
        feet = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
        h = np.maximum(b[:, 3] - b[:, 1], 1)
        nearest[a:e] = (np.linalg.norm(feet[:, None] - px[None], axis=2) / h[:, None]).min(1)
    cls = np.full(n, -1)
    matched = slot >= 0
    t = np.where(matched, team[np.maximum(slot, 0)], -1)
    gk = matched & ((slot == gk_slots[0]) | (slot == gk_slots[1]))
    cls[matched & iso & ~gk] = t[matched & iso & ~gk]
    cls[matched & iso & gk] = 2 + t[matched & iso & gk]
    cls[~matched & (nearest > 1.5)] = 4
    x = colour_features(crops)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(x[cls >= 0], cls[cls >= 0])
    return clf, {"persons": int(n), "labelled": int((cls >= 0).sum()),
                 "class_counts": {int(k): int(v) for k, v in zip(*np.unique(cls[cls >= 0], return_counts=True))}}


def colour_predict(clf, x, home_left, feet_x):
    """Team per person; home_left (per person) = home defends x < 52.5 in that frame (goalkeeper class)."""
    p = clf.predict_proba(x)
    full = np.zeros((len(x), 5))
    full[:, clf.classes_] = p
    merged = np.c_[full[:, 0], full[:, 1], full[:, 2] + full[:, 3], full[:, 4]]
    k = merged.argmax(1)
    team = np.array([0, 1, -1, -1])[k]
    gk_team = np.where(np.isfinite(feet_x), np.where((feet_x < 52.5) == home_left, 0, 1), -1)
    team = np.where(k == 2, gk_team, team)
    return team, merged.max(1)


# ---------------------------------------------------------------- team at ball
def assess(ball, box, cteam, cconf, ident):
    out = {"team_colour": -1, "team_ident": -1, "team": -1, "confidence": 0.0, "colour_confidence": 0.0,
           "n_candidates": 0, "reason": ""}
    if box is None or not len(box):
        out["reason"] = "no_persons"
        return out
    b = box.astype(np.float64)
    h = np.maximum(b[:, 3] - b[:, 1], 1.0)
    bx, by = ball[1], ball[2]
    dx = np.maximum(0, np.maximum((b[:, 0] - 0.2 * h) - bx, bx - (b[:, 2] + 0.2 * h)))
    dy = np.maximum(0, np.maximum((b[:, 3] - 0.15 * h) - by, by - (b[:, 3] + 0.05 * h)))
    cand = np.flatnonzero(np.hypot(dx, dy) / h <= K)
    out["n_candidates"] = int(len(cand))
    if not len(cand):
        out["reason"] = "no_candidate"
        return out
    if np.all(by < b[cand, 3] - AIR_FRAC * h[cand]):
        out["reason"] = "airborne"
        return out
    ct, cc, ci = cteam[cand], cconf[cand], ident[cand]
    is_other = (ct < 0) & (cc >= OTHER_TAU)
    team_c = ~is_other & (ct >= 0) & (cc >= TAU)
    uncertain = ~is_other & ~team_c
    colour_teams = set(ct[team_c].tolist())
    if is_other.all():
        out["reason"] = "only_other"
        return out
    if len(colour_teams) == 1 and not uncertain.any():
        out["team_colour"] = colour_teams.pop()
        out["colour_confidence"] = float(cc[team_c].min())
    ident_teams = set(ci[~is_other].tolist())
    if len(ident_teams) == 1 and -1 not in ident_teams:
        out["team_ident"] = ident_teams.pop()
    if len(colour_teams) > 1:
        out["reason"] = "duel"
    elif uncertain.any():
        out["reason"] = "uncertain_colour"
    if out["team_colour"] >= 0:
        if out["team_ident"] >= 0 and out["team_ident"] != out["team_colour"]:
            out["reason"] = "conflict"
        else:
            out["team"] = out["team_colour"]
            out["confidence"] = float(cc[team_c].min())
    return out


# ---------------------------------------------------------------- stages
def stage_balls(args, frames, mats):
    need = shard_of(grow(frames), args.shard, args.nshards, args.chunk)
    have = load_ball_cache(args.work)
    need = [f for f in need.tolist() if f not in have]
    model = load_yolo(args.ball_weights)
    dev = device_arg(args.device)
    out, t0 = {}, time.time()
    for f, image in Video(data_paths(args.data_dir)["video"]).frames(need):
        out[str(f)] = detect_balls(image, model, dev, mats[f])
        if len(out) % 2000 == 0:
            print("balls", args.shard, len(out), len(need), "{:.0f}s".format(time.time() - t0), flush=True)
    if out:
        (Path(args.work) / "balls_{}_{}_{}.json".format(args.tag, args.shard, args.nshards)).write_text(json.dumps(out))
    print("balls done", args.shard, len(out), "{:.0f}s".format(time.time() - t0), flush=True)


def person_frames(frames, cands, constants):
    return sorted(set(colour_training_frames(constants["masked_windows"]).tolist())
                  | {f for f in frames.tolist() if dense_best(cands.get(f), constants["static_cells"]) is not None})


def stage_persons(args, frames, constants):
    cands = load_ball_cache(args.work)
    missing = [f for f in frames.tolist() if f not in cands]
    if missing:
        raise SystemExit("ball cache missing {} target frames (run balls first)".format(len(missing)))
    have = load_person_cache(args.work)
    need = [f for f in shard_of(person_frames(frames, cands, constants), args.shard, args.nshards, args.chunk).tolist()
            if f not in have]
    model = load_yolo(args.person_weights)
    dev = device_arg(args.device)
    rows, t0 = {}, time.time()
    for f, image in Video(data_paths(args.data_dir)["video"]).frames(need):
        rows[f] = detect_persons(image, model, dev)
        if len(rows) % 2000 == 0:
            print("persons", args.shard, len(rows), len(need), "{:.0f}s".format(time.time() - t0), flush=True)
    if rows:
        save_person_cache(Path(args.work) / "persons_{}_{}_{}.npz".format(args.tag, args.shard, args.nshards), rows)
    print("persons done", args.shard, len(rows), "{:.0f}s".format(time.time() - t0), flush=True)


def stage_assemble(args, frames, pos, team, gk_slots, mats, constants):
    t0 = time.time()
    cells = constants["static_cells"]
    home_left = C.home_left_frames(pos, team)
    cands = load_ball_cache(args.work)
    persons = load_person_cache(args.work)
    model_path = Path(args.work) / "colour_model.pkl"
    if model_path.exists():
        clf, cinfo = pickle.loads(model_path.read_bytes())
    else:
        clf, cinfo = train_colour(persons, pos, team, mats, gk_slots, constants["masked_windows"])
        model_path.write_bytes(pickle.dumps((clf, cinfo)))
    players = lambda t: pos[t][np.isfinite(pos[t]).all(1)]
    selected = {}
    for t in frames.tolist():
        b = fill_select(t, cands, players(t), cells)
        if b is not None:
            selected[t] = b
    obs = speed_filter([{"frame": int(t), "pitch_x": float(b[5][0]), "pitch_y": float(b[5][1]), "confidence": float(b[0])}
                        for t, b in sorted(selected.items())])
    fill_frames = {o["frame"] for o in obs}
    rows = []
    for t in frames.tolist():
        b, tier = None, None
        if t in fill_frames:
            b = fill_select(t, cands, players(t), cells)
            tier = "fill" if b is not None else None
        if b is None:
            b = dense_best(cands.get(t), cells)
            tier = "dense" if b is not None else None
        if b is None:
            continue
        box = None
        cteam = cconf = ident = None
        if t in persons and len(persons[t][0]):
            box = persons[t][0]
            fr = np.full(len(box), t)
            cteam, cconf = colour_predict(clf, colour_features(persons[t][2]), home_left[fr], feet_pitch_x(fr, box, mats))
            ident = np.full(len(box), -1)
            m = mats[t]
            ok = np.isfinite(pos[t]).all(1)
            if np.isfinite(m).all() and ok.any():
                slot, _, _ = match_persons(box, to_pixels(m, pos[t][ok]), np.flatnonzero(ok))
                ident = np.where(slot >= 0, team[np.maximum(slot, 0)], -1)
        r = assess(b, box, cteam, cconf, ident)
        r["frame"], r["tier"] = t, tier
        rows.append(r)
    dec = [r for r in rows if r["team"] >= 0]
    col = [r for r in rows if r["team_colour"] >= 0]
    np.savez(args.out,
             frame=np.array([r["frame"] for r in dec], dtype=np.int64),
             team=np.array([r["team"] for r in dec], dtype=np.int8),
             confidence=np.array([r["confidence"] for r in dec], dtype=np.float32),
             tier=np.array([r["tier"] for r in dec]),
             n_candidates=np.array([r["n_candidates"] for r in dec], dtype=np.int16),
             team_ident=np.array([r["team_ident"] for r in dec], dtype=np.int8),
             colour_frame=np.array([r["frame"] for r in col], dtype=np.int64),
             colour_team=np.array([r["team_colour"] for r in col], dtype=np.int8),
             colour_confidence=np.array([r["colour_confidence"] for r in col], dtype=np.float32))
    reasons = {}
    for r in rows:
        if r["team"] < 0:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    summary = {"frames": int(len(frames)), "fill_observations": len(obs), "frames_with_ball": len(rows),
               "decided": len(dec), "decided_by_tier": {k: sum(1 for r in dec if r["tier"] == k) for k in ("fill", "dense")},
               "abstain_reasons": reasons, "colour_model": cinfo, "seconds": round(time.time() - t0, 1)}
    Path(str(args.out) + ".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def run_sharded(args):
    """Launch the balls and then the persons stage as len(devices) x procs shard processes, then assemble."""
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    n = len(devices) * args.procs
    Path(args.work).mkdir(parents=True, exist_ok=True)
    common = [sys.executable, str(Path(__file__).resolve()), "--frames", args.frames, "--work", args.work,
              "--ball-weights", args.ball_weights, "--person-weights", args.person_weights, "--tag", args.tag,
              "--chunk", str(args.chunk)] + (["--data-dir", args.data_dir] if args.data_dir else [])
    for stage in ("balls", "persons"):
        t0, procs = time.time(), []
        for i in range(n):
            log = open(Path(args.work) / "log_{}_{}_{}.txt".format(stage, args.tag, i), "w")
            procs.append(subprocess.Popen([common[0], common[1], stage] + common[2:] + [
                "--device", devices[i % len(devices)], "--shard", str(i), "--nshards", str(n), "--skip-hash"],
                stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit("{} shards failed: {} (see {}/log_{}_*.txt)".format(stage, codes, args.work, stage))
        print(stage, "done in {:.0f}s".format(time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("balls", "persons", "assemble", "all", "run"))
    ap.add_argument("--frames", default="real")
    ap.add_argument("--data-dir", default=None, help="provided data bundle (default: games.DATA_DIR)")
    ap.add_argument("--ball-weights", default=str(REPO / "cache/yolo-sn-ball-opt.pt"))
    ap.add_argument("--person-weights", default=str(REPO / "cache/yolo26l.pt"))
    ap.add_argument("--work", required=True, help="cache dir for detections and the colour model")
    ap.add_argument("--device", default="0")
    ap.add_argument("--devices", default="0", help="run: comma-separated GPU indices")
    ap.add_argument("--procs", type=int, default=1, help="run: shard processes per device")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=500, help="frames per shard chunk (work split only)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-hash", action="store_true")
    args = ap.parse_args()
    if args.stage in ("assemble", "all", "run") and not args.out:
        raise SystemExit("--out required")
    Path(args.work).mkdir(parents=True, exist_ok=True)
    if not args.skip_hash and args.stage in ("balls", "persons", "all", "run"):
        for key, path in (("ball", args.ball_weights), ("person", args.person_weights)):
            if sha256(path) != EXPECTED_SHA256[key]:
                raise SystemExit("{} weights sha256 mismatch: {} (run tools/fetch_video_weights.py)".format(key, path))
    devices = [d.strip() for d in (args.devices if args.stage == "run" else args.device).split(",") if d.strip()]
    constants = match_constants(args.work, devices, args.data_dir, Path(args.ball_weights).parent)
    if args.stage == "run":
        run_sharded(args)
    frames = target_frames(args.frames, load_ball(args.data_dir), constants["masked_windows"])
    mats = load_homographies(args.data_dir) if args.stage != "persons" else None
    if args.stage in ("balls", "all"):
        stage_balls(args, frames, mats)
    if args.stage in ("persons", "all"):
        stage_persons(args, frames, constants)
    if args.stage in ("assemble", "all", "run"):
        pos, team, gk_slots = load_tracking(args.data_dir)
        stage_assemble(args, frames, pos, team, gk_slots, mats, constants)


if __name__ == "__main__":
    main()
