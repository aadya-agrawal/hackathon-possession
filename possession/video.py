import csv
from dataclasses import asdict, dataclass

import numpy as np

from possession import AWAY, HOME


@dataclass
class VideoMetadata:
    frames: int
    fps: float
    width: int
    height: int


@dataclass
class RecoveryInterval:
    start_frame: int
    end_frame: int
    context_start_frame: int
    context_end_frame: int
    boundary_frame: int
    from_team: int
    to_team: int
    missing_frames: int
    boundary_distance_frames: int
    homography_valid_fraction: float
    median_reprojection_error: float
    left_anchor: bool
    right_anchor: bool
    priority: float

    def to_dict(self):
        result = asdict(self)
        result["from_team"] = "H" if self.from_team == HOME else "A"
        result["to_team"] = "H" if self.to_team == HOME else "A"
        return result


@dataclass
class TemporalBallPath:
    xy: np.ndarray
    confidence: np.ndarray
    candidate_index: np.ndarray
    total_cost: float


@dataclass
class KalmanBallPath:
    xy: np.ndarray
    velocity: np.ndarray
    position_std: np.ndarray
    candidate_index: np.ndarray
    observation_confidence: np.ndarray
    total_cost: float


@dataclass
class KalmanTrackletPath:
    xy: np.ndarray
    velocity: np.ndarray
    position_std: np.ndarray
    candidate_index: np.ndarray
    observation_confidence: np.ndarray
    tracklets: list


@dataclass
class _KalmanHypothesis:
    state: np.ndarray
    covariance: np.ndarray
    cost: float
    missing_frames: int
    parent: object
    candidate_index: int


def _project(matrix, x, y):
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return np.asarray([
        (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2])
        / denominator,
        (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])
        / denominator,
    ])


def inspect_video(path):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "video diagnostics require OpenCV; install opencv-python"
        ) from error

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("{}: cannot open video".format(path))
    try:
        return VideoMetadata(
            frames=int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            height=int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
    finally:
        capture.release()


def score_player_relative_candidates(
    candidates_by_frame,
    frame_indexes,
    player_positions,
    *,
    pitch,
    distance_scale,
    distance_weight,
):
    if len(candidates_by_frame) != len(frame_indexes):
        raise ValueError("candidate frames and frame indexes must have equal lengths")
    if distance_scale <= 0 or distance_weight < 0:
        raise ValueError("invalid player-relative scoring parameters")
    pitch_scale = np.asarray(pitch, dtype=np.float64)
    if pitch_scale.shape != (2,) or np.any(pitch_scale <= 0):
        raise ValueError("pitch must contain positive length and width")

    ranked_frames = []
    for frame, candidates in zip(frame_indexes, candidates_by_frame):
        if int(frame) not in player_positions:
            raise ValueError(
                "missing player positions for frame {}".format(frame)
            )
        teams = player_positions[int(frame)]
        ranked = []
        for candidate_index, candidate in enumerate(candidates):
            normalized = np.asarray(
                candidate["normalized_pitch"], dtype=np.float64
            )
            confidence = float(candidate["confidence"])
            if (
                normalized.shape != (2,)
                or not np.isfinite(normalized).all()
                or not np.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                raise ValueError(
                    "video candidates must have finite pitch and confidence"
                )
            pitch_xy = normalized * pitch_scale
            team_distances = []
            for positions in teams:
                positions = np.asarray(positions, dtype=np.float64)
                if positions.size == 0:
                    team_distances.append(float("inf"))
                else:
                    team_distances.append(float(np.min(
                        np.linalg.norm(positions - pitch_xy, axis=1)
                    )))
            nearest_team = int(np.argmin(team_distances))
            nearest_distance = team_distances[nearest_team]
            distance_cost = np.log1p(nearest_distance / distance_scale)
            score = (
                np.log(max(confidence, 1e-6))
                - distance_weight * distance_cost
            )
            scored = dict(candidate)
            scored.update({
                "candidate_index": candidate_index,
                "pitch": pitch_xy,
                "nearest_player_distance": nearest_distance,
                "nearest_player_team": nearest_team,
                "team_player_distances": np.asarray(
                    team_distances, dtype=np.float64
                ),
                "player_relative_score": float(score),
            })
            ranked.append(scored)
        ranked.sort(key=lambda candidate: (
            -candidate["player_relative_score"],
            candidate["candidate_index"],
        ))
        ranked_frames.append(ranked)
    return ranked_frames


def load_homography_quality(path, n_frames):
    valid = np.zeros(n_frames, dtype=bool)
    reprojection_error = np.full(n_frames, np.nan, dtype=np.float32)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "reprojection_error", "valid"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                "{}: missing homography columns {}".format(path, missing)
            )
        count = 0
        for expected, row in enumerate(reader):
            frame = int(row["frame"])
            if frame != expected:
                raise ValueError(
                    "{}: expected homography frame {}, found {}".format(
                        path, expected, frame
                    )
                )
            if frame >= n_frames:
                raise ValueError(
                    "{}: more than {} homography rows".format(path, n_frames)
                )
            valid[frame] = bool(int(row["valid"]))
            if row["reprojection_error"]:
                reprojection_error[frame] = float(row["reprojection_error"])
            count += 1
    if count != n_frames:
        raise ValueError(
            "{}: expected {} homography rows, found {}".format(
                path, n_frames, count
            )
        )
    return reprojection_error, valid


def load_homography_matrices(path, n_frames):
    columns = [
        "h{}{}".format(row, column)
        for row in range(3)
        for column in range(3)
    ]
    matrices = np.full((n_frames, 3, 3), np.nan, dtype=np.float64)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "valid"} | set(columns)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                "{}: missing homography columns {}".format(path, missing)
            )
        count = 0
        for expected, row in enumerate(reader):
            frame = int(row["frame"])
            if frame != expected or frame >= n_frames:
                raise ValueError(
                    "{}: invalid homography frame {}".format(path, frame)
                )
            if bool(int(row["valid"])):
                matrices[frame] = np.asarray(
                    [float(row[name]) for name in columns]
                ).reshape(3, 3)
            count += 1
    if count != n_frames:
        raise ValueError(
            "{}: expected {} homography rows, found {}".format(
                path, n_frames, count
            )
        )
    return matrices


def _tile_regions(width, height, overlap_fraction):
    if not 0.0 <= overlap_fraction < 0.5:
        raise ValueError("tile overlap fraction must be in [0, 0.5)")
    overlap_x = int(round(width * overlap_fraction))
    overlap_y = int(round(height * overlap_fraction))
    tile_width = (width + overlap_x + 1) // 2
    tile_height = (height + overlap_y + 1) // 2
    return [
        (0, 0, tile_width, tile_height),
        (width - tile_width, 0, width, tile_height),
        (0, height - tile_height, tile_width, height),
        (width - tile_width, height - tile_height, width, height),
    ]


def _tiled_ball_predictions(
    image,
    model,
    *,
    class_id,
    confidence,
    image_size,
    device,
    overlap_fraction,
):
    height, width = image.shape[:2]
    regions = _tile_regions(width, height, overlap_fraction)
    tiles = [
        image[y0:y1, x0:x1]
        for x0, y0, x1, y1 in regions
    ]
    results = model.predict(
        tiles,
        classes=[class_id],
        conf=confidence,
        imgsz=image_size,
        device=device,
        batch=1,
        verbose=False,
    )
    if len(results) != len(regions):
        raise ValueError("detector did not return one result per tile")
    for (x0, y0, _, _), result in zip(regions, results):
        if result.boxes is None:
            continue
        centers = result.boxes.xywh[:, :2].cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for center, score in zip(centers, scores):
            yield (
                np.asarray(
                    [center[0] + x0, center[1] + y0], dtype=np.float64
                ),
                float(score),
            )


def _merge_pixel_candidate(candidates, candidate, merge_radius_pixels):
    duplicate = next((
        existing
        for existing in candidates
        if np.linalg.norm(existing["pixel"] - candidate["pixel"])
        <= merge_radius_pixels
    ), None)
    if duplicate is None:
        candidates.append(candidate)
    elif candidate["confidence"] > duplicate["confidence"]:
        duplicate.update(candidate)


def detect_tiled_ball_pixels(
    image,
    model,
    *,
    class_id,
    confidence,
    image_size,
    device,
    overlap_fraction,
    merge_radius_pixels,
):
    candidates = []
    for pixel, score in _tiled_ball_predictions(
        image, model, class_id=class_id, confidence=confidence,
        image_size=image_size, device=device,
        overlap_fraction=overlap_fraction,
    ):
        _merge_pixel_candidate(
            candidates, {"pixel": pixel, "confidence": score},
            merge_radius_pixels,
        )
    return candidates


def detect_tiled_ball_candidates(
    image,
    matrix,
    model,
    *,
    class_id,
    confidence,
    image_size,
    device,
    overlap_fraction,
    merge_radius_pixels,
    pitch_margin,
):
    inverse = np.linalg.inv(matrix)
    candidates = []
    for pixel, score in _tiled_ball_predictions(
        image, model, class_id=class_id, confidence=confidence,
        image_size=image_size, device=device,
        overlap_fraction=overlap_fraction,
    ):
        normalized = _project(inverse, pixel[0], pixel[1])
        if not (
            -pitch_margin <= normalized[0] <= 1.0 + pitch_margin
            and -pitch_margin <= normalized[1] <= 1.0 + pitch_margin
        ):
            continue
        _merge_pixel_candidate(
            candidates, {
                "pixel": pixel,
                "normalized_pitch": normalized,
                "confidence": score,
            },
            merge_radius_pixels,
        )
    return candidates


def select_input_ball_anchor_samples(
    raw_ball_xy, homography_error, homography_valid, half_blocks,
    *,
    fps, minimum_gap_frames, maximum_homography_error,
    maximum_ball_speed, stable_sample_count,
):
    n_frames = len(raw_ball_xy)
    if (
        np.shape(raw_ball_xy) != (n_frames, 2)
        or np.shape(homography_error) != (n_frames,)
        or np.shape(homography_valid) != (n_frames,)
    ):
        raise ValueError("ball-anchor audit arrays must share the video frame count")
    if (
        fps <= 0 or minimum_gap_frames < 1
        or maximum_homography_error <= 0 or maximum_ball_speed <= 0
        or stable_sample_count < 0
    ):
        raise ValueError("invalid ball-anchor audit limits")
    if not half_blocks:
        raise ValueError("ball-anchor audit requires half boundaries")

    in_half = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_blocks:
        if first < 0 or last >= n_frames or first > last:
            raise ValueError("invalid ball-anchor audit half boundaries")
        if np.any(in_half[first:last + 1]):
            raise ValueError("ball-anchor audit halves overlap")
        in_half[first:last + 1] = True
    present = np.isfinite(raw_ball_xy).all(axis=1)
    missing = ~present
    starts = np.flatnonzero(missing & np.r_[True, present[:-1]])
    stops = np.flatnonzero(missing & np.r_[present[1:], True]) + 1
    quality = (
        np.asarray(homography_valid, dtype=bool)
        & np.isfinite(homography_error)
        & (homography_error <= maximum_homography_error)
    )

    samples = []
    long_runs = 0
    in_half_runs = 0
    for start, stop in zip(starts, stops):
        length = int(stop - start)
        if length < minimum_gap_frames:
            continue
        long_runs += 1
        if start == 0 or stop >= n_frames:
            continue
        half = next((
            half for half, first, last in half_blocks
            if first <= start - 1 and stop <= last
        ), None)
        if half is None:
            continue
        in_half_runs += 1
        for side, frame, direction in (
            ("left", start - 1, -1), ("right", stop, 1),
        ):
            if not quality[frame]:
                continue
            support = 0
            speeds = []
            previous = None
            for step in range(5):
                neighbor = frame + direction * step
                if (
                    neighbor < 0 or neighbor >= n_frames
                    or not in_half[neighbor] or not present[neighbor]
                ):
                    break
                if previous is not None:
                    speeds.append(
                        float(np.linalg.norm(
                            raw_ball_xy[neighbor] - raw_ball_xy[previous]
                        ) * fps)
                    )
                support += 1
                previous = neighbor
            local_max_speed = max(speeds) if speeds else None
            samples.append({
                "frame": int(frame),
                "half": half,
                "kind": "endpoint",
                "side": side,
                "gap_frames": length,
                "support_frames": support,
                "maximum_local_speed_mps": local_max_speed,
                "supported": (
                    support >= 3
                    and local_max_speed is not None
                    and local_max_speed <= maximum_ball_speed
                ),
            })

    stable = in_half & present & quality
    if n_frames < 11:
        stable[:] = False
    else:
        for offset in range(1, 6):
            stable[offset:] &= present[:-offset]
            stable[:n_frames - offset] &= present[offset:]
            stable[:offset] = False
            stable[n_frames - offset:] = False
    stable_count = 0
    for index, (half, first, last) in enumerate(half_blocks):
        quota = (
            stable_sample_count // len(half_blocks)
            + (index < stable_sample_count % len(half_blocks))
        )
        stable_frames = np.flatnonzero(stable[first:last + 1]) + first
        if len(stable_frames) > quota:
            if quota:
                indexes = np.linspace(
                    0, len(stable_frames) - 1, quota, dtype=int
                )
                stable_frames = stable_frames[indexes]
            else:
                stable_frames = stable_frames[:0]
        stable_count += len(stable_frames)
        samples.extend({
            "frame": int(frame),
            "half": half,
            "kind": "stable",
            "side": None,
            "gap_frames": 0,
            "support_frames": 11,
            "maximum_local_speed_mps": None,
            "supported": True,
        } for frame in stable_frames)
    samples.sort(key=lambda item: item["frame"])
    endpoint_samples = [
        sample for sample in samples if sample["kind"] == "endpoint"
    ]
    local_speeds = [
        sample["maximum_local_speed_mps"]
        for sample in endpoint_samples
        if sample["maximum_local_speed_mps"] is not None
    ]
    return samples, {
        "raw_observed_frames": int(np.count_nonzero(present)),
        "all_missing_runs": len(starts),
        "minimum_gap_frames": minimum_gap_frames,
        "long_missing_runs": long_runs,
        "long_runs_inside_halves": in_half_runs,
        "homography_qualified_endpoints": sum(
            sample["kind"] == "endpoint" for sample in samples
        ),
        "endpoints_with_three_motion_supported_frames": sum(
            sample["kind"] == "endpoint" and sample["supported"]
            for sample in samples
        ),
        "endpoint_support_frame_counts": {
            str(count): sum(
                item["support_frames"] == count for item in endpoint_samples
            )
            for count in range(1, 6)
        },
        "maximum_local_step_speed_mps_p50": (
            float(np.percentile(local_speeds, 50)) if local_speeds else None
        ),
        "maximum_local_step_speed_mps_p90": (
            float(np.percentile(local_speeds, 90)) if local_speeds else None
        ),
        "stable_reference_frames": stable_count,
    }


def make_long_gap_edge_dropout(raw_ball_xy, endpoints, half_blocks,
                               minimum_gap_frames, edge_width_frames):
    if minimum_gap_frames < 1 or edge_width_frames < 1:
        raise ValueError("edge-dropout duration and width must be positive")
    bounds = {
        half: (first, last) for half, first, last in half_blocks
    }
    observed = np.isfinite(raw_ball_xy).all(axis=1)
    eligible = [
        item for item in endpoints
        if item["kind"] == "endpoint"
        and item["gap_frames"] >= minimum_gap_frames
    ]
    if not eligible:
        raise ValueError(
            "{}-frame stress has no eligible observed gap endpoints".format(
                minimum_gap_frames
            )
        )
    hidden = np.zeros(len(raw_ball_xy), dtype=bool)
    for item in eligible:
        first, last = bounds[item["half"]]
        direction = -1 if item["side"] == "left" else 1
        for offset in range(edge_width_frames):
            frame = item["frame"] + direction * offset
            if first <= frame <= last and observed[frame]:
                hidden[frame] = True
    perturbed = raw_ball_xy.copy()
    perturbed[hidden] = np.nan
    return perturbed, hidden, eligible


def select_balanced_long_gap_runs(
    samples, *, minimum_gap_frames=269, extreme_gap_frames=938,
    runs_per_cohort=4,
):
    if (
        minimum_gap_frames < 1 or extreme_gap_frames <= minimum_gap_frames
        or runs_per_cohort < 1
    ):
        raise ValueError("invalid input-only long-gap sampling limits")
    paired = {}
    for sample in samples:
        if sample["kind"] != "endpoint" or sample["gap_frames"] < minimum_gap_frames:
            continue
        start = (
            sample["frame"] + 1 if sample["side"] == "left"
            else sample["frame"] - sample["gap_frames"]
        )
        key = (sample["half"], start, sample["gap_frames"])
        paired.setdefault(key, set()).add(sample["side"])
    cohorts = {}
    for (half, start, length), sides in paired.items():
        if sides != {"left", "right"}:
            continue
        band = (
            "{}_plus".format(extreme_gap_frames)
            if length >= extreme_gap_frames
            else "{}_to_{}".format(minimum_gap_frames, extreme_gap_frames - 1)
        )
        cohorts.setdefault((half, band), []).append((start, length))
    selected = []
    for (half, band), options in sorted(cohorts.items()):
        options.sort()
        if len(options) < runs_per_cohort:
            raise ValueError(
                "input-only {} half {} has only {} eligible gaps".format(
                    band, half, len(options)
                )
            )
        indexes = np.linspace(
            0, len(options) - 1, runs_per_cohort, dtype=int
        )
        selected.extend({
            "half": half,
            "band": band,
            "start_frame": int(options[index][0]),
            "gap_frames": int(options[index][1]),
        } for index in indexes)
    if not selected:
        raise ValueError("no input-only paired long-gap cohorts available")
    return selected


def audit_input_ball_anchor_video(
    video_path, model_path, samples, raw_ball_xy, matrices, pitch,
    *,
    fps, device,
):
    try:
        import cv2
        from ultralytics import YOLO, settings
    except ImportError as error:
        raise RuntimeError(
            "ball-anchor input audit requires OpenCV and Ultralytics"
        ) from error
    metadata = inspect_video(video_path)
    if (
        metadata.frames != len(raw_ball_xy)
        or len(matrices) != len(raw_ball_xy)
        or abs(metadata.fps - fps) > 1e-6
    ):
        raise ValueError("ball-anchor audit video, tracking and homography are misaligned")
    settings.update({"sync": False})
    model = YOLO(model_path)
    training = (model.ckpt or {}).get("train_args", {}).get("data")
    if (
        str(training).split("/")[-1] != "coco.yaml"
        or model.names.get(32) != "sports ball"
    ):
        raise ValueError("ball-anchor audit requires a generic COCO sports-ball model")

    groups = {
        name: {
            "sampled_frames": 0,
            "on_screen_frames": 0,
            "frames_with_candidates": 0,
            "nearest_match_20px": 0,
            "nearest_match_40px": 0,
            "nearest_match_80px": 0,
            "highest_confidence_match_40px": 0,
            "nearest_distances_when_detected": [],
        }
        for name in (
            "all_endpoints", "motion_supported_endpoints",
            "other_endpoints", "long_250_plus_endpoints",
            "very_long_938_plus_endpoints",
            "left_endpoints", "right_endpoints", "stable_reference",
        )
    }
    for half in {sample["half"] for sample in samples}:
        groups["half_{}_endpoints".format(half)] = {
            key: [] if key == "nearest_distances_when_detected" else 0
            for key in groups["all_endpoints"]
        }
        groups["half_{}_stable".format(half)] = {
            key: [] if key == "nearest_distances_when_detected" else 0
            for key in groups["all_endpoints"]
        }
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("{}: cannot open video".format(video_path))
    try:
        for sample in samples:
            frame = sample["frame"]
            if not 0 <= frame < len(raw_ball_xy):
                raise ValueError("ball-anchor sample frame out of range")
            names = (
                ["stable_reference", "half_{}_stable".format(sample["half"])]
                if sample["kind"] == "stable"
                else [
                    "all_endpoints",
                    "half_{}_endpoints".format(sample["half"]),
                    "motion_supported_endpoints" if sample["supported"]
                    else "other_endpoints",
                    "long_250_plus_endpoints" if sample["gap_frames"] >= 250
                    else None,
                    "very_long_938_plus_endpoints"
                    if sample["gap_frames"] >= 938 else None,
                    "{}_endpoints".format(sample["side"]),
                ]
            )
            selected_groups = [groups[name] for name in names if name is not None]
            for group in selected_groups:
                group["sampled_frames"] += 1
            pixel = _project(
                matrices[frame],
                raw_ball_xy[frame, 0] / pitch[0],
                raw_ball_xy[frame, 1] / pitch[1],
            )
            if not (
                np.isfinite(pixel).all()
                and 0 <= pixel[0] < metadata.width
                and 0 <= pixel[1] < metadata.height
            ):
                continue
            for group in selected_groups:
                group["on_screen_frames"] += 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            success, image = capture.read()
            if (
                not success
                or int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != frame + 1
            ):
                raise ValueError("{}: cannot decode exact frame {}".format(
                    video_path, frame
                ))
            candidates = detect_tiled_ball_pixels(
                image, model, class_id=32, confidence=0.03,
                image_size=1280, device=device, overlap_fraction=0.1,
                merge_radius_pixels=20,
            )
            if not candidates:
                continue
            distances = [
                float(np.linalg.norm(candidate["pixel"] - pixel))
                for candidate in candidates
            ]
            nearest = min(distances)
            strongest = max(
                range(len(candidates)),
                key=lambda index: candidates[index]["confidence"],
            )
            for group in selected_groups:
                group["frames_with_candidates"] += 1
                group["nearest_distances_when_detected"].append(nearest)
                for radius in (20, 40, 80):
                    group["nearest_match_{}px".format(radius)] += nearest <= radius
                group["highest_confidence_match_40px"] += (
                    distances[strongest] <= 40
                )
    finally:
        capture.release()

    for group in groups.values():
        distances = group.pop("nearest_distances_when_detected")
        group["nearest_distance_p50_px_if_detected"] = (
            float(np.median(distances)) if distances else None
        )
        group["nearest_distance_p90_px_if_detected"] = (
            float(np.percentile(distances, 90)) if distances else None
        )
    return {
        "model_training_data": training,
        "video_fps": metadata.fps,
        "video_frames": metadata.frames,
        "video_width": metadata.width,
        "video_height": metadata.height,
        "groups": groups,
    }


def select_temporal_ball_path(
    candidates_by_frame,
    left_anchor,
    right_anchor,
    *,
    fps,
    pitch,
    max_gap_frames,
    max_speed,
    max_acceleration,
    confidence_reward=2.0,
    gap_penalty=1.0,
    speed_weight=0.25,
    acceleration_weight=0.5,
):
    n_frames = len(candidates_by_frame)
    empty = TemporalBallPath(
        xy=np.full((n_frames, 2), np.nan, dtype=np.float32),
        confidence=np.zeros(n_frames, dtype=np.float32),
        candidate_index=np.full(n_frames, -1, dtype=np.int32),
        total_cost=float("inf"),
    )
    if n_frames == 0:
        return empty
    if fps <= 0 or max_gap_frames < 0:
        raise ValueError("fps must be positive and max gap must be non-negative")
    if max_speed <= 0 or max_acceleration <= 0:
        raise ValueError("speed and acceleration limits must be positive")

    pitch_scale = np.asarray(pitch, dtype=np.float64)
    if pitch_scale.shape != (2,) or np.any(pitch_scale <= 0):
        raise ValueError("pitch must contain positive length and width")

    left_frame, left_xy = left_anchor
    right_frame, right_xy = right_anchor
    left_xy = np.asarray(left_xy, dtype=np.float64)
    right_xy = np.asarray(right_xy, dtype=np.float64)
    if (
        left_xy.shape != (2,)
        or right_xy.shape != (2,)
        or not np.isfinite(left_xy).all()
        or not np.isfinite(right_xy).all()
        or left_frame >= 0
        or right_frame < n_frames
    ):
        raise ValueError("anchors must be finite and bracket the candidate frames")

    nodes = [{
        "frame": int(left_frame),
        "xy": left_xy,
        "confidence": 1.0,
        "local_frame": -1,
        "candidate_index": -1,
    }]
    for frame, candidates in enumerate(candidates_by_frame):
        for candidate_index, candidate in enumerate(candidates):
            normalized = np.asarray(candidate["normalized_pitch"], dtype=np.float64)
            confidence = float(candidate["confidence"])
            if (
                normalized.shape != (2,)
                or not np.isfinite(normalized).all()
                or not np.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                raise ValueError("video candidates must have finite pitch and confidence")
            nodes.append({
                "frame": frame,
                "xy": normalized * pitch_scale,
                "confidence": confidence,
                "local_frame": frame,
                "candidate_index": candidate_index,
            })
    right_id = len(nodes)
    nodes.append({
        "frame": int(right_frame),
        "xy": right_xy,
        "confidence": 1.0,
        "local_frame": n_frames,
        "candidate_index": -1,
    })
    if right_id == 1:
        return empty

    maximum_step_frames = max_gap_frames + 1

    def edge(first, second):
        elapsed_frames = nodes[second]["frame"] - nodes[first]["frame"]
        if elapsed_frames <= 0 or elapsed_frames > maximum_step_frames:
            return None
        velocity = (
            (nodes[second]["xy"] - nodes[first]["xy"])
            * fps
            / float(elapsed_frames)
        )
        speed = float(np.linalg.norm(velocity))
        if speed > max_speed:
            return None
        cost = (
            gap_penalty * max(0, elapsed_frames - 1)
            + speed_weight * (speed / max_speed) ** 2
        )
        return velocity, cost

    costs = {}
    predecessors = {}
    for candidate_id in range(1, right_id):
        first_edge = edge(0, candidate_id)
        if first_edge is None:
            continue
        _, edge_cost = first_edge
        state = (0, candidate_id)
        costs[state] = (
            edge_cost
            - confidence_reward * nodes[candidate_id]["confidence"]
        )
        predecessors[state] = None

    for middle_id in range(1, right_id):
        incoming_states = [
            state for state in costs if state[1] == middle_id
        ]
        if not incoming_states:
            continue
        for next_id in range(middle_id + 1, right_id):
            if (
                nodes[next_id]["frame"] - nodes[middle_id]["frame"]
                > maximum_step_frames
            ):
                break
            next_edge = edge(middle_id, next_id)
            if next_edge is None:
                continue
            next_velocity, next_edge_cost = next_edge
            for state in incoming_states:
                previous_edge = edge(*state)
                previous_velocity = previous_edge[0]
                elapsed = (
                    nodes[next_id]["frame"] - nodes[state[0]]["frame"]
                ) / (2.0 * fps)
                acceleration = float(
                    np.linalg.norm(next_velocity - previous_velocity) / elapsed
                )
                if acceleration > max_acceleration:
                    continue
                new_state = (middle_id, next_id)
                new_cost = (
                    costs[state]
                    + next_edge_cost
                    + acceleration_weight
                    * (acceleration / max_acceleration) ** 2
                    - confidence_reward * nodes[next_id]["confidence"]
                )
                if (
                    new_state not in costs
                    or new_cost < costs[new_state] - 1e-12
                ):
                    costs[new_state] = new_cost
                    predecessors[new_state] = state

    best_state = None
    best_cost = float("inf")
    for state, path_cost in costs.items():
        final_edge = edge(state[1], right_id)
        previous_edge = edge(*state)
        if final_edge is None or previous_edge is None:
            continue
        final_velocity, final_edge_cost = final_edge
        elapsed = (
            nodes[right_id]["frame"] - nodes[state[0]]["frame"]
        ) / (2.0 * fps)
        acceleration = float(
            np.linalg.norm(final_velocity - previous_edge[0]) / elapsed
        )
        if acceleration > max_acceleration:
            continue
        total_cost = (
            path_cost
            + final_edge_cost
            + acceleration_weight * (acceleration / max_acceleration) ** 2
        )
        if total_cost < best_cost - 1e-12:
            best_cost = total_cost
            best_state = state

    if best_state is None:
        return empty

    selected_nodes = [best_state[1]]
    state = best_state
    while predecessors[state] is not None:
        selected_nodes.append(state[0])
        state = predecessors[state]
    selected_nodes.reverse()

    result = empty
    for node_id in selected_nodes:
        node = nodes[node_id]
        frame = node["local_frame"]
        result.xy[frame] = node["xy"]
        result.confidence[frame] = node["confidence"]
        result.candidate_index[frame] = node["candidate_index"]
    result.total_cost = float(best_cost)
    return result


def select_kalman_ball_path(
    candidates_by_frame,
    left_anchor,
    right_anchor,
    *,
    fps,
    pitch,
    beam_width,
    maximum_missing_frames,
    process_acceleration_std,
    measurement_noise_base,
    measurement_noise_minimum,
    measurement_noise_maximum,
    initial_velocity_std,
    innovation_gate,
    missing_penalty,
    confidence_reward,
):
    n_frames = len(candidates_by_frame)

    def empty_result():
        return KalmanBallPath(
            xy=np.full((n_frames, 2), np.nan, dtype=np.float32),
            velocity=np.full((n_frames, 2), np.nan, dtype=np.float32),
            position_std=np.full(n_frames, np.inf, dtype=np.float32),
            candidate_index=np.full(n_frames, -1, dtype=np.int32),
            observation_confidence=np.zeros(n_frames, dtype=np.float32),
            total_cost=float("inf"),
        )

    if n_frames == 0:
        return empty_result()
    if fps <= 0 or beam_width < 1 or maximum_missing_frames < 0:
        raise ValueError("invalid Kalman timing or beam configuration")
    positive_parameters = (
        process_acceleration_std,
        measurement_noise_base,
        measurement_noise_minimum,
        measurement_noise_maximum,
        initial_velocity_std,
        innovation_gate,
    )
    if any(value <= 0 for value in positive_parameters):
        raise ValueError("Kalman noise and gate parameters must be positive")
    if measurement_noise_minimum > measurement_noise_maximum:
        raise ValueError("minimum measurement noise exceeds maximum")
    if missing_penalty < 0 or confidence_reward < 0:
        raise ValueError("Kalman path costs must be non-negative")

    pitch_scale = np.asarray(pitch, dtype=np.float64)
    left_frame, left_xy = left_anchor
    right_frame, right_xy = right_anchor
    left_xy = np.asarray(left_xy, dtype=np.float64)
    right_xy = np.asarray(right_xy, dtype=np.float64)
    if (
        pitch_scale.shape != (2,)
        or np.any(pitch_scale <= 0)
        or left_xy.shape != (2,)
        or right_xy.shape != (2,)
        or not np.isfinite(left_xy).all()
        or not np.isfinite(right_xy).all()
        or left_frame != -1
        or right_frame != n_frames
    ):
        raise ValueError(
            "Kalman anchors must be finite and immediately bracket the frames"
        )

    candidate_positions = []
    candidate_confidences = []
    for candidates in candidates_by_frame:
        positions = []
        confidences = []
        for candidate in candidates:
            normalized = np.asarray(
                candidate["normalized_pitch"], dtype=np.float64
            )
            confidence = float(candidate["confidence"])
            if (
                normalized.shape != (2,)
                or not np.isfinite(normalized).all()
                or not np.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                raise ValueError(
                    "video candidates must have finite pitch and confidence"
                )
            positions.append(normalized * pitch_scale)
            confidences.append(confidence)
        candidate_positions.append(positions)
        candidate_confidences.append(confidences)

    dt = 1.0 / float(fps)
    transition = np.asarray([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    acceleration_mapping = np.asarray([
        [0.5 * dt * dt, 0.0],
        [0.0, 0.5 * dt * dt],
        [dt, 0.0],
        [0.0, dt],
    ])
    process_covariance = (
        acceleration_mapping
        @ acceleration_mapping.T
        * process_acceleration_std ** 2
    )
    observation = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    identity = np.eye(4, dtype=np.float64)

    def measurement_std(confidence):
        scaled = measurement_noise_base / np.sqrt(max(confidence, 1e-3))
        return float(np.clip(
            scaled,
            measurement_noise_minimum,
            measurement_noise_maximum,
        ))

    def predict(state, covariance):
        return (
            transition @ state,
            transition @ covariance @ transition.T + process_covariance,
        )

    def update(state, covariance, measurement, noise_std):
        measurement_covariance = np.eye(2) * noise_std ** 2
        innovation = measurement - observation @ state
        innovation_covariance = (
            observation @ covariance @ observation.T
            + measurement_covariance
        )
        solved = np.linalg.solve(innovation_covariance, innovation)
        distance_squared = float(innovation @ solved)
        if distance_squared > innovation_gate ** 2:
            return None
        gain = np.linalg.solve(
            innovation_covariance,
            observation @ covariance,
        ).T
        updated_state = state + gain @ innovation
        residual = identity - gain @ observation
        updated_covariance = (
            residual @ covariance @ residual.T
            + gain @ measurement_covariance @ gain.T
        )
        _, log_determinant = np.linalg.slogdet(innovation_covariance)
        return (
            updated_state,
            updated_covariance,
            0.5 * (distance_squared + log_determinant),
        )

    initial_state = np.asarray(
        [left_xy[0], left_xy[1], 0.0, 0.0], dtype=np.float64
    )
    initial_covariance = np.diag([
        measurement_noise_minimum ** 2,
        measurement_noise_minimum ** 2,
        initial_velocity_std ** 2,
        initial_velocity_std ** 2,
    ])
    beam = [_KalmanHypothesis(
        state=initial_state,
        covariance=initial_covariance,
        cost=0.0,
        missing_frames=0,
        parent=None,
        candidate_index=-1,
    )]

    for frame in range(n_frames):
        branches = []
        for hypothesis in beam:
            predicted_state, predicted_covariance = predict(
                hypothesis.state, hypothesis.covariance
            )
            if hypothesis.missing_frames < maximum_missing_frames:
                branches.append(_KalmanHypothesis(
                    state=predicted_state,
                    covariance=predicted_covariance,
                    cost=hypothesis.cost + missing_penalty,
                    missing_frames=hypothesis.missing_frames + 1,
                    parent=hypothesis,
                    candidate_index=-1,
                ))
            for candidate_index, (position, confidence) in enumerate(zip(
                candidate_positions[frame],
                candidate_confidences[frame],
            )):
                updated = update(
                    predicted_state,
                    predicted_covariance,
                    position,
                    measurement_std(confidence),
                )
                if updated is None:
                    continue
                state, covariance, likelihood_cost = updated
                branches.append(_KalmanHypothesis(
                    state=state,
                    covariance=covariance,
                    cost=(
                        hypothesis.cost
                        + likelihood_cost
                        - confidence_reward * confidence
                    ),
                    missing_frames=0,
                    parent=hypothesis,
                    candidate_index=candidate_index,
                ))
        if not branches:
            return empty_result()
        branches.sort(key=lambda item: (
            item.cost,
            item.missing_frames,
            item.candidate_index,
        ))
        beam = branches[:beam_width]

    final_candidates = []
    anchor_noise = measurement_noise_minimum
    for hypothesis in beam:
        predicted_state, predicted_covariance = predict(
            hypothesis.state, hypothesis.covariance
        )
        updated = update(
            predicted_state,
            predicted_covariance,
            right_xy,
            anchor_noise,
        )
        if updated is None:
            continue
        _, _, likelihood_cost = updated
        final_candidates.append((
            hypothesis.cost + likelihood_cost,
            hypothesis,
        ))
    if not final_candidates:
        return empty_result()
    final_candidates.sort(key=lambda item: item[0])
    total_cost, best = final_candidates[0]

    selections = np.full(n_frames, -1, dtype=np.int32)
    cursor = best
    for frame in range(n_frames - 1, -1, -1):
        selections[frame] = cursor.candidate_index
        cursor = cursor.parent

    filtered_states = []
    filtered_covariances = []
    predicted_states = []
    predicted_covariances = []
    observation_confidence = np.zeros(n_frames, dtype=np.float32)
    state = initial_state
    covariance = initial_covariance
    for frame, candidate_index in enumerate(selections):
        state, covariance = predict(state, covariance)
        predicted_states.append(state.copy())
        predicted_covariances.append(covariance.copy())
        if candidate_index >= 0:
            confidence = candidate_confidences[frame][candidate_index]
            updated = update(
                state,
                covariance,
                candidate_positions[frame][candidate_index],
                measurement_std(confidence),
            )
            state, covariance, _ = updated
            observation_confidence[frame] = confidence
        filtered_states.append(state.copy())
        filtered_covariances.append(covariance.copy())

    state, covariance = predict(state, covariance)
    predicted_states.append(state.copy())
    predicted_covariances.append(covariance.copy())
    state, covariance, _ = update(
        state, covariance, right_xy, anchor_noise
    )
    filtered_states.append(state.copy())
    filtered_covariances.append(covariance.copy())

    smoothed_states = [
        value.copy() for value in filtered_states
    ]
    smoothed_covariances = [
        value.copy() for value in filtered_covariances
    ]
    for frame in range(n_frames - 1, -1, -1):
        smoothing_gain = np.linalg.solve(
            predicted_covariances[frame + 1],
            transition @ filtered_covariances[frame],
        ).T
        smoothed_states[frame] = (
            filtered_states[frame]
            + smoothing_gain
            @ (
                smoothed_states[frame + 1]
                - predicted_states[frame + 1]
            )
        )
        smoothed_covariances[frame] = (
            filtered_covariances[frame]
            + smoothing_gain
            @ (
                smoothed_covariances[frame + 1]
                - predicted_covariances[frame + 1]
            )
            @ smoothing_gain.T
        )

    states = np.asarray(smoothed_states[:n_frames])
    position_std = np.sqrt(np.maximum(
        0.0,
        np.asarray([
            np.trace(value[:2, :2]) / 2.0
            for value in smoothed_covariances[:n_frames]
        ]),
    ))
    return KalmanBallPath(
        xy=states[:, :2].astype(np.float32),
        velocity=states[:, 2:].astype(np.float32),
        position_std=position_std.astype(np.float32),
        candidate_index=selections,
        observation_confidence=observation_confidence,
        total_cost=float(total_cost),
    )


def select_kalman_ball_tracklets(
    candidates_by_frame,
    *,
    fps,
    pitch,
    anchor_confidence,
    minimum_tracklet_frames,
    maximum_tracklet_frames,
    beam_width,
    maximum_missing_frames,
    process_acceleration_std,
    measurement_noise_base,
    measurement_noise_minimum,
    measurement_noise_maximum,
    initial_velocity_std,
    innovation_gate,
    missing_penalty,
    confidence_reward,
):
    n_frames = len(candidates_by_frame)
    result = KalmanTrackletPath(
        xy=np.full((n_frames, 2), np.nan, dtype=np.float32),
        velocity=np.full((n_frames, 2), np.nan, dtype=np.float32),
        position_std=np.full(n_frames, np.inf, dtype=np.float32),
        candidate_index=np.full(n_frames, -1, dtype=np.int32),
        observation_confidence=np.zeros(n_frames, dtype=np.float32),
        tracklets=[],
    )
    if n_frames == 0:
        return result
    if not 0.0 <= anchor_confidence <= 1.0:
        raise ValueError("tracklet anchor confidence must be between zero and one")
    if (
        minimum_tracklet_frames < 3
        or maximum_tracklet_frames < minimum_tracklet_frames
    ):
        raise ValueError("invalid tracklet duration limits")

    anchor_frames = [
        frame for frame, candidates in enumerate(candidates_by_frame)
        if any(
            float(candidate["confidence"]) >= anchor_confidence
            for candidate in candidates
        )
    ]
    pitch_scale = np.asarray(pitch, dtype=np.float64)

    def anchor_candidates(frame):
        return [
            (index, candidate)
            for index, candidate in enumerate(candidates_by_frame[frame])
            if float(candidate["confidence"]) >= anchor_confidence
        ]

    anchor_position_std = lambda confidence: float(np.clip(
        measurement_noise_base / np.sqrt(max(confidence, 1e-3)),
        measurement_noise_minimum,
        measurement_noise_maximum,
    ))

    anchor_cursor = 0
    while anchor_cursor < len(anchor_frames) - 1:
        start = anchor_frames[anchor_cursor]
        possible_end_indexes = [
            index
            for index in range(anchor_cursor + 1, len(anchor_frames))
            if minimum_tracklet_frames
            <= anchor_frames[index] - start + 1
            <= maximum_tracklet_frames
        ]
        accepted = None
        for end_index in reversed(possible_end_indexes):
            end = anchor_frames[end_index]
            interior = candidates_by_frame[start + 1:end]
            candidates = []
            for start_index, start_candidate in anchor_candidates(start):
                start_xy = (
                    np.asarray(
                        start_candidate["normalized_pitch"],
                        dtype=np.float64,
                    )
                    * pitch_scale
                )
                for end_candidate_index, end_candidate in anchor_candidates(end):
                    end_xy = (
                        np.asarray(
                            end_candidate["normalized_pitch"],
                            dtype=np.float64,
                        )
                        * pitch_scale
                    )
                    path = select_kalman_ball_path(
                        interior,
                        (-1, start_xy),
                        (len(interior), end_xy),
                        fps=fps,
                        pitch=pitch,
                        beam_width=beam_width,
                        maximum_missing_frames=maximum_missing_frames,
                        process_acceleration_std=process_acceleration_std,
                        measurement_noise_base=measurement_noise_base,
                        measurement_noise_minimum=measurement_noise_minimum,
                        measurement_noise_maximum=measurement_noise_maximum,
                        initial_velocity_std=initial_velocity_std,
                        innovation_gate=innovation_gate,
                        missing_penalty=missing_penalty,
                        confidence_reward=confidence_reward,
                    )
                    if not np.isfinite(path.total_cost):
                        continue
                    endpoint_cost = -confidence_reward * (
                        float(start_candidate["confidence"])
                        + float(end_candidate["confidence"])
                    )
                    candidates.append((
                        path.total_cost + endpoint_cost,
                        start_index,
                        start_candidate,
                        end_candidate_index,
                        end_candidate,
                        path,
                    ))
            if candidates:
                candidates.sort(key=lambda item: (
                    item[0], item[1], item[3]
                ))
                accepted = (end_index, end, candidates[0])
                break

        if accepted is None:
            anchor_cursor += 1
            continue

        end_index, end, candidate = accepted
        (
            total_cost,
            start_candidate_index,
            start_candidate,
            end_candidate_index,
            end_candidate,
            path,
        ) = candidate
        frames = np.arange(start, end + 1)
        result.xy[start] = (
            np.asarray(start_candidate["normalized_pitch"]) * pitch_scale
        )
        result.xy[start + 1:end] = path.xy
        result.xy[end] = (
            np.asarray(end_candidate["normalized_pitch"]) * pitch_scale
        )
        if len(path.velocity):
            result.velocity[start] = path.velocity[0]
            result.velocity[start + 1:end] = path.velocity
            result.velocity[end] = path.velocity[-1]
        result.position_std[start] = anchor_position_std(
            float(start_candidate["confidence"])
        )
        result.position_std[start + 1:end] = path.position_std
        result.position_std[end] = anchor_position_std(
            float(end_candidate["confidence"])
        )
        result.candidate_index[start] = start_candidate_index
        result.candidate_index[start + 1:end] = path.candidate_index
        result.candidate_index[end] = end_candidate_index
        result.observation_confidence[start] = float(
            start_candidate["confidence"]
        )
        result.observation_confidence[start + 1:end] = (
            path.observation_confidence
        )
        result.observation_confidence[end] = float(
            end_candidate["confidence"]
        )
        result.tracklets.append({
            "start_frame": int(start),
            "end_frame": int(end + 1),
            "frames": int(len(frames)),
            "observations": int(np.count_nonzero(
                result.candidate_index[frames] >= 0
            )),
            "maximum_position_std": float(np.max(
                result.position_std[frames]
            )),
            "total_cost": float(total_cost),
        })
        anchor_cursor = end_index
    return result


def select_anchored_kalman_tracklet(
    candidates_by_frame,
    anchor_xy,
    *,
    anchor_side,
    fps,
    pitch,
    endpoint_confidence,
    minimum_tracklet_frames,
    maximum_tracklet_frames,
    beam_width,
    maximum_missing_frames,
    process_acceleration_std,
    measurement_noise_base,
    measurement_noise_minimum,
    measurement_noise_maximum,
    initial_velocity_std,
    innovation_gate,
    missing_penalty,
    confidence_reward,
):
    n_frames = len(candidates_by_frame)
    result = KalmanTrackletPath(
        xy=np.full((n_frames, 2), np.nan, dtype=np.float32),
        velocity=np.full((n_frames, 2), np.nan, dtype=np.float32),
        position_std=np.full(n_frames, np.inf, dtype=np.float32),
        candidate_index=np.full(n_frames, -1, dtype=np.int32),
        observation_confidence=np.zeros(n_frames, dtype=np.float32),
        tracklets=[],
    )
    if anchor_side not in ("left", "right"):
        raise ValueError("anchor side must be left or right")
    if not 0.0 <= endpoint_confidence <= 1.0:
        raise ValueError("endpoint confidence must be between zero and one")
    if (
        minimum_tracklet_frames < 2
        or maximum_tracklet_frames < minimum_tracklet_frames
    ):
        raise ValueError("invalid anchored tracklet duration limits")
    if n_frames == 0:
        return result

    ordered = (
        list(candidates_by_frame)
        if anchor_side == "left"
        else list(reversed(candidates_by_frame))
    )
    maximum_index = min(n_frames, maximum_tracklet_frames) - 1
    minimum_index = minimum_tracklet_frames - 1
    pitch_scale = np.asarray(pitch, dtype=np.float64)
    accepted = None
    for endpoint_frame in range(maximum_index, minimum_index - 1, -1):
        endpoint_candidates = [
            (index, candidate)
            for index, candidate in enumerate(ordered[endpoint_frame])
            if float(candidate["confidence"]) >= endpoint_confidence
        ]
        if not endpoint_candidates:
            continue
        candidates = []
        interior = ordered[:endpoint_frame]
        for endpoint_index, endpoint_candidate in endpoint_candidates:
            endpoint_xy = (
                np.asarray(
                    endpoint_candidate["normalized_pitch"],
                    dtype=np.float64,
                )
                * pitch_scale
            )
            path = select_kalman_ball_path(
                interior,
                (-1, np.asarray(anchor_xy, dtype=np.float64)),
                (len(interior), endpoint_xy),
                fps=fps,
                pitch=pitch,
                beam_width=beam_width,
                maximum_missing_frames=maximum_missing_frames,
                process_acceleration_std=process_acceleration_std,
                measurement_noise_base=measurement_noise_base,
                measurement_noise_minimum=measurement_noise_minimum,
                measurement_noise_maximum=measurement_noise_maximum,
                initial_velocity_std=initial_velocity_std,
                innovation_gate=innovation_gate,
                missing_penalty=missing_penalty,
                confidence_reward=confidence_reward,
            )
            if np.isfinite(path.total_cost):
                candidates.append((
                    (
                        path.total_cost
                        - confidence_reward
                        * float(endpoint_candidate["confidence"])
                    ),
                    endpoint_index,
                    endpoint_candidate,
                    endpoint_xy,
                    path,
                ))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]))
            accepted = (endpoint_frame, candidates[0])
            break
    if accepted is None:
        return result

    endpoint_frame, candidate = accepted
    (
        total_cost,
        endpoint_index,
        endpoint_candidate,
        endpoint_xy,
        path,
    ) = candidate
    ordered_xy = np.vstack([path.xy, endpoint_xy])
    ordered_velocity = np.vstack([
        path.velocity,
        path.velocity[-1] if len(path.velocity) else np.zeros(2),
    ])
    endpoint_std = float(np.clip(
        measurement_noise_base / np.sqrt(max(
            float(endpoint_candidate["confidence"]), 1e-3
        )),
        measurement_noise_minimum,
        measurement_noise_maximum,
    ))
    ordered_std = np.r_[path.position_std, endpoint_std]
    ordered_indexes = np.r_[path.candidate_index, endpoint_index]
    ordered_confidence = np.r_[
        path.observation_confidence,
        float(endpoint_candidate["confidence"]),
    ]
    if anchor_side == "left":
        indexes = np.arange(endpoint_frame + 1)
        velocity = ordered_velocity
    else:
        indexes = np.arange(n_frames - 1, n_frames - endpoint_frame - 2, -1)
        velocity = -ordered_velocity
    result.xy[indexes] = ordered_xy
    result.velocity[indexes] = velocity
    result.position_std[indexes] = ordered_std
    result.candidate_index[indexes] = ordered_indexes
    result.observation_confidence[indexes] = ordered_confidence
    result.tracklets.append({
        "start_frame": int(np.min(indexes)),
        "end_frame": int(np.max(indexes) + 1),
        "frames": int(len(indexes)),
        "observations": int(np.count_nonzero(ordered_indexes >= 0)),
        "maximum_position_std": float(np.max(ordered_std)),
        "anchor_side": anchor_side,
        "total_cost": float(total_cost),
    })
    return result


def select_edge_anchored_ball_paths(
    candidates_by_frame,
    left_anchor,
    right_anchor,
    *,
    fps,
    pitch,
    beam_width,
    maximum_missing_frames,
    process_acceleration_std,
    measurement_noise_base,
    measurement_noise_minimum,
    measurement_noise_maximum,
    initial_velocity_std,
    innovation_gate,
    missing_penalty,
    confidence_reward,
):
    n_frames = len(candidates_by_frame)
    if not (
        np.isfinite(left_anchor).all()
        and np.isfinite(right_anchor).all()
    ):
        raise ValueError("edge-anchored ball paths require finite observed anchors")
    result = KalmanTrackletPath(
        xy=np.full((n_frames, 2), np.nan, dtype=np.float32),
        velocity=np.full((n_frames, 2), np.nan, dtype=np.float32),
        position_std=np.full(n_frames, np.inf, dtype=np.float32),
        candidate_index=np.full(n_frames, -1, dtype=np.int32),
        observation_confidence=np.zeros(n_frames, dtype=np.float32),
        tracklets=[],
    )
    middle = (n_frames + 1) // 2
    for start, stop, anchor, side in (
        (0, middle, left_anchor, "left"),
        (middle, n_frames, right_anchor, "right"),
    ):
        if stop - start < 4:
            continue
        path = select_anchored_kalman_tracklet(
            candidates_by_frame[start:stop],
            anchor,
            anchor_side=side,
            fps=fps,
            pitch=pitch,
            endpoint_confidence=0.1,
            minimum_tracklet_frames=4,
            maximum_tracklet_frames=min(25, stop - start),
            beam_width=beam_width,
            maximum_missing_frames=maximum_missing_frames,
            process_acceleration_std=process_acceleration_std,
            measurement_noise_base=measurement_noise_base,
            measurement_noise_minimum=measurement_noise_minimum,
            measurement_noise_maximum=measurement_noise_maximum,
            initial_velocity_std=initial_velocity_std,
            innovation_gate=innovation_gate,
            missing_penalty=missing_penalty,
            confidence_reward=confidence_reward,
        )
        result.xy[start:stop] = path.xy
        result.velocity[start:stop] = path.velocity
        result.position_std[start:stop] = path.position_std
        result.candidate_index[start:stop] = path.candidate_index
        result.observation_confidence[start:stop] = path.observation_confidence
        result.tracklets.extend({
            **tracklet,
            "start_frame": tracklet["start_frame"] + start,
            "end_frame": tracklet["end_frame"] + start,
        } for tracklet in path.tracklets)
    return result


def detect_interval_paths(
    video_path,
    model_path,
    intervals,
    source,
    trajectory_xy,
    matrices,
    *,
    fps,
    pitch,
    class_id,
    confidence,
    image_size,
    device,
    overlap_fraction,
    merge_radius_pixels,
    pitch_margin,
    maximum_gap_frames,
    maximum_speed,
    maximum_acceleration,
    confidence_reward,
    gap_penalty,
    speed_weight,
    acceleration_weight,
    path_mode="temporal",
    kalman_beam_width=24,
    kalman_maximum_missing_frames=5,
    kalman_process_acceleration_std=15.0,
    kalman_measurement_noise_base=0.5,
    kalman_measurement_noise_minimum=0.1,
    kalman_measurement_noise_maximum=2.0,
    kalman_initial_velocity_std=30.0,
    kalman_innovation_gate=4.0,
    kalman_missing_penalty=3.0,
    kalman_confidence_reward=2.0,
    require_generic_coco=False,
):
    try:
        import cv2
        from ultralytics import YOLO, settings
    except ImportError as error:
        raise RuntimeError(
            "video path selection requires OpenCV and Ultralytics"
        ) from error

    if len(source) != len(trajectory_xy) or len(matrices) != len(source):
        raise ValueError("video path inputs must have equal lengths")

    runs = []
    seen = set()
    for interval in intervals:
        boundary = interval.boundary_frame
        if not 0 <= boundary < len(source) or source[boundary] != 0:
            continue
        start = boundary
        while start > 0 and source[start - 1] == 0:
            start -= 1
        end = boundary + 1
        while end < len(source) and source[end] == 0:
            end += 1
        if (
            start == 0
            or end == len(source)
            or not np.isfinite(trajectory_xy[start - 1]).all()
            or not np.isfinite(trajectory_xy[end]).all()
            or (start, end) in seen
        ):
            continue
        if path_mode == "edge_anchored" and (
            source[start - 1] != 1 or source[end] != 1
        ):
            continue
        seen.add((start, end))
        runs.append((start, end, boundary))

    settings.update({"sync": False})
    model = YOLO(model_path)
    if class_id not in model.names:
        raise ValueError("{}: unknown detector class {}".format(model_path, class_id))
    if require_generic_coco and (
        str((model.ckpt or {}).get("train_args", {}).get("data")).split("/")[-1]
        != "coco.yaml"
        or model.names[class_id] != "sports ball"
    ):
        raise ValueError("input-only path audit requires a generic COCO sports-ball model")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("{}: cannot open video".format(video_path))

    reports = []
    try:
        for start, end, boundary in runs:
            candidates_by_frame = [[] for _ in range(end - start)]
            windows = [(start, end)]
            if path_mode == "edge_anchored":
                windows = []
                if end - start >= 8:
                    windows = [(start, min(start + 25, end))]
                    if end - start > 25:
                        windows.append(
                            (max(end - 25, start + 25), end)
                        )
            for first, last in windows:
                capture.set(cv2.CAP_PROP_POS_FRAMES, first)
                for frame in range(first, last):
                    success, image = capture.read()
                    if not success:
                        raise ValueError(
                            "{}: cannot decode frame {}".format(video_path, frame)
                        )
                    candidates_by_frame[frame - start] = (
                        detect_tiled_ball_candidates(
                            image,
                            matrices[frame],
                            model,
                            class_id=class_id,
                            confidence=confidence,
                            image_size=image_size,
                            device=device,
                            overlap_fraction=overlap_fraction,
                            merge_radius_pixels=merge_radius_pixels,
                            pitch_margin=pitch_margin,
                        )
                    )

            if path_mode == "temporal":
                path = select_temporal_ball_path(
                    candidates_by_frame,
                    (-1, trajectory_xy[start - 1]),
                    (end - start, trajectory_xy[end]),
                    fps=fps,
                    pitch=pitch,
                    max_gap_frames=maximum_gap_frames,
                    max_speed=maximum_speed,
                    max_acceleration=maximum_acceleration,
                    confidence_reward=confidence_reward,
                    gap_penalty=gap_penalty,
                    speed_weight=speed_weight,
                    acceleration_weight=acceleration_weight,
                )
                selected = np.isfinite(path.xy).all(axis=1)
                smoothed = selected
                position_std = np.full(len(selected), np.nan)
                path_confidence = path.confidence
            elif path_mode == "kalman_rts":
                path = select_kalman_ball_path(
                    candidates_by_frame,
                    (-1, trajectory_xy[start - 1]),
                    (end - start, trajectory_xy[end]),
                    fps=fps,
                    pitch=pitch,
                    beam_width=kalman_beam_width,
                    maximum_missing_frames=kalman_maximum_missing_frames,
                    process_acceleration_std=(
                        kalman_process_acceleration_std
                    ),
                    measurement_noise_base=kalman_measurement_noise_base,
                    measurement_noise_minimum=(
                        kalman_measurement_noise_minimum
                    ),
                    measurement_noise_maximum=(
                        kalman_measurement_noise_maximum
                    ),
                    initial_velocity_std=kalman_initial_velocity_std,
                    innovation_gate=kalman_innovation_gate,
                    missing_penalty=kalman_missing_penalty,
                    confidence_reward=kalman_confidence_reward,
                )
                smoothed = np.isfinite(path.xy).all(axis=1)
                selected = smoothed & (path.candidate_index >= 0)
                position_std = path.position_std
                path_confidence = path.observation_confidence
            elif path_mode == "edge_anchored":
                path = select_edge_anchored_ball_paths(
                    candidates_by_frame,
                    trajectory_xy[start - 1],
                    trajectory_xy[end],
                    fps=fps,
                    pitch=pitch,
                    beam_width=kalman_beam_width,
                    maximum_missing_frames=min(
                        kalman_maximum_missing_frames, 5
                    ),
                    process_acceleration_std=(
                        kalman_process_acceleration_std
                    ),
                    measurement_noise_base=kalman_measurement_noise_base,
                    measurement_noise_minimum=kalman_measurement_noise_minimum,
                    measurement_noise_maximum=kalman_measurement_noise_maximum,
                    initial_velocity_std=kalman_initial_velocity_std,
                    innovation_gate=kalman_innovation_gate,
                    missing_penalty=kalman_missing_penalty,
                    confidence_reward=kalman_confidence_reward,
                )
                smoothed = np.isfinite(path.xy).all(axis=1)
                selected = smoothed & (path.candidate_index >= 0)
                position_std = path.position_std
                path_confidence = path.observation_confidence
            else:
                raise ValueError(
                    "video path mode must be temporal, kalman_rts or edge_anchored"
                )
            present = np.asarray([
                bool(candidates) for candidates in candidates_by_frame
            ])
            present_with_anchors = np.r_[True, present, True]
            present_frames = np.flatnonzero(present_with_anchors)
            maximum_detector_gap = int(
                np.max(np.diff(present_frames) - 1)
            )
            selected_indexes = np.flatnonzero(selected)
            selected_points = [{
                "frame": int(start + index),
                "pitch_x": float(path.xy[index, 0]),
                "pitch_y": float(path.xy[index, 1]),
                "confidence": float(path_confidence[index]),
                "candidate_index": int(path.candidate_index[index]),
                "position_std": (
                    float(position_std[index])
                    if np.isfinite(position_std[index])
                    else None
                ),
            } for index in selected_indexes]
            candidate_frames = [{
                "frame": int(start + index),
                "candidates": [{
                    "pitch_x": float(
                        candidate["normalized_pitch"][0] * pitch[0]
                    ),
                    "pitch_y": float(
                        candidate["normalized_pitch"][1] * pitch[1]
                    ),
                    "confidence": float(candidate["confidence"]),
                } for candidate in candidates],
            } for index, candidates in enumerate(candidates_by_frame)
                if candidates]

            selected_with_anchors = np.r_[
                True, selected, True
            ]
            selected_frames = np.flatnonzero(selected_with_anchors)
            maximum_selected_gap = int(
                np.max(np.diff(selected_frames) - 1)
            )
            reports.append({
                "boundary_frame": int(boundary),
                "start_frame": int(start),
                "end_frame": int(end),
                "frames": int(end - start),
                "detector_frame_coverage": float(np.mean(present)),
                "candidates_per_frame": float(np.mean([
                    len(candidates)
                    for candidates in candidates_by_frame
                ])),
                "maximum_detector_gap": maximum_detector_gap,
                "left_detector_gap": int(present_frames[1] - 1),
                "right_detector_gap": int(
                    len(present) - present_frames[-2]
                ),
                "selected_frames": int(np.count_nonzero(selected)),
                "selected_frame_coverage": float(np.mean(selected)),
                "smoothed_frames": int(np.count_nonzero(smoothed)),
                "maximum_position_std": (
                    float(np.max(position_std[smoothed]))
                    if np.any(smoothed)
                    and np.isfinite(position_std[smoothed]).all()
                    else None
                ),
                "maximum_selected_gap": maximum_selected_gap,
                "path_mode": path_mode,
                "total_cost": (
                    float(sum(item["total_cost"] for item in path.tracklets))
                    if path_mode == "edge_anchored" and path.tracklets
                    else float(path.total_cost)
                    if path_mode != "edge_anchored" and np.isfinite(path.total_cost)
                    else None
                ),
                "edge_tracklets": (
                    path.tracklets if path_mode == "edge_anchored" else []
                ),
                "left_anchor": {
                    "frame": int(start - 1),
                    "pitch_x": float(trajectory_xy[start - 1, 0]),
                    "pitch_y": float(trajectory_xy[start - 1, 1]),
                },
                "right_anchor": {
                    "frame": int(end),
                    "pitch_x": float(trajectory_xy[end, 0]),
                    "pitch_y": float(trajectory_xy[end, 1]),
                },
                "points": selected_points,
                "candidate_frames": candidate_frames,
            })
    finally:
        capture.release()
    return reports


def detect_interval_coverage(
    video_path,
    model_path,
    intervals,
    matrices,
    *,
    class_id,
    confidence,
    image_size,
    device,
    overlap_fraction,
    merge_radius_pixels,
    pitch_margin,
):
    try:
        import cv2
        from ultralytics import YOLO, settings
    except ImportError as error:
        raise RuntimeError(
            "video detection requires OpenCV and Ultralytics"
        ) from error

    settings.update({"sync": False})
    model = YOLO(model_path)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("{}: cannot open video".format(video_path))
    reports = []
    try:
        for interval in intervals:
            capture.set(cv2.CAP_PROP_POS_FRAMES, interval.start_frame)
            counts = []
            for frame in range(interval.start_frame, interval.end_frame):
                success, image = capture.read()
                if not success:
                    raise ValueError(
                        "{}: cannot decode frame {}".format(video_path, frame)
                    )
                candidates = detect_tiled_ball_candidates(
                    image,
                    matrices[frame],
                    model,
                    class_id=class_id,
                    confidence=confidence,
                    image_size=image_size,
                    device=device,
                    overlap_fraction=overlap_fraction,
                    merge_radius_pixels=merge_radius_pixels,
                    pitch_margin=pitch_margin,
                )
                counts.append(len(candidates))
            present = np.asarray(counts) > 0
            longest_gap = 0
            current_gap = 0
            for detected in present:
                current_gap = 0 if detected else current_gap + 1
                longest_gap = max(longest_gap, current_gap)
            reports.append({
                "boundary_frame": interval.boundary_frame,
                "start_frame": interval.start_frame,
                "end_frame": interval.end_frame,
                "frames": interval.end_frame - interval.start_frame,
                "frame_coverage": float(np.mean(present)),
                "candidates_per_frame": float(np.mean(counts)),
                "maximum_no_detection_run": int(longest_gap),
            })
    finally:
        capture.release()
    return reports
def _contiguous_runs(mask):
    boundaries = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(mask)]
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if mask[start]
    ]


def _team_switches(labels):
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return changes[
        np.isin(labels[changes - 1], [HOME, AWAY])
        & np.isin(labels[changes], [HOME, AWAY])
    ]


def rank_counterfactual_boundary_influence(
    evidence,
    tags,
    labels,
    source,
    intervals,
    config,
    *,
    window_frames,
    search_radius,
    control_score,
    force_strong_control,
    maximum_boundaries,
):
    from possession.decoder import Evidence, decode

    if (
        len(labels) != len(source)
        or len(evidence.control) != len(labels)
        or len(evidence.dead) != len(labels)
    ):
        raise ValueError("counterfactual influence inputs must have equal lengths")
    windows = sorted({int(value) for value in window_frames})
    if (
        not windows
        or windows[0] < 1
        or search_radius < 1
        or control_score <= 0
        or maximum_boundaries < 1
    ):
        raise ValueError("invalid counterfactual influence configuration")

    switches = _team_switches(labels)
    interval_boundaries = {interval.boundary_frame for interval in intervals}
    candidates = [
        int(boundary) for boundary in switches
        if int(boundary) in interval_boundaries
    ]
    missing = np.asarray(source) == 0

    def missing_count(start, end):
        start = max(0, int(start))
        end = min(len(labels), int(end))
        return int(np.count_nonzero(missing[start:end]))

    candidates.sort(key=lambda boundary: (
        -missing_count(boundary - search_radius, boundary + search_radius),
        boundary,
    ))
    candidates = candidates[:maximum_boundaries]

    def counterfactual(boundary, team, start, end):
        control = evidence.control.copy()
        strong = evidence.strong_control.copy()
        control[start:end, team] = np.maximum(
            control[start:end, team], control_score
        )
        if force_strong_control:
            other = AWAY if team == HOME else HOME
            control[start:end, other] = np.minimum(
                control[start:end, other], 0.0
            )
            strong[start:end, team] = True
            strong[start:end, other] = False
        changed = decode(
            Evidence(
                control=control,
                dead=evidence.dead,
                strong_control=strong,
            ),
            tags,
            config,
        )
        from_team = int(labels[boundary - 1])
        to_team = int(labels[boundary])
        changed_switches = _team_switches(changed)
        matching = changed_switches[
            (changed[changed_switches - 1] == from_team)
            & (changed[changed_switches] == to_team)
            & (np.abs(changed_switches - boundary) <= search_radius)
        ]
        if matching.size == 0:
            return None, int(np.count_nonzero(
                changed[
                    max(0, boundary - search_radius):
                    min(len(labels), boundary + search_radius)
                ]
                != labels[
                    max(0, boundary - search_radius):
                    min(len(labels), boundary + search_radius)
                ]
            ))
        changed_boundary = int(
            matching[np.argmin(np.abs(matching - boundary))]
        )
        return changed_boundary - boundary, int(np.count_nonzero(
            changed[
                max(0, boundary - search_radius):
                min(len(labels), boundary + search_radius)
            ]
            != labels[
                max(0, boundary - search_radius):
                min(len(labels), boundary + search_radius)
            ]
        ))

    reports = []
    for boundary in candidates:
        from_team = int(labels[boundary - 1])
        to_team = int(labels[boundary])
        earlier = []
        later = []
        for window in windows:
            start = max(0, boundary - window)
            shift, changed_frames = counterfactual(
                boundary, to_team, start, boundary
            )
            earlier.append({
                "window_frames": window,
                "boundary_shift_frames": shift,
                "changed_label_frames": changed_frames,
            })
            end = min(len(labels), boundary + window)
            shift, changed_frames = counterfactual(
                boundary, from_team, boundary, end
            )
            later.append({
                "window_frames": window,
                "boundary_shift_frames": shift,
                "changed_label_frames": changed_frames,
            })

        earlier_windows = [
            item["window_frames"] for item in earlier
            if item["boundary_shift_frames"] is not None
            and item["boundary_shift_frames"] < 0
        ]
        later_windows = [
            item["window_frames"] for item in later
            if item["boundary_shift_frames"] is not None
            and item["boundary_shift_frames"] > 0
        ]
        pre_missing = missing_count(
            boundary - search_radius, boundary
        )
        post_missing = missing_count(
            boundary, boundary + search_radius
        )
        maximum_shift = max([
            abs(item["boundary_shift_frames"])
            for item in earlier + later
            if item["boundary_shift_frames"] is not None
        ] or [0])
        actionable_missing = (
            pre_missing * int(bool(earlier_windows))
            + post_missing * int(bool(later_windows))
        )
        priority = (
            actionable_missing / float(2 * search_radius)
            + min(1.0, maximum_shift / float(search_radius))
        )
        reports.append({
            "boundary_frame": boundary,
            "from_team": "H" if from_team == HOME else "A",
            "to_team": "H" if to_team == HOME else "A",
            "pre_boundary_missing_frames": pre_missing,
            "post_boundary_missing_frames": post_missing,
            "minimum_earlier_window_frames": (
                min(earlier_windows) if earlier_windows else None
            ),
            "minimum_later_window_frames": (
                min(later_windows) if later_windows else None
            ),
            "maximum_boundary_shift_frames": maximum_shift,
            "priority": float(priority),
            "earlier_counterfactuals": earlier,
            "later_counterfactuals": later,
        })
    reports.sort(key=lambda item: (
        -item["priority"], item["boundary_frame"]
    ))
    return reports


def rank_recovery_intervals(
    source,
    labels,
    homography_error,
    homography_valid,
    *,
    boundary_radius,
    minimum_missing_frames,
    minimum_valid_fraction,
    maximum_median_error,
    context_frames,
    maximum_intervals,
):
    n_frames = len(labels)
    if not (
        len(source)
        == len(homography_error)
        == len(homography_valid)
        == n_frames
    ):
        raise ValueError("video recovery inputs must have equal lengths")
    if boundary_radius < 1 or minimum_missing_frames < 1:
        raise ValueError(
            "video recovery radius and minimum missing frames must be positive"
        )
    if not 0.0 <= minimum_valid_fraction <= 1.0:
        raise ValueError(
            "minimum homography valid fraction must be between zero and one"
        )
    if maximum_median_error <= 0.0 or maximum_intervals < 1:
        raise ValueError(
            "maximum homography error and interval count must be positive"
        )

    switches = _team_switches(labels)
    if switches.size == 0:
        return []
    intervals = []
    for run_start, run_end in _contiguous_runs(source == 0):
        distances = np.maximum(
            np.maximum(run_start - switches, switches - (run_end - 1)),
            0,
        )
        nearest_index = int(np.argmin(distances))
        distance = int(distances[nearest_index])
        if distance > boundary_radius:
            continue
        boundary = int(switches[nearest_index])
        start = max(run_start, boundary - boundary_radius)
        end = min(run_end, boundary + boundary_radius + 1)
        missing_frames = end - start
        if missing_frames < minimum_missing_frames:
            continue

        interval_valid = homography_valid[start:end]
        valid_fraction = float(np.mean(interval_valid))
        if valid_fraction < minimum_valid_fraction:
            continue
        usable_error = homography_error[start:end][interval_valid]
        finite_error = usable_error[np.isfinite(usable_error)]
        if finite_error.size == 0:
            continue
        median_error = float(np.median(finite_error))
        if median_error > maximum_median_error:
            continue

        left_anchor = bool(
            run_start > 0 and source[run_start - 1] != 0
        )
        right_anchor = bool(
            run_end < n_frames and source[run_end] != 0
        )
        proximity = 1.0 - distance / float(boundary_radius + 1)
        coverage = min(1.0, missing_frames / float(boundary_radius))
        quality = 1.0 / (1.0 + median_error)
        anchors = 0.5 * (int(left_anchor) + int(right_anchor))
        priority = (
            0.4 * proximity
            + 0.25 * coverage
            + 0.2 * valid_fraction
            + 0.1 * quality
            + 0.05 * anchors
        )
        intervals.append(RecoveryInterval(
            start_frame=start,
            end_frame=end,
            context_start_frame=max(0, start - context_frames),
            context_end_frame=min(n_frames, end + context_frames),
            boundary_frame=boundary,
            from_team=int(labels[boundary - 1]),
            to_team=int(labels[boundary]),
            missing_frames=missing_frames,
            boundary_distance_frames=distance,
            homography_valid_fraction=valid_fraction,
            median_reprojection_error=median_error,
            left_anchor=left_anchor,
            right_anchor=right_anchor,
            priority=float(priority),
        ))
    intervals.sort(
        key=lambda item: (
            -item.priority,
            item.boundary_distance_frames,
            item.start_frame,
        )
    )
    return intervals[:maximum_intervals]
