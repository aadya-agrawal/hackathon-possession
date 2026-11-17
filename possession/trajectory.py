import json
from dataclasses import dataclass

import numpy as np


@dataclass
class BallTrajectory:
    xy: np.ndarray
    velocity: np.ndarray
    speed: np.ndarray
    confidence: np.ndarray
    source: np.ndarray


def load_video_ball_observations(path):
    with open(path) as handle:
        report = json.load(handle)
    observations = report.get("observations")
    if not isinstance(observations, list):
        raise ValueError("{}: missing observations list".format(path))
    frames = []
    xy = []
    confidence = []
    for observation in observations:
        try:
            frame = int(observation["frame"])
            position = [
                float(observation["pitch_x"]),
                float(observation["pitch_y"]),
            ]
            score = float(observation["confidence"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "{}: malformed video ball observation".format(path)
            ) from error
        frames.append(frame)
        xy.append(position)
        confidence.append(score)
    return (
        np.asarray(frames, dtype=np.int64),
        np.asarray(xy, dtype=np.float32).reshape(-1, 2),
        np.asarray(confidence, dtype=np.float32),
    )


def fuse_video_ball_observations(
    trajectory,
    frame_indexes,
    xy,
    confidence,
    *,
    fps,
    max_speed,
    replace_source_codes=(0,),
):
    frames = np.asarray(frame_indexes, dtype=np.int64)
    positions = np.asarray(xy, dtype=np.float32)
    scores = np.asarray(confidence, dtype=np.float32)
    if (
        positions.shape != (len(frames), 2)
        or scores.shape != (len(frames),)
    ):
        raise ValueError("video observation arrays have incompatible shapes")
    if (
        np.any(frames < 0)
        or np.any(frames >= len(trajectory.xy))
        or len(np.unique(frames)) != len(frames)
    ):
        raise ValueError("video observation frames must be unique and in range")
    if (
        not np.isfinite(positions).all()
        or not np.isfinite(scores).all()
        or np.any(scores <= 0)
        or np.any(scores > 1)
    ):
        raise ValueError("video observations must be finite with confidence in (0, 1]")
    order = np.argsort(frames)
    frames = frames[order]
    positions = positions[order]
    scores = scores[order]
    consecutive = np.diff(frames) == 1
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps
    if np.any(speeds[consecutive] > max_speed):
        raise ValueError("video observations exceed maximum ball speed")

    fused_xy = trajectory.xy.copy()
    fused_source = trajectory.source.copy()
    fused_confidence = trajectory.confidence.copy()
    replace = np.isin(fused_source[frames], replace_source_codes)
    fused_frames = frames[replace]
    fused_xy[fused_frames] = positions[replace]
    fused_source[fused_frames] = 4
    fused_confidence[fused_frames] = scores[replace]
    _repair_fused_joins(fused_xy, fused_source, fused_confidence, fps, max_speed)
    velocity = _compute_velocity(fused_xy, fps)
    speed = np.linalg.norm(velocity, axis=1)
    speed[~np.isfinite(velocity).all(axis=1)] = np.nan
    return BallTrajectory(
        xy=fused_xy,
        velocity=velocity,
        speed=speed.astype(np.float32),
        confidence=fused_confidence,
        source=fused_source,
    )


def _repair_fused_joins(xy, source, confidence, fps, max_speed):
    """Make every join between fused (source 4) runs and retained frames physically possible.

    A predicted extrapolation that contradicts an observation is dropped in favour of it; an
    observed or interpolated neighbour is kept and the fused run is trimmed from that edge.
    """
    def drop(frames):
        xy[frames] = np.nan
        source[frames] = 0
        confidence[frames] = 0.0

    def feasible(a, b):
        return np.linalg.norm(xy[a] - xy[b]) * fps <= max_speed * abs(b - a)

    n = len(source)
    fused = np.r_[False, source == 4, False]
    edges = np.flatnonzero(np.diff(fused.astype(np.int8)))
    for start, stop in zip(edges[::2], edges[1::2]):
        for side in (-1, 1):
            while start < stop:
                edge = start if side < 0 else stop - 1
                neighbour = edge + side
                if not 0 <= neighbour < n or not np.isfinite(xy[neighbour]).all() or feasible(edge, neighbour):
                    break
                if source[neighbour] == 3:
                    run = neighbour
                    while 0 <= run < n and source[run] == 3:
                        run += side
                    drop(np.arange(min(neighbour, run - side), max(neighbour, run - side) + 1))
                    break
                drop(edge)
                if side < 0:
                    start += 1
                else:
                    stop -= 1


def _isolated_jump_candidates(xy, max_step):
    """Frames whose (unmodified) neighbours make them isolated jumps, or None when the screen below
    cannot be trusted. Comparisons far from their thresholds are decided in float64; ones within a
    relative 1e-5 (far above float32 rounding) are re-evaluated with the reference expressions."""
    if (not isinstance(xy, np.ndarray) or xy.ndim != 2 or xy.dtype.kind != "f" or not np.isfinite(max_step)
            or np.any(np.abs(xy[np.isfinite(xy)]) > 1e15)):
        return None
    finite = np.isfinite(xy).all(axis=1)
    values = xy.astype(np.float64)
    frames = np.arange(1, len(xy) - 1)
    frames = frames[finite[frames - 1] & finite[frames] & finite[frames + 1]]
    incoming = np.sqrt(np.sum((values[frames] - values[frames - 1]) ** 2, axis=1))
    outgoing = np.sqrt(np.sum((values[frames + 1] - values[frames]) ** 2, axis=1))
    bypass = np.sqrt(np.sum((values[frames + 1] - values[frames - 1]) ** 2, axis=1))
    limits = ((incoming, max_step), (outgoing, max_step), (bypass, 2.0 * max_step))
    unsure = np.zeros(len(frames), dtype=bool)
    for distance, limit in limits:
        unsure |= np.abs(distance - limit) <= 1e-5 * abs(limit)
    jumps = (incoming > max_step) & (outgoing > max_step) & (bypass <= 2.0 * max_step)
    for index in np.flatnonzero(unsure):
        frame = frames[index]
        previous, current, following = xy[frame - 1], xy[frame], xy[frame + 1]
        jumps[index] = (
            np.linalg.norm(current - previous) > max_step
            and np.linalg.norm(following - current) > max_step
            and np.linalg.norm(following - previous) <= 2.0 * max_step
        )
    return frames[jumps]


def _reject_isolated_jumps(xy, fps, max_speed):
    cleaned = xy.copy()
    max_step = max_speed / float(fps)
    candidates = _isolated_jump_candidates(xy, max_step)
    if candidates is not None:
        # A rejected frame becomes its successor's missing previous position, so in a run of
        # consecutive candidates every other frame is rejected, starting with the first.
        rejected = -2
        for frame in candidates.tolist():
            if frame - 1 != rejected:
                cleaned[frame] = np.nan
                rejected = frame
        return cleaned
    for frame in range(1, len(cleaned) - 1):
        previous = cleaned[frame - 1]
        current = cleaned[frame]
        following = cleaned[frame + 1]
        if not (
            np.isfinite(previous).all()
            and np.isfinite(current).all()
            and np.isfinite(following).all()
        ):
            continue
        incoming = np.linalg.norm(current - previous)
        outgoing = np.linalg.norm(following - current)
        bypass = np.linalg.norm(following - previous)
        if incoming > max_step and outgoing > max_step and bypass <= 2.0 * max_step:
            cleaned[frame] = np.nan
    return cleaned


def _interpolate_short_gaps(xy, fps, max_gap_frames, max_speed):
    repaired = xy.copy()
    source = np.zeros(len(xy), dtype=np.int8)
    source[np.isfinite(xy).all(axis=1)] = 1
    valid = np.flatnonzero(source == 1)

    for left, right in zip(valid[:-1], valid[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        elapsed = (right - left) / float(fps)
        required_speed = np.linalg.norm(repaired[right] - repaired[left]) / elapsed
        if required_speed > max_speed:
            continue
        fractions = np.arange(1, gap + 1, dtype=np.float32) / float(gap + 1)
        repaired[left + 1:right] = (
            repaired[left][None, :]
            + fractions[:, None] * (repaired[right] - repaired[left])[None, :]
        )
        source[left + 1:right] = 2
    return repaired, source


def _compute_velocity(xy, fps):
    velocity = np.full_like(xy, np.nan, dtype=np.float32)
    valid = np.isfinite(xy).all(axis=1)
    if isinstance(xy, np.ndarray) and xy.ndim == 2:
        # Same per-frame expressions, vectorised: centred difference, else forward, else backward.
        n = len(xy)
        frames = np.arange(n)
        centred = np.zeros(n, dtype=bool)
        centred[1:-1] = valid[:-2] & valid[2:]
        forward = np.zeros(n, dtype=bool)
        forward[:-1] = ~centred[:-1] & valid[:-1] & valid[1:]
        backward = np.zeros(n, dtype=bool)
        backward[1:] = ~centred[1:] & ~forward[1:] & valid[:-1] & valid[1:]
        rows = frames[centred]
        velocity[rows] = (xy[rows + 1] - xy[rows - 1]) * (fps / 2.0)
        rows = frames[forward]
        velocity[rows] = (xy[rows + 1] - xy[rows]) * fps
        rows = frames[backward]
        velocity[rows] = (xy[rows] - xy[rows - 1]) * fps
        return velocity
    for frame in range(len(xy)):
        left = frame - 1
        right = frame + 1
        if left >= 0 and right < len(xy) and valid[left] and valid[right]:
            velocity[frame] = (xy[right] - xy[left]) * (fps / 2.0)
        elif right < len(xy) and valid[frame] and valid[right]:
            velocity[frame] = (xy[right] - xy[frame]) * fps
        elif left >= 0 and valid[left] and valid[frame]:
            velocity[frame] = (xy[frame] - xy[left]) * fps
    return velocity


def _quadratic_predictions(
    history_xy,
    fps,
    horizon,
    max_speed,
    max_acceleration,
    pitch,
    pitch_margin,
):
    history_length = len(history_xy)
    time = np.arange(-(history_length - 1), 1, dtype=np.float64)
    design = np.vander(time, N=3, increasing=True)
    coefficients = np.linalg.lstsq(design, history_xy, rcond=None)[0]

    acceleration = 2.0 * coefficients[2] * fps * fps
    acceleration_norm = np.linalg.norm(acceleration)
    if acceleration_norm > max_acceleration:
        coefficients[2] *= max_acceleration / acceleration_norm

    predictions = []
    previous = history_xy[-1]
    max_step = max_speed / float(fps)
    for step in range(1, horizon + 1):
        position = (
            coefficients[0]
            + coefficients[1] * step
            + coefficients[2] * step * step
        )
        if np.linalg.norm(position - previous) > max_step:
            break
        if pitch is not None:
            length, width = pitch
            if not (
                -pitch_margin <= position[0] <= length + pitch_margin
                and -pitch_margin <= position[1] <= width + pitch_margin
            ):
                break
        predictions.append(position.astype(np.float32))
        previous = position
    return predictions


def _predict_gap_edges(
    xy,
    source,
    fps,
    history_frames,
    horizon_frames,
    max_speed,
    max_acceleration,
    pitch,
    pitch_margin,
):
    if horizon_frames <= 0:
        return xy, source, np.zeros(len(xy), dtype=np.float32)

    forward = np.full_like(xy, np.nan, dtype=np.float32)
    backward = np.full_like(xy, np.nan, dtype=np.float32)
    missing = source == 0
    boundaries = np.flatnonzero(missing[1:] != missing[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(missing)]

    for start, end in zip(starts, ends):
        if not missing[start]:
            continue
        gap_length = end - start
        prediction_length = min(horizon_frames, gap_length)

        history_start = start - history_frames
        if history_start >= 0 and np.all(source[history_start:start] == 1):
            predictions = _quadratic_predictions(
                xy[history_start:start],
                fps,
                prediction_length,
                max_speed,
                max_acceleration,
                pitch,
                pitch_margin,
            )
            for offset, position in enumerate(predictions):
                forward[start + offset] = position

        history_end = end + history_frames
        if history_end <= len(xy) and np.all(source[end:history_end] == 1):
            predictions = _quadratic_predictions(
                xy[end:history_end][::-1],
                fps,
                prediction_length,
                max_speed,
                max_acceleration,
                pitch,
                pitch_margin,
            )
            for offset, position in enumerate(predictions):
                backward[end - offset - 1] = position

    predicted_xy = xy.copy()
    predicted_source = source.copy()
    confidence = np.zeros(len(xy), dtype=np.float32)
    for frame in np.flatnonzero(missing):
        has_forward = np.isfinite(forward[frame]).all()
        has_backward = np.isfinite(backward[frame]).all()
        if not has_forward and not has_backward:
            continue
        if has_forward and has_backward:
            left = frame
            while left > 0 and missing[left - 1]:
                left -= 1
            right = frame
            while right + 1 < len(xy) and missing[right + 1]:
                right += 1
            from_left = frame - left + 1
            from_right = right - frame + 1
            forward_weight = from_right / float(from_left + from_right)
            position = (
                forward_weight * forward[frame]
                + (1.0 - forward_weight) * backward[frame]
            )
            prediction_distance = min(from_left, from_right)
        elif has_forward:
            position = forward[frame]
            left = frame
            while left > 0 and missing[left - 1]:
                left -= 1
            prediction_distance = frame - left + 1
        else:
            position = backward[frame]
            right = frame
            while right + 1 < len(xy) and missing[right + 1]:
                right += 1
            prediction_distance = right - frame + 1

        predicted_xy[frame] = position
        predicted_source[frame] = 3
        decay = (prediction_distance - 1) / float(max(horizon_frames - 1, 1))
        confidence[frame] = 0.45 - 0.35 * min(decay, 1.0)
    return predicted_xy, predicted_source, confidence


def _prune_prediction_jumps(xy, source, confidence, fps, max_speed):
    pruned_xy = xy.copy()
    pruned_source = source.copy()
    pruned_confidence = confidence.copy()
    valid = np.isfinite(pruned_xy).all(axis=1)
    distances = np.linalg.norm(np.diff(pruned_xy, axis=0), axis=1)
    violations = (
        valid[:-1]
        & valid[1:]
        & (distances > max_speed / float(fps))
        & ((pruned_source[:-1] == 3) | (pruned_source[1:] == 3))
    )
    remove = np.zeros(len(pruned_xy), dtype=bool)
    indexes = np.flatnonzero(violations)
    remove[indexes] |= pruned_source[indexes] == 3
    remove[indexes + 1] |= pruned_source[indexes + 1] == 3
    pruned_xy[remove] = np.nan
    pruned_source[remove] = 0
    pruned_confidence[remove] = 0.0
    return pruned_xy, pruned_source, pruned_confidence


def repair_ball_trajectory(
    raw_xy,
    fps,
    max_gap_frames=20,
    max_speed=45.0,
    prediction_history_frames=7,
    prediction_horizon_frames=10,
    max_acceleration=20.0,
    pitch=None,
    pitch_margin=2.0,
):
    cleaned = _reject_isolated_jumps(raw_xy, fps, max_speed)
    repaired, source = _interpolate_short_gaps(
        cleaned, fps, max_gap_frames, max_speed
    )
    repaired, source, prediction_confidence = _predict_gap_edges(
        repaired,
        source,
        fps,
        prediction_history_frames,
        prediction_horizon_frames,
        max_speed,
        max_acceleration,
        pitch,
        pitch_margin,
    )
    repaired, source, prediction_confidence = _prune_prediction_jumps(
        repaired,
        source,
        prediction_confidence,
        fps,
        max_speed,
    )
    confidence = np.zeros(len(repaired), dtype=np.float32)
    confidence[source == 1] = 1.0
    confidence[source == 2] = 0.55
    confidence[source == 3] = prediction_confidence[source == 3]
    velocity = _compute_velocity(repaired, fps)
    speed = np.linalg.norm(velocity, axis=1)
    speed[~np.isfinite(velocity).all(axis=1)] = np.nan
    return BallTrajectory(
        xy=repaired,
        velocity=velocity,
        speed=speed.astype(np.float32),
        confidence=confidence,
        source=source,
    )
