import contextlib
import functools
import io
import multiprocessing
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from possession import AWAY, HOME


# Lowest/highest few x positions per team, enough to tolerate stray or mislabelled tracks.
TEAM_X_RANKS = 3

# Fast ATD parsing: only strictly well-formed files ("id:x,y,s;...;x,y,s:x,y,s" lines, plain decimal
# or NaN tokens) take the vectorised path; anything else falls back to the line-by-line parsers.
_ATD_ALLOWED = b"0123456789.,;:\nNa-+eE"
_ATD_SEPARATORS = np.zeros(256, dtype=bool)
_ATD_SEPARATORS[list(b",;:\n")] = True
_ATD_NON_DIGIT = np.ones(256, dtype=np.int64)
_ATD_NON_DIGIT[list(b"0123456789")] = 0
_ATD_PARALLEL_BYTES = 16 << 20
_ATD_CHUNK = {}


@dataclass(frozen=True)
class _AtdArrays:
    # Raw normalised ATD values for frames 1..len: players (frames, slots, [x, y, speed]), ball (frames, 3)
    players: np.ndarray
    ball: np.ndarray


def _parse_atd_chunk(chunk, n_slots):
    """Float64 rows (id, slot values..., ball values) for whole well-formed ATD lines, else None."""
    if not chunk or chunk.translate(None, _ATD_ALLOWED):
        return None
    data = np.frombuffer(chunk, dtype=np.uint8)
    separators = np.flatnonzero(_ATD_SEPARATORS[data])
    pattern = np.frombuffer(b":" + b",,;" * (n_slots - 1) + b",,:,,\n", dtype=np.uint8)
    width = len(pattern)
    if (not len(separators) or len(separators) % width or separators[0] == 0
            or separators[-1] != len(data) - 1 or not np.all(np.diff(separators) > 1)):
        return None
    if not np.array_equal(data[separators].reshape(-1, width), np.broadcast_to(pattern, (len(separators) // width, width))):
        return None
    # Frame ids must be plain digits, as int() accepts in the line parser.
    starts = np.r_[0, separators[width - 1::width][:-1] + 1]
    colons = separators[0::width]
    bounds = np.column_stack([starts, colons]).ravel()
    if np.add.reduceat(_ATD_NON_DIGIT[data], bounds)[0::2].any():
        return None
    try:
        import pandas as pd
        values = pd.read_csv(
            io.BytesIO(chunk.replace(b":", b",").replace(b";", b",")), header=None,
            names=range(width), dtype=np.float64, engine="c", float_precision="round_trip",
            na_values=["NaN"], keep_default_na=False,
        ).to_numpy(dtype=np.float64)
    except Exception:
        return None
    if values.shape != (len(separators) // width, width):
        return None
    return values


def _parse_atd_worker(index):
    start, stop = _ATD_CHUNK["bounds"][index]
    return _parse_atd_chunk(_ATD_CHUNK["data"][start:stop], _ATD_CHUNK["n_slots"])


@functools.lru_cache(maxsize=2)
def _cached_atd_arrays(path, size, mtime_ns, inode):
    with open(path, "rb") as handle:
        data = handle.read()
    if not data or b"\n" not in data:
        return None
    if not data.endswith(b"\n"):
        data += b"\n"
    n_slots = data[:data.index(b"\n")].count(b";") + 1
    chunks = 1 if len(data) < _ATD_PARALLEL_BYTES else min(16, os.cpu_count() or 1)
    bounds = [0]
    for index in range(1, chunks):
        cut = data.find(b"\n", max(bounds[-1], len(data) * index // chunks)) + 1
        if 0 < cut < len(data):
            bounds.append(cut)
    bounds.append(len(data))
    bounds = list(zip(bounds[:-1], bounds[1:]))
    parts = None
    if len(bounds) > 1:
        _ATD_CHUNK.update(data=data, bounds=bounds, n_slots=n_slots)
        try:
            # Forked workers inherit the bytes; chunks are independent and returned in file order.
            with ProcessPoolExecutor(len(bounds), mp_context=multiprocessing.get_context("fork")) as pool:
                parts = list(pool.map(_parse_atd_worker, range(len(bounds))))
        except Exception:
            parts = None
        finally:
            _ATD_CHUNK.clear()
    if parts is None:
        parts = [_parse_atd_chunk(data[start:stop], n_slots) for start, stop in bounds]
    if any(part is None for part in parts):
        return None
    values = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if not np.array_equal(values[:, 0], np.arange(1, len(values) + 1)):
        return None
    players = np.ascontiguousarray(values[:, 1:1 + 3 * n_slots].reshape(len(values), n_slots, 3))
    ball = np.ascontiguousarray(values[:, 1 + 3 * n_slots:])
    players.flags.writeable = False
    ball.flags.writeable = False
    return _AtdArrays(players=players, ball=ball)


def load_atd_arrays(raw_path):
    """Parsed ATD values shared by all readers of one file (cached per path/size/mtime), or None
    when the file is not strictly well-formed with frame ids 1..n."""
    try:
        stat = os.stat(raw_path)
        path = os.path.realpath(raw_path)
    except (OSError, TypeError, ValueError):
        return None
    return _cached_atd_arrays(path, stat.st_size, stat.st_mtime_ns, stat.st_ino)


def atd_positions(values, pitch):
    """Metres (frames, items, 2) from raw [x, y, ...] values; NaN where x or y is not finite."""
    length, width = pitch
    present = np.isfinite(values[..., 0]) & np.isfinite(values[..., 1])
    positions = np.full(values.shape[:-1] + (2,), np.nan)
    positions[..., 0] = np.where(present, values[..., 0] * length, np.nan)
    positions[..., 1] = np.where(present, values[..., 1] * width, np.nan)
    return positions, present


@dataclass
class TrackingMetadata:
    team_of_track: np.ndarray
    team_names: tuple
    goalkeeper_slots: tuple = ()


@dataclass
class ControlFeatures:
    nearest_distances: np.ndarray
    nearby_counts: np.ndarray
    nearest_player_xy: np.ndarray
    nearest_player_speed: np.ndarray
    nearest_player_velocity: np.ndarray
    player_counts: np.ndarray
    median_player_speed: np.ndarray
    stationary_player_fraction: np.ndarray
    nearest_player_slot: np.ndarray = None
    nearest_player_age_frames: np.ndarray = None
    team_x_extremes: np.ndarray = None


def _update_track_ages(positions, frame, last_seen, last_age, last_xy,
                       fps, max_gap_frames, max_speed):
    observed = np.isfinite(positions).all(axis=1)
    elapsed = frame - last_seen
    continuous = observed & (elapsed == 1)
    bridged = (
        observed & (last_seen >= 0) & (elapsed > 1)
        & (elapsed <= max_gap_frames + 1)
        & np.isfinite(last_xy).all(axis=1)
    )
    plausible = np.zeros(len(observed), dtype=bool)
    candidates = np.flatnonzero(bridged)
    if len(candidates):
        speeds = (
            np.linalg.norm(positions[candidates] - last_xy[candidates], axis=1)
            * fps / elapsed[candidates]
        )
        plausible[candidates] = speeds <= max_speed
    age = np.where(observed, np.where(continuous | plausible, last_age + 1, 1), 0)
    last_seen[observed] = frame
    last_age[observed] = age[observed]
    last_xy[observed] = positions[observed]
    return age


def load_tracking_metadata(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected two teams, found {}".format(meta_path, len(teams)))

    team_ids = [team.get("id") for team in teams]
    if not all(team_ids) or len(set(team_ids)) != 2:
        raise ValueError("{}: team IDs must be present and distinct".format(meta_path))
    players = list(root.iter("Player"))
    if any(player.get("teamId") not in team_ids for player in players):
        raise ValueError("{}: player references an unknown team ID".format(meta_path))
    home_id = team_ids[0]
    team_of_track = np.asarray(
        [HOME if player.get("teamId") == home_id else AWAY for player in players],
        dtype=np.int8,
    )
    if team_of_track.size == 0:
        raise ValueError("{}: no player track slots found".format(meta_path))

    names = tuple((team.findtext("Name") or "").strip() for team in teams)
    if not all(names):
        raise ValueError("{}: both teams must have names".format(meta_path))
    goalkeepers = tuple(
        slot for slot, player in enumerate(players)
        if any((parameter.findtext("Name") or "") == "position_type"
               and (parameter.findtext("Value") or "").strip().lower() == "goalkeeper"
               for parameter in player.iter("ProviderParameter"))
    )
    return TrackingMetadata(team_of_track=team_of_track, team_names=names, goalkeeper_slots=goalkeepers)


def goalkeeper_frame_teams(players_xy, metadata, home_attacks_high, pitch, radius):
    """Per-frame team of every track: goalkeeper tracks at the other team's goal count for that team.

    Goalkeepers do not stand at the opponent's goal, so a goalkeeper track there follows the other
    team's keeper (a tracking identity swap). home_attacks_high is +1/-1 per frame (0 = unknown).
    """
    n_frames, n_tracks = players_xy.shape[:2]
    if len(metadata.team_of_track) != n_tracks or home_attacks_high.shape != (n_frames,):
        raise ValueError("goalkeeper team correction needs aligned players and attack directions")
    teams = np.tile(metadata.team_of_track.astype(np.int8), (n_frames, 1))
    length = pitch[0]
    home_goal_x = np.where(home_attacks_high > 0, 0.0, length)
    known = home_attacks_high != 0
    for slot in metadata.goalkeeper_slots:
        team = int(metadata.team_of_track[slot])
        own_goal_x = home_goal_x if team == HOME else length - home_goal_x
        other_goal_x = length - own_goal_x
        x = players_xy[:, slot, 0]
        swapped = known & np.isfinite(x) & (np.abs(x - other_goal_x) < radius)
        teams[swapped, slot] = 1 - team
    return teams


def _parse_group(group):
    if not group or group[0].upper() == "N":
        return None
    values = group.split(",")
    if len(values) < 2:
        raise ValueError("malformed tracking group {!r}".format(group))
    x = float(values[0])
    y = float(values[1])
    speed = float(values[2]) if len(values) > 2 else np.nan
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y, speed


def load_raw_ball(raw_path, n_frames, pitch):
    atd = load_atd_arrays(raw_path)
    if atd is not None and len(atd.ball) == n_frames:
        ball, present = atd_positions(atd.ball, pitch)
        ball_xy = ball.astype(np.float32)
        ball_speed = np.where(present, atd.ball[:, 2], np.nan).astype(np.float32)
        return ball_xy, ball_speed
    return _load_raw_ball_lines(raw_path, n_frames, pitch)


def _load_raw_ball_lines(raw_path, n_frames, pitch):
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    expected_frame_id = 1
    length, width = pitch

    with open(raw_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split(":", 2)
            if len(parts) != 3:
                raise ValueError("{}:{}: expected three ATD fields".format(raw_path, line_number))
            frame_id = int(parts[0])
            if frame_id != expected_frame_id:
                raise ValueError(
                    "{}:{}: expected frame {}, found {}".format(
                        raw_path, line_number, expected_frame_id, frame_id
                    )
                )
            if frame_id > n_frames:
                raise ValueError("{}: contains more than {} frames".format(raw_path, n_frames))

            ball = _parse_group(parts[2])
            if ball is not None:
                ball_xy[frame_id - 1] = (ball[0] * length, ball[1] * width)
                ball_speed[frame_id - 1] = ball[2]
            expected_frame_id += 1

    if expected_frame_id - 1 != n_frames:
        raise ValueError(
            "{}: expected {} frames, found {}".format(raw_path, n_frames, expected_frame_id - 1)
        )
    return ball_xy, ball_speed


def load_player_positions_at_frames(
    raw_path,
    metadata,
    frame_indexes,
    pitch,
):
    requested = {int(frame) for frame in frame_indexes}
    if any(frame < 0 for frame in requested):
        raise ValueError("player position frame indexes must be non-negative")
    positions_by_frame = {}
    length, width = pitch
    n_tracks = len(metadata.team_of_track)

    with open(raw_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            frame = line_number - 1
            if frame not in requested:
                continue
            parts = line.rstrip("\n").split(":", 2)
            if len(parts) != 3 or int(parts[0]) - 1 != frame:
                raise ValueError(
                    "{}:{}: malformed or misaligned ATD frame".format(
                        raw_path, line_number
                    )
                )
            groups = parts[1].split(";")
            if len(groups) != n_tracks:
                raise ValueError(
                    "{}:{}: expected {} track slots, found {}".format(
                        raw_path, line_number, n_tracks, len(groups)
                    )
                )
            team_positions = ([], [])
            for slot, group in enumerate(groups):
                player = _parse_group(group)
                if player is None:
                    continue
                team = int(metadata.team_of_track[slot])
                team_positions[team].append((
                    player[0] * length,
                    player[1] * width,
                ))
            positions_by_frame[frame] = tuple(
                np.asarray(values, dtype=np.float32).reshape(-1, 2)
                for values in team_positions
            )
            if len(positions_by_frame) == len(requested):
                break
    missing = sorted(requested - set(positions_by_frame))
    if missing:
        raise ValueError(
            "{}: missing requested player frames {}".format(
                raw_path, missing[:10]
            )
        )
    return positions_by_frame


def _fast_control_inputs(atd, team_of_track, ball_xy, frame_teams, n_frames):
    """Per-frame track teams (frames, tracks) when the vectorised path reproduces the line parser."""
    team_of_track = np.asarray(team_of_track)
    if (atd is None or team_of_track.ndim != 1 or atd.players.shape[1] != len(team_of_track)
            or len(atd.players) > n_frames or not isinstance(ball_xy, np.ndarray)
            or ball_xy.ndim != 2 or ball_xy.shape[1] != 2 or ball_xy.dtype.kind != "f"
            or team_of_track.dtype.kind not in "biu" or not np.isin(team_of_track, (HOME, AWAY)).all()):
        return None
    n_rows, n_tracks = atd.players.shape[:2]
    if frame_teams is None:
        return np.broadcast_to(team_of_track.astype(np.int64), (n_rows, n_tracks))
    if (not isinstance(frame_teams, np.ndarray) or frame_teams.ndim != 2 or frame_teams.dtype.kind not in "biu"
            or frame_teams.shape[0] < n_rows or frame_teams.shape[1] < n_tracks):
        return None
    teams = frame_teams[:n_rows, :n_tracks].astype(np.int64)
    return teams if np.isin(teams, (HOME, AWAY)).all() else None


def _fill_control_features(atd, team_of_track, team_slots, ball_xy, pitch, teams, distance_thresholds,
                           nearest_count, stationary_speed_threshold, max_player_speed, fps,
                           resume_gap_frames, outputs):
    # Vectorised equivalent of the per-frame loop in extract_control_features (frames 1..n of the file).
    (nearest_distances, nearby_counts, nearest_player_xy, nearest_player_speed, nearest_player_velocity,
     nearest_player_slot, nearest_player_age, player_counts, team_x_extremes, median_player_speed,
     stationary_player_fraction) = outputs
    n_rows, n_tracks = atd.players.shape[:2]
    if n_rows == 0:
        return
    positions, present = atd_positions(atd.players, pitch)
    speed = atd.players[:, :, 2]
    rows = np.arange(n_rows)
    tracks = np.arange(n_tracks)

    # Own-track velocity between consecutive frames.
    velocity = np.full_like(positions, np.nan)
    velocity[1:] = player_velocity_steps(positions[:-1], positions[1:], fps, max_player_speed)

    # Track ages (_update_track_ages): runs of observations, optionally bridged by plausible short gaps.
    last_seen = np.maximum.accumulate(np.where(present, rows[:, None], -1), axis=0)
    previous = np.vstack([np.full((1, n_tracks), -1, dtype=last_seen.dtype), last_seen[:-1]])
    elapsed = rows[:, None] - previous
    linked = present & (previous >= 0) & (elapsed == 1)
    bridged_rows, bridged_tracks = np.nonzero(
        present & (previous >= 0) & (elapsed > 1) & (elapsed <= resume_gap_frames + 1))
    if len(bridged_rows):
        bridge_speed = (
            np.linalg.norm(positions[bridged_rows, bridged_tracks]
                           - positions[previous[bridged_rows, bridged_tracks], bridged_tracks], axis=1)
            * fps / elapsed[bridged_rows, bridged_tracks]
        )
        plausible = bridge_speed <= max_player_speed
        linked[bridged_rows[plausible], bridged_tracks[plausible]] = True
    observations = np.cumsum(present, axis=0)
    run_start = np.maximum.accumulate(np.where(present & ~linked, observations, 0), axis=0)
    age = np.where(present, observations - run_start + 1, 0)

    # Median speed and stationary share over finite speeds of tracked players.
    finite_speed = present & np.isfinite(speed)
    speed_count = np.count_nonzero(finite_speed, axis=1)
    speed32 = np.where(finite_speed, speed, np.nan).astype(np.float32)
    has_speed = np.flatnonzero(speed_count > 0)
    if len(has_speed):
        ordered = np.sort(speed32[has_speed], axis=1)
        count = speed_count[has_speed]
        low = ordered[np.arange(len(count)), (count - 1) // 2]
        high = ordered[np.arange(len(count)), count // 2]
        median_player_speed[has_speed] = np.where(count % 2 == 1, low, (low + high) / np.float32(2))
        below = np.count_nonzero(finite_speed[has_speed] & (speed32[has_speed] < stationary_speed_threshold), axis=1)
        stationary_player_fraction[has_speed] = below / count
        # Signed zeros make the partition order observable; recompute those frames exactly.
        signed_zero = np.flatnonzero((finite_speed & (speed == 0) & np.signbit(speed)).any(axis=1))
        for row in signed_zero:
            values = [speed[row, slot] for team in (HOME, AWAY)
                      for slot in np.flatnonzero(finite_speed[row] & (teams[row] == team))]
            median_player_speed[row] = np.median(np.asarray(values, dtype=np.float32))

    slot_ids = np.where(teams != team_of_track, n_tracks + tracks, team_slots)
    ball = ball_xy[:n_rows]
    ball_rows = np.isfinite(ball).all(axis=1)
    team_xy = positions.astype(np.float32)
    with np.errstate(invalid="ignore"):
        distances = np.linalg.norm(team_xy - ball[:, None, :], axis=2)
    far = distances.dtype.type(np.inf)
    take = max(0, min(nearest_count, n_tracks))
    x = positions[:, :, 0]
    for team in (HOME, AWAY):
        member = present & (teams == team)
        count = np.count_nonzero(member, axis=1)
        player_counts[:n_rows, team] = count
        # Stable sort keeps slot order among equal x, like sorted() on the slot-ordered list.
        ordered_x = np.sort(np.where(member, x, np.inf), axis=1, kind="stable")
        for rank in range(min(TEAM_X_RANKS, n_tracks)):
            ranked = np.flatnonzero(count > rank)
            team_x_extremes[ranked, team, 0, rank] = ordered_x[ranked, rank]
            team_x_extremes[ranked, team, 1, rank] = ordered_x[ranked, count[ranked] - 1 - rank]

        active = np.flatnonzero(ball_rows & (count > 0))
        if not len(active):
            continue
        team_distances = np.where(member[active], distances[active], far)
        if take:
            nearest_distances[active, team, :take] = np.sort(team_distances, axis=1)[:, :take]
        best = np.argmin(team_distances, axis=1)
        minimum = team_distances[np.arange(len(active)), best]
        # Ties (or non-finite minima) follow np.argsort on the slot-ordered team list exactly.
        ambiguous = np.flatnonzero(
            (np.count_nonzero(team_distances == minimum[:, None], axis=1) > 1) | ~np.isfinite(minimum))
        for index in ambiguous:
            row = active[index]
            slots = np.flatnonzero(member[row])
            order = np.argsort(np.linalg.norm(team_xy[row, slots] - ball_xy[row], axis=1))
            best[index] = slots[order[0]]
        nearest_player_xy[active, team] = team_xy[active, best]
        nearest_player_speed[active, team] = speed[active, best]
        nearest_player_velocity[active, team] = velocity[active, best]
        nearest_player_slot[active, team] = slot_ids[active, best]
        nearest_player_age[active, team] = age[active, best]
        for threshold_index, threshold in enumerate(distance_thresholds):
            nearby_counts[active, team, threshold_index] = np.count_nonzero(
                member[active] & (team_distances <= threshold), axis=1)


def extract_control_features(
    raw_path,
    metadata,
    ball_xy,
    pitch,
    distance_thresholds=(1.0, 2.0, 4.0),
    nearest_count=3,
    stationary_speed_threshold=1.0,
    max_player_speed=12.0,
    fps=25.0,
    player_reliability_resume_gap_frames=0,
    frame_teams=None,
):
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(max_player_speed) or max_player_speed <= 0:
        raise ValueError("control features require positive fps and player-speed limit")
    if (not isinstance(player_reliability_resume_gap_frames, int)
            or isinstance(player_reliability_resume_gap_frames, bool)
            or player_reliability_resume_gap_frames < 0):
        raise ValueError("player reliability resume gap must be a nonnegative integer")
    n_frames = len(ball_xy)
    nearest_distances = np.full(
        (n_frames, 2, nearest_count), np.inf, dtype=np.float32
    )
    nearby_counts = np.zeros(
        (n_frames, 2, len(distance_thresholds)), dtype=np.int16
    )
    nearest_player_xy = np.full((n_frames, 2, 2), np.nan, dtype=np.float32)
    nearest_player_speed = np.full((n_frames, 2), np.nan, dtype=np.float32)
    nearest_player_velocity = np.full((n_frames, 2, 2), np.nan, dtype=np.float32)
    nearest_player_slot = np.full((n_frames, 2), -1, dtype=np.int16)
    nearest_player_age = np.zeros((n_frames, 2), dtype=np.int32)
    player_counts = np.zeros((n_frames, 2), dtype=np.int16)
    team_x_extremes = np.full((n_frames, 2, 2, TEAM_X_RANKS), np.nan, dtype=np.float32)
    median_player_speed = np.full(n_frames, np.nan, dtype=np.float32)
    stationary_player_fraction = np.full(n_frames, np.nan, dtype=np.float32)

    length, width = pitch
    n_tracks = len(metadata.team_of_track)
    team_slots = np.empty(n_tracks, dtype=np.int16)
    for team in (HOME, AWAY):
        matching = np.flatnonzero(metadata.team_of_track == team)
        team_slots[matching] = np.arange(len(matching), dtype=np.int16)
    previous_positions = np.full((n_tracks, 2), np.nan, dtype=np.float64)
    last_seen = np.full(n_tracks, -1, dtype=np.int64)
    last_age = np.zeros(n_tracks, dtype=np.int32)
    last_xy = np.full((n_tracks, 2), np.nan, dtype=np.float64)
    previous_frame = None

    atd = load_atd_arrays(raw_path)
    fast = _fast_control_inputs(atd, metadata.team_of_track, ball_xy, frame_teams, n_frames)
    if fast is not None:
        _fill_control_features(
            atd, metadata.team_of_track, team_slots, ball_xy, pitch, fast, distance_thresholds,
            nearest_count, stationary_speed_threshold, max_player_speed, fps,
            player_reliability_resume_gap_frames,
            (nearest_distances, nearby_counts, nearest_player_xy, nearest_player_speed,
             nearest_player_velocity, nearest_player_slot, nearest_player_age, player_counts,
             team_x_extremes, median_player_speed, stationary_player_fraction),
        )

    # Line-by-line reference path, used when the file is not eligible for the vectorised path.
    with (contextlib.nullcontext(()) if fast is not None else open(raw_path)) as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split(":", 2)
            if len(parts) != 3:
                raise ValueError("{}:{}: expected three ATD fields".format(raw_path, line_number))
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n_frames:
                raise ValueError("{}:{}: frame is out of range".format(raw_path, line_number))
            if previous_frame is not None and frame <= previous_frame:
                raise ValueError("{}:{}: tracking frames must increase".format(raw_path, line_number))

            groups = parts[1].split(";")
            if len(groups) != n_tracks:
                raise ValueError(
                    "{}:{}: expected {} track slots, found {}".format(
                        raw_path, line_number, n_tracks, len(groups)
                    )
                )

            positions = ([], [])
            speeds = ([], [])
            slots = ([], [])
            current_positions = np.full_like(previous_positions, np.nan)
            for slot, group in enumerate(groups):
                player = _parse_group(group)
                if player is None:
                    continue
                team = int(metadata.team_of_track[slot])
                team_slot = team_slots[slot]
                if frame_teams is not None and frame_teams[frame, slot] != team:
                    # Reassigned track: a slot id outside both teams keeps identities distinct.
                    team = int(frame_teams[frame, slot])
                    team_slot = n_tracks + slot
                positions[team].append((player[0] * length, player[1] * width))
                speeds[team].append(player[2])
                slots[team].append((slot, team_slot))
                current_positions[slot] = (player[0] * length, player[1] * width)
            current_velocity = np.full_like(current_positions, np.nan)
            if previous_frame is not None and frame == previous_frame + 1:
                current_velocity = player_velocity_steps(previous_positions, current_positions, fps, max_player_speed)
            track_age = _update_track_ages(
                current_positions, frame, last_seen, last_age, last_xy,
                fps, player_reliability_resume_gap_frames, max_player_speed,
            )
            previous_positions = current_positions
            previous_frame = frame

            finite_speeds = np.asarray(
                [speed for team_speeds in speeds for speed in team_speeds
                 if np.isfinite(speed)],
                dtype=np.float32,
            )
            if finite_speeds.size:
                median_player_speed[frame] = np.median(finite_speeds)
                stationary_player_fraction[frame] = np.mean(
                    finite_speeds < stationary_speed_threshold
                )

            for team in (HOME, AWAY):
                player_counts[frame, team] = len(positions[team])
                if positions[team]:
                    team_x = sorted(xy[0] for xy in positions[team])
                    take = min(TEAM_X_RANKS, len(team_x))
                    team_x_extremes[frame, team, 0, :take] = team_x[:take]
                    team_x_extremes[frame, team, 1, :take] = team_x[::-1][:take]

            if not np.isfinite(ball_xy[frame]).all():
                continue

            for team in (HOME, AWAY):
                if not positions[team]:
                    continue

                team_xy = np.asarray(positions[team], dtype=np.float32)
                distances = np.linalg.norm(team_xy - ball_xy[frame], axis=1)
                order = np.argsort(distances)
                take = min(nearest_count, len(order))
                nearest_distances[frame, team, :take] = distances[order[:take]]
                nearest_player_xy[frame, team] = team_xy[order[0]]
                nearest_player_speed[frame, team] = speeds[team][order[0]]
                track_slot, team_slot = slots[team][order[0]]
                nearest_player_velocity[frame, team] = current_velocity[track_slot]
                nearest_player_slot[frame, team] = team_slot
                nearest_player_age[frame, team] = track_age[track_slot]
                for threshold_index, threshold in enumerate(distance_thresholds):
                    nearby_counts[frame, team, threshold_index] = np.count_nonzero(
                        distances <= threshold
                    )

    return ControlFeatures(
        nearest_distances=nearest_distances,
        nearby_counts=nearby_counts,
        nearest_player_xy=nearest_player_xy,
        nearest_player_speed=nearest_player_speed,
        nearest_player_velocity=nearest_player_velocity,
        player_counts=player_counts,
        median_player_speed=median_player_speed,
        stationary_player_fraction=stationary_player_fraction,
        nearest_player_slot=nearest_player_slot,
        nearest_player_age_frames=nearest_player_age,
        team_x_extremes=team_x_extremes,
    )


def _nearest_player_velocity(nearest_player_xy, fps, max_player_speed):
    # Legacy diagnostic proxy; inference follows the current player's own track.
    velocity = np.full_like(nearest_player_xy, np.nan)
    player_steps = (nearest_player_xy[1:] - nearest_player_xy[:-1]) * fps
    valid_steps = (
        np.isfinite(player_steps).all(axis=2)
        & (np.linalg.norm(player_steps, axis=2) <= max_player_speed)
    )
    velocity[1:][valid_steps] = player_steps[valid_steps]
    return velocity


def player_velocity_steps(previous, current, fps, max_speed):
    if (previous.shape != current.shape or current.ndim < 1 or current.shape[-1] != 2
            or not np.isfinite(fps) or fps <= 0 or not np.isfinite(max_speed) or max_speed <= 0):
        raise ValueError("player velocity requires aligned tracks and positive fps/speed")
    steps = (current - previous) * fps
    valid = np.isfinite(steps).all(axis=-1) & (np.linalg.norm(steps, axis=-1) <= max_speed)
    return np.where(valid[..., None], steps, np.nan)


def contact_motion_features(ball_xy, source, nearest, fps, radius, horizon_frames, motion_frames, max_speed):
    n = len(source)
    if (n < 1 or source.shape != (n,) or ball_xy.shape != (n, 2) or nearest.shape != (n, 2)
            or not np.isfinite(fps) or fps <= 0 or not np.isfinite(radius) or radius <= 0
            or not np.isfinite(max_speed) or max_speed <= 0):
        raise ValueError("contact motion requires aligned tracking and positive fps/radius/speed")
    if (not isinstance(horizon_frames, int) or not isinstance(motion_frames, int)
            or isinstance(horizon_frames, bool) or isinstance(motion_frames, bool)
            or horizon_frames < 1 or motion_frames < 1):
        raise ValueError("contact motion requires positive integer frame windows")
    rows = np.arange(n)
    observed = source == 1
    closest_distance = np.full((n, 2), np.inf, dtype=np.float32)
    closest_rows = np.repeat(rows[:, None], 2, axis=1)
    for lag in range(min(horizon_frames, n)):
        stop = n - lag
        nearer = observed[lag:, None] & (nearest[lag:] < closest_distance[:stop])
        closest_distance[:stop][nearer] = nearest[lag:][nearer]
        candidate_rows = np.repeat((rows[:stop] + lag)[:, None], 2, axis=1)
        closest_rows[:stop][nearer] = candidate_rows[nearer]
    before_rows = np.maximum(0, closest_rows - motion_frames)
    after_rows = np.minimum(n - 1, closest_rows + motion_frames)
    cumulative = np.r_[0, np.cumsum(observed, dtype=np.int32)]
    continuous = (
        (closest_rows >= motion_frames) & (closest_rows + motion_frames < n)
        & (cumulative[after_rows + 1] - cumulative[before_rows] == 2 * motion_frames + 1)
        & (closest_distance <= radius) & observed[:, None]
    )
    velocity_before = (ball_xy[closest_rows] - ball_xy[before_rows]) * fps / motion_frames
    velocity_after = (ball_xy[after_rows] - ball_xy[closest_rows]) * fps / motion_frames
    speed_before = np.linalg.norm(velocity_before, axis=2)
    speed_after = np.linalg.norm(velocity_after, axis=2)
    total = speed_before + speed_after
    valid = (
        continuous & np.isfinite(velocity_before).all(axis=2) & np.isfinite(velocity_after).all(axis=2)
        & (total > 0) & (np.maximum(speed_before, speed_after) <= max_speed)
    )
    disturbance = np.full((n, 2), np.nan, dtype=np.float32)
    np.divide(np.linalg.norm(velocity_after - velocity_before, axis=2), total,
              out=disturbance, where=valid)
    return {
        "contact_motion_disturbance": np.clip(disturbance, 0, 1),
        "contact_offset_seconds": np.where(continuous, (closest_rows - rows[:, None]) / fps, np.nan),
    }


def control_features_from_arrays(home, away, ball_xy, fps, config):
    n_frames = len(ball_xy)
    nearest_count = int(config["nearest_player_count"])
    thresholds = config["distance_thresholds_metres"]
    nearest = np.full((n_frames, 2, nearest_count), np.inf, dtype=np.float32)
    counts = np.zeros((n_frames, 2, len(thresholds)), dtype=np.int16)
    nearest_xy = np.full((n_frames, 2, 2), np.nan, dtype=np.float32)
    nearest_speed = np.full((n_frames, 2), np.nan, dtype=np.float32)
    nearest_velocity = np.full((n_frames, 2, 2), np.nan, dtype=np.float32)
    nearest_slot = np.full((n_frames, 2), -1, dtype=np.int16)
    nearest_age = np.zeros((n_frames, 2), dtype=np.int32)
    player_counts = np.zeros((n_frames, 2), dtype=np.int16)
    team_x_extremes = np.full((n_frames, 2, 2, TEAM_X_RANKS), np.nan, dtype=np.float32)
    all_speeds = []
    resume_gap = config.get("player_reliability_resume_gap_frames", 0)
    if (not isinstance(resume_gap, int) or isinstance(resume_gap, bool) or resume_gap < 0):
        raise ValueError("player reliability resume gap must be a nonnegative integer")
    for team, players in enumerate((home, away)):
        if players.ndim != 3 or players.shape[0] != n_frames or players.shape[2] != 2:
            raise ValueError("player arrays must have shape (frames, tracks, 2)")
        if players.shape[1] == 0:
            raise ValueError("each team must have at least one track slot")
        valid = np.isfinite(players).all(axis=2)
        player_counts[:, team] = np.count_nonzero(valid, axis=1)
        take = min(TEAM_X_RANKS, players.shape[1])
        ascending = np.sort(np.where(valid, players[:, :, 0], np.inf), axis=1)[:, :take]
        descending = -np.sort(np.where(valid, -players[:, :, 0], np.inf), axis=1)[:, :take]
        team_x_extremes[:, team, 0, :take] = np.where(np.isfinite(ascending), ascending, np.nan)
        team_x_extremes[:, team, 1, :take] = np.where(np.isfinite(descending), descending, np.nan)
        speed = np.full(valid.shape, np.nan, dtype=np.float32)
        speed[1:] = np.linalg.norm(np.diff(players, axis=0) * fps, axis=2)
        speed[~valid] = np.nan
        all_speeds.append(speed)
        distances = np.linalg.norm(players - ball_xy[:, None, :], axis=2)
        distances[~np.isfinite(distances)] = np.inf
        order = np.argsort(distances, axis=1)
        take = min(nearest_count, players.shape[1])
        nearest[:, team, :take] = np.take_along_axis(
            distances, order[:, :take], axis=1
        )
        rows = np.arange(n_frames)
        visible = np.isfinite(nearest[:, team, 0])
        nearest_xy[visible, team] = players[rows[visible], order[visible, 0]]
        nearest_slot[visible, team] = order[visible, 0]
        age = np.zeros_like(valid, dtype=np.int32)
        last_seen = np.full(players.shape[1], -1, dtype=np.int64)
        last_age = np.zeros(players.shape[1], dtype=np.int32)
        last_xy = np.full((players.shape[1], 2), np.nan, dtype=players.dtype)
        for frame in range(n_frames):
            age[frame] = _update_track_ages(
                players[frame], frame, last_seen, last_age, last_xy, fps, resume_gap,
                config["max_player_speed_metres_per_second"],
            )
        nearest_age[visible, team] = age[rows[visible], order[visible, 0]]
        nearest_speed[visible, team] = speed[rows[visible], order[visible, 0]]
        velocity = np.full_like(players, np.nan, dtype=np.float32)
        velocity[1:] = player_velocity_steps(
            players[:-1], players[1:], fps, config["max_player_speed_metres_per_second"]
        )
        nearest_velocity[visible, team] = velocity[rows[visible], order[visible, 0]]
        for index, threshold in enumerate(thresholds):
            counts[:, team, index] = np.count_nonzero(
                distances <= threshold, axis=1
            )
    speeds = np.concatenate(all_speeds, axis=1)
    finite_speeds = np.isfinite(speeds)
    available = np.any(finite_speeds, axis=1)
    median_speed = np.full(n_frames, np.nan, dtype=np.float32)
    stationary_fraction = np.full(n_frames, np.nan, dtype=np.float32)
    median_speed[available] = np.nanmedian(speeds[available], axis=1)
    stationary_fraction[available] = (
        np.count_nonzero(
            finite_speeds[available]
            & (speeds[available] < config["stationary_speed_threshold_metres_per_second"]),
            axis=1,
        ) / np.count_nonzero(finite_speeds[available], axis=1)
    )
    return ControlFeatures(
        nearest_distances=nearest,
        nearby_counts=counts,
        nearest_player_xy=nearest_xy,
        nearest_player_speed=nearest_speed,
        nearest_player_velocity=nearest_velocity,
        player_counts=player_counts,
        median_player_speed=median_speed,
        stationary_player_fraction=stationary_fraction,
        nearest_player_slot=nearest_slot,
        nearest_player_age_frames=nearest_age,
        team_x_extremes=team_x_extremes,
    )
