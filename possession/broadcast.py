"""Automatic broadcast-to-tactical alignment and broadcast view state, from tracking only.

The broadcast recording has its own tracking timeline with cuts, replays and edits. Player
positions on the pitch are camera independent, so broadcast frames are matched to tactical frames
by player formation. Offsets are discovered from the data (never hardcoded), and a tactical frame
is only described by the broadcast when matched live footage at the same offset brackets it.
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from possession.tracking import _parse_group, atd_positions, load_atd_arrays

_SEARCH = {}
# Candidate count from which formation_distance uses the running-minimum evaluation.
_RUNNING_MIN_CANDIDATES = 64


def _global_match(frame):
    # Forked worker: the arrays are inherited, so only the frame index crosses processes.
    s = _SEARCH
    distance = formation_distance(s["broadcast_xy"][frame], s["broadcast_team"], s["candidates"], s["tactical_team"])
    best = int(np.nanargmin(np.where(np.isfinite(distance), distance, np.inf)))
    return best, float(distance[best])

UNCOVERED, LIVE_VIEW, OTHER_VIEW, NO_PITCH, REPLAY = 0, 1, 2, 3, 4


def _nearest_order(tactical_xy, tactical_team, team, players):
    # Memory order the reference expression gives its (candidates, players) result; it fixes the
    # summation order over players. It depends on strides, not on the number of candidates.
    probe = tactical_xy[:3][:, tactical_team == team]
    distance = np.linalg.norm(probe[:, None, :, :] - players[None, :, None, :], axis=3)
    nearest = np.nanmin(np.where(np.isfinite(distance), distance, np.inf), axis=2)
    if nearest.flags.c_contiguous:
        return "C"
    return "F" if nearest.flags.f_contiguous else None


def formation_distance(broadcast_xy, broadcast_team, tactical_xy, tactical_team, cap=5.0):
    """Mean capped distance from each broadcast-tracked player to the nearest same-team tactical
    player. broadcast_xy (K,2) for one frame; tactical_xy (N,M,2) for N candidate frames."""
    total = np.zeros(len(tactical_xy))
    count = 0
    for team in (0, 1):
        players = broadcast_xy[broadcast_team == team]
        players = players[np.isfinite(players).all(axis=1)]
        if not len(players):
            continue
        candidates = tactical_xy[:, tactical_team == team]
        order = (_nearest_order(tactical_xy, tactical_team, team, players)
                 if len(candidates) >= _RUNNING_MIN_CANDIDATES and candidates.shape[1] else None)
        if order is not None:
            # sqrt is monotonic, so the nearest squared distance gives the identical nearest distance;
            # a running minimum over tactical slots avoids the (frames, players, slots, 2) array.
            squared = np.full((len(candidates), len(players)), np.nan, order=order)
            for slot in range(candidates.shape[1]):
                dx = candidates[:, slot, None, 0] - players[None, :, 0]
                dy = candidates[:, slot, None, 1] - players[None, :, 1]
                np.fmin(squared, dx * dx + dy * dy, out=squared)
            nearest = np.sqrt(np.where(np.isnan(squared), np.inf, squared), order=order)
        else:
            distance = np.linalg.norm(candidates[:, None, :, :] - players[None, :, None, :], axis=3)
            nearest = np.nanmin(np.where(np.isfinite(distance), distance, np.inf), axis=2)
        total += np.minimum(nearest, cap).sum(axis=1)
        count += len(players)
    return total / count if count else np.full(len(tactical_xy), np.nan)


def _verify_frames(frames):
    s = _SEARCH
    out = []
    for frame in frames:
        best_distance, best_offset = np.inf, None
        for offset in s["offsets"]:
            rows = np.arange(frame + offset - s["radius"], frame + offset + s["radius"] + 1)
            rows = rows[(rows >= 0) & (rows < len(s["tactical_xy"]))]
            if not len(rows):
                continue
            distance = formation_distance(s["broadcast_xy"][frame], s["broadcast_team"],
                                          s["tactical_xy"][rows], s["tactical_team"])
            k = int(np.nanargmin(np.where(np.isfinite(distance), distance, np.inf)))
            if distance[k] < best_distance:
                best_distance, best_offset = distance[k], int(rows[k]) - int(frame)
        out.append((best_distance, best_offset))
    return out


def _replay_hit(frame):
    s = _SEARCH
    first = max(0, s["mapped"][frame] - s["lookback"])
    last = s["mapped"][frame] - s["lag"]
    if last <= first:
        return False
    distance = formation_distance(s["broadcast_xy"][frame], s["broadcast_team"],
                                  s["tactical_xy"][first:last], s["tactical_team"])
    return bool(np.nanmin(np.where(np.isfinite(distance), distance, np.inf)) <= s["limit"])


def discover_offsets(tactical_xy, tactical_team, broadcast_xy, broadcast_team, broadcast_valid, config):
    """Candidate offsets (tactical - broadcast) supported by enough independent global matches."""
    counts = np.isfinite(broadcast_xy[..., 0]).sum(axis=1)
    usable = np.flatnonzero(broadcast_valid & (counts >= config["broadcast_minimum_players"]))
    if not len(usable):
        return []
    samples = usable[np.linspace(0, len(usable) - 1, min(len(usable), config["broadcast_offset_samples"])).astype(int)]
    stride = config["broadcast_search_stride_frames"]
    candidates = tactical_xy[::stride]
    votes = []
    _SEARCH.update(broadcast_xy=broadcast_xy, broadcast_team=broadcast_team,
                   candidates=candidates, tactical_team=tactical_team)
    workers = min(len(samples), os.cpu_count() or 1, 32)
    try:
        # Independent searches; results are identical to a sequential loop, in sample order.
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
            matches = list(pool.map(_global_match, samples, chunksize=4))
    finally:
        _SEARCH.clear()
    for frame, (best, distance) in zip(samples, matches):
        if distance <= config["broadcast_match_distance_metres"]:
            votes.append(best * stride - int(frame))
    votes = np.sort(np.asarray(votes))
    offsets = []
    tolerance = config["broadcast_offset_tolerance_frames"] + stride
    while len(votes):
        centre = votes[np.argmax([np.sum(np.abs(votes - v) <= tolerance) for v in votes])]
        near = np.abs(votes - centre) <= tolerance
        if near.sum() < config["broadcast_minimum_offset_votes"]:
            break
        offsets.append(int(np.median(votes[near])))
        votes = votes[~near]
    return sorted(offsets)


def _bracket_medians(matched_offset, lefts, rights, window, radius):
    """Per bracket, np.median of the finite offsets within `window` frames that lie within
    2*radius of the left match's offset (matched offsets are whole frame counts)."""
    matches = np.flatnonzero(np.isfinite(matched_offset))
    values = np.unique(matched_offset[matches])
    if not len(lefts):
        return []
    if len(values) > 64 or not np.array_equal(values, np.round(values)):
        medians = []
        for left, right in zip(lefts, rights):
            local = matched_offset[max(0, left - window):right + window + 1]
            local = local[np.isfinite(local) & (np.abs(local - matched_offset[left]) <= 2 * radius)]
            medians.append(np.median(local))
        return medians
    # Counts of each distinct offset before every frame, so window counts are two lookups.
    counts = np.zeros((len(matched_offset) + 1, len(values)), dtype=np.int64)
    counts[matches + 1, np.searchsorted(values, matched_offset[matches])] = 1
    np.cumsum(counts, axis=0, out=counts)
    lower = np.maximum(0, lefts - window)
    upper = np.minimum(len(matched_offset), rights + window + 1)
    near = np.abs(values[None, :] - matched_offset[lefts][:, None]) <= 2 * radius
    window_counts = np.where(near, counts[upper] - counts[lower], 0)
    total = window_counts.sum(axis=1)
    below = np.cumsum(window_counts, axis=1)
    low = values[np.argmax(below > ((total - 1) // 2)[:, None], axis=1)]
    high = values[np.argmax(below > (total // 2)[:, None], axis=1)]
    # Same arithmetic as np.median: the middle value, or the mean of the two middle values.
    return np.where(total % 2 == 1, low, (low + high) / 2)


def align_broadcast(tactical_xy, tactical_team, broadcast_xy, broadcast_team, broadcast_valid, config):
    """Per-broadcast-frame tactical frame (or -1) and per-tactical-frame broadcast view state."""
    n_tactical, n_broadcast = len(tactical_xy), len(broadcast_xy)
    offsets = discover_offsets(tactical_xy, tactical_team, broadcast_xy, broadcast_team,
                               broadcast_valid, config)
    counts = np.isfinite(broadcast_xy[..., 0]).sum(axis=1)
    radius = config["broadcast_offset_tolerance_frames"]
    matched_offset = np.full(n_broadcast, np.nan)
    frames = np.flatnonzero(broadcast_valid & (counts >= config["broadcast_minimum_players"]))
    _SEARCH.update(broadcast_xy=broadcast_xy, broadcast_team=broadcast_team, tactical_xy=tactical_xy,
                   tactical_team=tactical_team, offsets=offsets, radius=radius)
    chunks = np.array_split(frames, max(1, min(len(frames) // 500, 64)))
    try:
        with ProcessPoolExecutor(min(len(chunks), os.cpu_count() or 1, 32),
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            verified = [item for chunk in pool.map(_verify_frames, chunks) for item in chunk]
    finally:
        _SEARCH.clear()
    for frame, (best_distance, best_offset) in zip(frames, verified):
        if best_distance <= config["broadcast_match_distance_metres"]:
            matched_offset[frame] = best_offset
    # A frame inherits an offset only when matches at that offset bracket it within the gap limit.
    mapped = np.full(n_broadcast, -1, dtype=np.int64)
    matches = np.flatnonzero(np.isfinite(matched_offset))
    window = config["broadcast_offset_smoothing_frames"]
    lefts, rights = matches[:-1], matches[1:]
    bracketed = ~((np.abs(matched_offset[lefts] - matched_offset[rights]) > 2 * radius)
                  | (rights - lefts > config["broadcast_maximum_bracket_frames"]))
    lefts, rights = lefts[bracketed], rights[bracketed]
    for left, right, median in zip(lefts, rights, _bracket_medians(matched_offset, lefts, rights, window, radius)):
        offset = int(round(median))
        mapped[left:right + 1] = np.arange(left, right + 1) + offset
    mapped[(mapped < 0) | (mapped >= n_tactical)] = -1
    state = np.full(n_tactical, UNCOVERED, dtype=np.int8)
    covered = np.flatnonzero(mapped >= 0)
    view = np.where(~broadcast_valid[covered], NO_PITCH,
                    np.where(np.isfinite(matched_offset[covered]), LIVE_VIEW, OTHER_VIEW))
    state[mapped[covered]] = view
    replay_frames = 0
    if config.get("broadcast_replay_enabled", False):
        # Pitch shown but not the live moment: if it matches an earlier moment, it is a replay.
        stride = config["broadcast_replay_stride_frames"]
        other_view = ((mapped >= 0) & broadcast_valid
                      & (counts >= config["broadcast_minimum_players"]) & ~np.isfinite(matched_offset))
        candidates = np.flatnonzero(other_view)[::stride]
        _SEARCH.update(broadcast_xy=broadcast_xy, broadcast_team=broadcast_team, tactical_xy=tactical_xy,
                       tactical_team=tactical_team, mapped=mapped, limit=config["broadcast_match_distance_metres"],
                       lookback=config["broadcast_replay_lookback_frames"],
                       lag=config["broadcast_replay_minimum_lag_frames"])
        try:
            with ProcessPoolExecutor(min(max(1, len(candidates)), os.cpu_count() or 1, 32),
                                     mp_context=multiprocessing.get_context("fork")) as pool:
                found = list(pool.map(_replay_hit, candidates, chunksize=16))
        finally:
            _SEARCH.clear()
        hits = [int(frame) for frame, hit in zip(candidates, found) if hit]
        replay = np.zeros(n_broadcast, dtype=bool)
        for left, right in zip(hits[:-1], hits[1:]):
            if right - left <= 2 * stride:
                replay[left:right + 1] = True
        replay[hits] = True
        replay &= other_view
        state[mapped[replay]] = REPLAY
        replay_frames = int(np.count_nonzero(replay))
    diagnostics = {
        "broadcast_replay_frames": replay_frames,
        "broadcast_offsets": offsets,
        "broadcast_matched_frames": int(len(matches)),
        "broadcast_covered_tactical_frames": int(np.count_nonzero(state != UNCOVERED)),
        "broadcast_no_pitch_tactical_frames": int(np.count_nonzero(state == NO_PITCH)),
    }
    return mapped, state, diagnostics


def broadcast_ball_observations(mapped, view_state, broadcast_ball, broadcast_xy, tactical_source, fps, config):
    """Broadcast ball positions for tactical frames whose own ball is missing.

    Only matched live footage counts, and only when the broadcast ball is plausible: a broadcast
    player is close to it and it is not moving fast (aerial or misdetected balls project badly).
    """
    n_broadcast = len(mapped)
    span = 4
    speed = np.full(n_broadcast, np.inf)
    if n_broadcast > span:
        step = np.linalg.norm(broadcast_ball[span:] - broadcast_ball[:-span], axis=1) * fps / span
        speed[span // 2:n_broadcast - span // 2] = np.where(np.isfinite(step), step, np.inf)
    distance = np.linalg.norm(broadcast_xy - broadcast_ball[:, None, :], axis=2)
    nearest = np.min(np.where(np.isfinite(distance), distance, np.inf), axis=1)
    live = mapped >= 0
    live[live] = view_state[mapped[live]] == LIVE_VIEW
    keep = (live & np.isfinite(broadcast_ball).all(axis=1)
            & (nearest <= config["broadcast_ball_player_distance_metres"])
            & (speed <= config["broadcast_ball_maximum_speed_mps"]))
    frames = np.flatnonzero(keep)
    frames = frames[tactical_source[mapped[frames]] == 0]
    tactical, first = np.unique(mapped[frames], return_index=True)
    xy = broadcast_ball[frames[first]].astype(np.float32)
    # Drop observations whose step from the previous kept frame is physically impossible.
    kept = np.ones(len(tactical), dtype=bool)
    last = -1
    for i in range(len(tactical)):
        if (last >= 0 and tactical[i] - tactical[last] == 1
                and np.linalg.norm(xy[i] - xy[last]) * fps > config["max_ball_speed_metres_per_second"]):
            kept[i] = False
            continue
        last = i
    return tactical[kept], xy[kept], np.ones(int(kept.sum()), dtype=np.float32)


def load_player_array(raw_path, team_of_track, pitch):
    """All tracked player positions in metres, (frames, slots, 2) with NaN where untracked."""
    atd = load_atd_arrays(raw_path)
    if atd is not None and len(atd.players) and atd.players.shape[1] == len(team_of_track):
        return atd_positions(atd.players, pitch)[0]
    length, width = pitch
    rows = []
    with open(raw_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split(":", 2)
            if len(parts) != 3 or int(parts[0]) != line_number:
                raise ValueError("{}:{}: malformed or misaligned ATD frame".format(raw_path, line_number))
            groups = parts[1].split(";")
            if len(groups) != len(team_of_track):
                raise ValueError("{}:{}: unexpected track slot count".format(raw_path, line_number))
            frame = np.full((len(groups), 2), np.nan)
            for slot, group in enumerate(groups):
                player = _parse_group(group)
                if player is not None:
                    frame[slot] = (player[0] * length, player[1] * width)
            rows.append(frame)
    return np.asarray(rows)


def validate_broadcast_config(config):
    enabled = config.get("broadcast_stoppage_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("broadcast_stoppage_enabled must be boolean")
    if not enabled:
        return
    for key, minimum in (("broadcast_minimum_players", 2), ("broadcast_offset_samples", 10),
                         ("broadcast_search_stride_frames", 1), ("broadcast_offset_tolerance_frames", 0),
                         ("broadcast_minimum_offset_votes", 2), ("broadcast_offset_smoothing_frames", 1),
                         ("broadcast_maximum_bracket_frames", 1), ("broadcast_stoppage_minimum_frames", 1)):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError("{} must be an integer of at least {}".format(key, minimum))
    if config.get("broadcast_replay_enabled", False):
        for key, minimum in (("broadcast_replay_stride_frames", 1), ("broadcast_replay_lookback_frames", 1),
                             ("broadcast_replay_minimum_lag_frames", 1)):
            value = config.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError("{} must be an integer of at least {}".format(key, minimum))
    fusion = config.get("broadcast_ball_fusion_enabled", False)
    if not isinstance(fusion, bool):
        raise ValueError("broadcast_ball_fusion_enabled must be boolean")
    keys = ("broadcast_match_distance_metres", "broadcast_stoppage_dead_weight",
            "broadcast_stoppage_ball_speed_mps")
    if fusion:
        keys += ("broadcast_ball_player_distance_metres", "broadcast_ball_maximum_speed_mps")
    for key in keys:
        value = config.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not np.isfinite(value) or value <= 0):
            raise ValueError("{} must be finite and positive".format(key))


def broadcast_stoppage_intervals(view_state, trajectory, tag_features, fps, config):
    """Dead-ball mask from sustained broadcast cutaways (replays, close-ups) during covered time.

    Broadcasters cut away from the pitch when play stops. A cutaway only counts when matched
    live footage at the same offset brackets it, and it ends early when the tactical ball is seen
    moving fast (play has resumed) or at a tagged restart.
    """
    n = len(trajectory.source)
    if view_state.shape != (n,):
        raise ValueError("broadcast view state must align with the tactical timeline")
    observed = trajectory.source == 1
    # Speed over 0.2 s so per-frame position noise cannot fake a moving ball.
    span = max(1, int(round(0.2 * fps)))
    speed = np.zeros(n)
    if n > span:
        step = np.linalg.norm(trajectory.xy[span:] - trajectory.xy[:-span], axis=1) * fps / span
        paired = observed[span:] & observed[:-span] & np.isfinite(step)
        speed[span:][paired] = step[paired]
    fast = speed >= config["broadcast_stoppage_ball_speed_mps"]
    fast = np.convolve(fast.astype(int), np.ones(3, dtype=int), "same") >= 3
    restarts = np.asarray(tag_features.restart_frames if tag_features.restart_frames is not None else [],
                          dtype=np.int64)
    cutaway = np.r_[False, np.isin(view_state, (NO_PITCH, REPLAY)), False]
    edges = np.flatnonzero(np.diff(cutaway.astype(int)))
    dead = np.zeros(n, dtype=bool)
    diagnostics = {"broadcast_cutaways": 0, "broadcast_cutaways_truncated": 0,
                   "broadcast_stoppage_frames": 0}
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start < config["broadcast_stoppage_minimum_frames"]:
            continue
        ends = [stop]
        moving = np.flatnonzero(fast[start:stop])
        if moving.size:
            ends.append(start + int(moving[0]))
        inner = restarts[(restarts > start) & (restarts < stop)]
        if inner.size:
            ends.append(int(inner[0]))
        end = min(ends)
        diagnostics["broadcast_cutaways_truncated"] += int(end < stop)
        if end - start >= config["broadcast_stoppage_minimum_frames"]:
            dead[start:end] = True
            diagnostics["broadcast_cutaways"] += 1
    diagnostics["broadcast_stoppage_frames"] = int(np.count_nonzero(dead))
    return dead, diagnostics
