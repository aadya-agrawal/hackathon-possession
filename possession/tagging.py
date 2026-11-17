import difflib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace

import numpy as np

from possession import AWAY, HOME, UNKNOWN


@dataclass
class TagFeatures:
    team_prior: np.ndarray
    dead_hint: np.ndarray
    pre_restart_window: np.ndarray
    restart_team: np.ndarray
    restart_events: tuple = ()
    restart_frames: np.ndarray = None
    restart_localized: np.ndarray = None
    restart_types: tuple = ()
    restart_teams: np.ndarray = None
    restart_team_inferred: np.ndarray = None


def _normalize_name(value):
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _team_code(tagged_name, team_names):
    normalized = _normalize_name(tagged_name)
    if not normalized or normalized == "na":
        return UNKNOWN
    candidates = [_normalize_name(name) for name in team_names]
    scores = [difflib.SequenceMatcher(None, normalized, candidate).ratio()
              for candidate in candidates]
    best = int(np.argmax(scores))
    return best if scores[best] >= 0.6 else UNKNOWN


def load_tag_features(
    smart_tagging_path,
    team_names,
    n_frames,
    fps,
    restart_offset_frames,
    restart_prior_frames=50,
    dead_lookback_frames=375,
    team_prior_mode="all",
):
    if team_prior_mode not in ("all", "exclude_defending", "signed"):
        raise ValueError("tag team prior mode must be all, exclude_defending or signed")
    team_sum_delta = np.zeros(n_frames + 1, dtype=np.float32)
    team_count_delta = np.zeros(n_frames + 1, dtype=np.int16)
    dead_delta = np.zeros(n_frames + 1, dtype=np.int16)
    pre_restart_delta = np.zeros(n_frames + 1, dtype=np.int16)
    restart_team = np.full(n_frames, UNKNOWN, dtype=np.int8)
    restart_events = []
    restart_types = []

    for instance in ET.parse(smart_tagging_path).iter("instance"):
        start = max(0, int(round(float(instance.findtext("start")) * fps)))
        end = min(n_frames, int(round(float(instance.findtext("end")) * fps)))
        if end <= start:
            continue

        code = (instance.findtext("code") or "").strip().lower()
        tags = {
            label.findtext("group"): label.findtext("text")
            for label in instance.findall("label")
        }
        team = _team_code(tags.get("Team"), team_names)
        # Defending phases (and defensive transitions) name the team without the ball.
        defending = code.startswith("defend") or code.startswith("defensive")
        if team != UNKNOWN and not (team_prior_mode == "exclude_defending" and defending):
            sign = 1.0 if team == HOME else -1.0
            if team_prior_mode == "signed" and defending:
                sign = -sign
            team_sum_delta[start] += sign
            team_sum_delta[end] -= sign
            team_count_delta[start] += 1
            team_count_delta[end] -= 1

        if code == "set pieces":
            dead_delta[start] += 1
            restart = min(end, start + restart_offset_frames)
            restart_events.append((start, end, restart, team))
            restart_types.append(
                (tags.get("Type") or "unknown").strip().lower()
            )
            dead_delta[restart] -= 1
            lookback_start = max(0, restart - dead_lookback_frames)
            pre_restart_delta[lookback_start] += 1
            pre_restart_delta[restart] -= 1
            if team != UNKNOWN and restart < n_frames:
                restart_end = min(n_frames, restart + restart_prior_frames)
                restart_team[restart:restart_end] = team

    team_sum = np.cumsum(team_sum_delta[:-1])
    team_count = np.cumsum(team_count_delta[:-1])
    team_prior = np.zeros(n_frames, dtype=np.float32)
    np.divide(team_sum, team_count, out=team_prior, where=team_count > 0)
    dead_hint = np.cumsum(dead_delta[:-1]) > 0
    pre_restart_window = np.cumsum(pre_restart_delta[:-1]) > 0
    return TagFeatures(
        team_prior=team_prior,
        dead_hint=dead_hint,
        pre_restart_window=pre_restart_window,
        restart_team=restart_team,
        restart_events=tuple(restart_events),
        restart_frames=np.asarray(
            [event[2] for event in restart_events], dtype=np.int32
        ),
        restart_localized=np.zeros(len(restart_events), dtype=bool),
        restart_types=tuple(restart_types),
        restart_teams=np.asarray(
            [event[3] for event in restart_events], dtype=np.int8
        ),
        restart_team_inferred=np.zeros(len(restart_events), dtype=bool),
    )


def localize_restarts(tag_features, trajectory, fps, config):
    if (
        not config["restart_localization_enabled"]
        or not tag_features.restart_events
    ):
        return tag_features

    n_frames = len(trajectory.xy)
    radius = int(config["restart_localization_search_radius_frames"])
    window = int(config["restart_localization_motion_window_frames"])
    min_post_speed = float(
        config["restart_localization_min_post_speed_metres_per_second"]
    )
    max_pre_speed = float(
        config["restart_localization_max_pre_speed_metres_per_second"]
    )
    min_speed_gain = float(
        config["restart_localization_min_speed_gain_metres_per_second"]
    )
    min_pre_observed = int(
        config["restart_localization_min_pre_observed_frames"]
    )
    max_reappearance_pre_observed = int(
        config["restart_localization_max_reappearance_pre_observed_frames"]
    )
    min_post_reliable = int(
        config["restart_localization_min_post_reliable_frames"]
    )
    dead_lookback = int(config["dead_lookback_frames"])
    restart_prior = int(config["restart_prior_frames"])

    restart_frames = []
    localized = []
    movement_speed = np.full(n_frames, np.nan, dtype=np.float32)
    valid_xy = np.isfinite(trajectory.xy).all(axis=1)
    valid_displacement = valid_xy[window:] & valid_xy[:-window]
    movement_speed[window:][valid_displacement] = (
        np.linalg.norm(
            trajectory.xy[window:][valid_displacement]
            - trajectory.xy[:-window][valid_displacement],
            axis=1,
        )
        * (float(fps) / float(window))
    )
    for start, end, fixed_restart, team in tag_features.restart_events:
        search_start = max(start, fixed_restart - radius)
        search_end = min(end, n_frames - window, fixed_restart + radius)
        candidates = []
        for frame in range(search_start, search_end + 1):
            pre_start = max(start, frame - window)
            pre_speed = movement_speed[pre_start:frame]
            post_speed = movement_speed[frame:frame + window]
            pre_observed = np.count_nonzero(
                trajectory.source[pre_start:frame] == 1
            )
            post_reliable = np.count_nonzero(
                np.isin(
                    trajectory.source[frame:frame + window],
                    [1, 2],
                )
            )
            post_median = (
                np.nanmedian(post_speed)
                if np.isfinite(post_speed).any()
                else np.nan
            )
            if (
                post_reliable < min_post_reliable
                or not np.isfinite(movement_speed[frame])
                or movement_speed[frame] < min_post_speed
                or not np.isfinite(post_median)
                or post_median < min_post_speed
            ):
                continue

            pre_median = (
                np.nanmedian(pre_speed)
                if np.isfinite(pre_speed).any()
                else np.nan
            )
            onset = (
                pre_observed >= min_pre_observed
                and np.isfinite(pre_median)
                and pre_median <= max_pre_speed
                and post_median - pre_median >= min_speed_gain
            )
            reappearance = (
                pre_observed <= max_reappearance_pre_observed
            )
            if onset or reappearance:
                candidates.append(frame)

        if candidates:
            restart = min(
                candidates,
                key=lambda frame: (abs(frame - fixed_restart), frame),
            )
            localized.append(True)
        else:
            restart = fixed_restart
            localized.append(False)
        restart_frames.append(restart)

    dead_delta = np.zeros(n_frames + 1, dtype=np.int16)
    pre_restart_delta = np.zeros(n_frames + 1, dtype=np.int16)
    restart_team = np.full(n_frames, UNKNOWN, dtype=np.int8)
    for event, restart in zip(tag_features.restart_events, restart_frames):
        start, _, _, team = event
        dead_delta[start] += 1
        dead_delta[restart] -= 1
        lookback_start = max(0, restart - dead_lookback)
        pre_restart_delta[lookback_start] += 1
        pre_restart_delta[restart] -= 1
        if team != UNKNOWN and restart < n_frames:
            restart_end = min(n_frames, restart + restart_prior)
            restart_team[restart:restart_end] = team

    return replace(
        tag_features,
        dead_hint=np.cumsum(dead_delta[:-1]) > 0,
        pre_restart_window=np.cumsum(pre_restart_delta[:-1]) > 0,
        restart_team=restart_team,
        restart_frames=np.asarray(restart_frames, dtype=np.int32),
        restart_localized=np.asarray(localized, dtype=bool),
    )


def estimate_restart_owners(tag_features, trajectory, control_features, config):
    n_events = len(tag_features.restart_events)
    predictions = np.full(n_events, UNKNOWN, dtype=np.int8)
    winner_fractions = np.zeros(n_events, dtype=np.float32)
    vote_counts = np.zeros(n_events, dtype=np.int16)
    nearest = control_features.nearest_distances[:, :, 0]
    nearby = control_features.nearby_counts[:, :, 1]
    reliable = np.isin(trajectory.source, [1, 2])
    type_windows = config["restart_ownership_window_frames_by_type"]
    default_window = int(config["restart_ownership_default_window_frames"])
    max_distance = float(
        config["restart_ownership_max_distance_metres"]
    )
    distance_margin = float(
        config["restart_ownership_distance_margin_metres"]
    )
    minimum_votes = int(config["restart_ownership_minimum_votes"])
    minimum_fraction = float(
        config["restart_ownership_minimum_winner_fraction"]
    )

    for index, restart in enumerate(tag_features.restart_frames):
        restart_type = tag_features.restart_types[index]
        window = int(type_windows.get(restart_type, default_window))
        end = min(len(reliable), int(restart) + window)
        votes = []
        for frame in range(int(restart), end):
            if not reliable[frame] or not np.isfinite(nearest[frame]).all():
                continue
            team = int(np.argmin(nearest[frame]))
            opponent = AWAY if team == HOME else HOME
            advantage = nearest[frame, opponent] - nearest[frame, team]
            density_advantage = nearby[frame, team] > nearby[frame, opponent]
            if (
                nearest[frame, team] <= max_distance
                and (
                    advantage >= distance_margin
                    or density_advantage
                )
            ):
                votes.append(team)

        vote_counts[index] = len(votes)
        if len(votes) < minimum_votes:
            continue
        away_fraction = float(np.mean(np.asarray(votes) == AWAY))
        winner = AWAY if away_fraction >= 0.5 else HOME
        winner_fraction = max(away_fraction, 1.0 - away_fraction)
        winner_fractions[index] = winner_fraction
        if winner_fraction >= minimum_fraction:
            predictions[index] = winner
    return predictions, winner_fractions, vote_counts


def infer_restart_ownership(
    tag_features, trajectory, control_features, config
):
    if (
        not config["restart_ownership_enabled"]
        or not tag_features.restart_events
    ):
        return tag_features

    predictions, _, _ = estimate_restart_owners(
        tag_features, trajectory, control_features, config
    )
    event_teams = tag_features.restart_teams.copy()
    inferred = np.zeros(len(event_teams), dtype=bool)
    eligible_types = set(config["restart_ownership_eligible_types"])
    phase_window = int(config["restart_ownership_phase_window_frames"])
    minimum_phase_prior = float(
        config["restart_ownership_minimum_phase_prior"]
    )
    prior_frames_by_type = (
        config["restart_ownership_prior_frames_by_type"]
    )
    default_prior_frames = int(
        config["restart_ownership_default_prior_frames"]
    )
    restart_team = tag_features.restart_team.copy()

    for index, prediction in enumerate(predictions):
        if (
            event_teams[index] != UNKNOWN
            or prediction == UNKNOWN
            or tag_features.restart_types[index] not in eligible_types
        ):
            continue
        restart = int(tag_features.restart_frames[index])
        phase_end = min(len(tag_features.team_prior), restart + phase_window)
        phase = tag_features.team_prior[restart:phase_end]
        mean_phase = float(np.mean(phase)) if len(phase) else 0.0
        phase_team = UNKNOWN
        if abs(mean_phase) >= minimum_phase_prior:
            phase_team = HOME if mean_phase > 0.0 else AWAY
        if phase_team != UNKNOWN and phase_team != prediction:
            continue

        event_teams[index] = prediction
        inferred[index] = True
        restart_type = tag_features.restart_types[index]
        prior_frames = int(
            prior_frames_by_type.get(
                restart_type, default_prior_frames
            )
        )
        restart_end = min(len(restart_team), restart + prior_frames)
        restart_team[restart:restart_end] = prediction

    return replace(
        tag_features,
        restart_team=restart_team,
        restart_teams=event_teams,
        restart_team_inferred=inferred,
    )
