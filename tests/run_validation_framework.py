import argparse
from types import SimpleNamespace
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require
from possession import AWAY, HOME
from possession.ballfree import configured_home_probability
from possession.broadcast import broadcast_ball_observations, load_player_array
from possession.cli import DEFAULT_CONFIG, broadcast_view_state, configured_frame_teams, load_config, load_video_team
from possession.decoder import build_evidence, decode
from possession.tagging import (
    estimate_restart_owners,
    infer_restart_ownership,
    load_tag_features,
    localize_restarts,
)
from possession.tracking import (
    extract_control_features,
    load_raw_ball,
    load_tracking_metadata,
)
from possession.trajectory import (
    fuse_video_ball_observations,
    load_video_ball_observations,
    repair_ball_trajectory,
)
from possession.video import (
    load_homography_quality, make_long_gap_edge_dropout,
    select_input_ball_anchor_samples,
)


HALF_STARTS = [first for _, first, _ in half_video_blocks(GAME)]
HALF_BLOCKS = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
DEFAULT_POLICY = os.path.join(ROOT, "tests", "robustness_policy.json")
DEFAULT_REFERENCE = os.path.join(ROOT, "submissions", "possession_mvp.csv")
DEFAULT_REPORT = os.path.join(ROOT, "artifacts", "validation_framework.json")
DEFAULT_LEDGER = os.path.join(ROOT, "experiments", "results.csv")
LEDGER_FIELDS = [
    "recorded_at_utc",
    "framework_version",
    "family",
    "variant",
    "decision",
    "passed",
    "checks_passed",
    "checks_total",
    "failed_checks",
    "config_sha256",
    "clean_label_sha256",
    "reference_label_sha256",
    "possessions",
    "shorter_than_one_second",
    "strong_event_persistence",
    "restarts_localized",
    "restart_speed_gain",
    "noise_0.10m_frame_agreement",
    "noise_0.10m_boundary_recall",
    "noise_0.25m_frame_agreement",
    "noise_0.25m_boundary_recall",
    "missing_25_frames_frame_agreement",
    "retain_50_percent_frame_agreement",
    "retain_50_percent_boundary_recall",
    "restart_owners_inferred",
    "restart_owner_proxy_coverage",
    "restart_owner_proxy_accuracy",
    "decoded_boundary_majority_persistence",
    "decoded_boundary_strong_persistence",
    "team_switches",
    "notes",
]


@dataclass
class Context:
    n_frames: int
    fps: float
    pitch: tuple
    raw_path: str
    metadata: object
    base_tags: object
    broadcast_view: object = None
    ballfree_probability: object = None
    broadcast_alignment: object = None
    frame_teams: object = None
    players_xy: object = None


@dataclass
class Inference:
    labels: np.ndarray
    trajectory: object
    control: object
    tags: object
    evidence: object


def prepare_context(data_dir, config):
    game = GAMES[GAME]
    meta_path, raw_path, tagging_path = require(
        ["atd_meta", "atd_raw", "smart_tagging"], data_dir=data_dir
    )
    metadata = load_tracking_metadata(meta_path)
    base_tags = load_tag_features(
        tagging_path,
        metadata.team_names,
        game["n_video_frames"],
        game["fps"],
        int(config["restart_offset_frames"]),
        int(config["restart_prior_frames"]),
        int(config["dead_lookback_frames"]),
        team_prior_mode=config.get("tag_team_prior_mode", "all"),
    )
    broadcast_view = ballfree_probability = players_xy = alignment = None
    if config.get("broadcast_stoppage_enabled", False) or config.get("semi_markov_ballfree_enabled", False):
        players_xy = load_player_array(raw_path, metadata.team_of_track, game["pitch"])
    if config.get("broadcast_stoppage_enabled", False):
        # Alignment and ball-free evidence use players only, so ball perturbations reuse them.
        broadcast_view, _, alignment = broadcast_view_state(
            data_dir, raw_path, metadata, game["n_video_frames"], game["pitch"], config,
            tactical_xy=players_xy, with_alignment=True,
        )
    # The ball-aware model reads the ball, so it is recomputed for every (perturbed) trajectory.
    if config.get("semi_markov_ballfree_enabled", False) and config.get("ballfree_model") != "sequence_ball":
        ballfree_probability = configured_home_probability(
            players_xy, metadata.team_of_track, game["pitch"], game["fps"],
            [(first, last + 1) for _, first, last in half_video_blocks(GAME)], config,
        )
    return Context(
        n_frames=game["n_video_frames"],
        fps=game["fps"],
        pitch=game["pitch"],
        raw_path=raw_path,
        metadata=metadata,
        base_tags=base_tags,
        broadcast_view=broadcast_view,
        ballfree_probability=ballfree_probability,
        broadcast_alignment=alignment,
        frame_teams=configured_frame_teams(players_xy, metadata, game["pitch"], config),
        players_xy=players_xy,
    )


_BALLFREE_MEMO = {}


def ballfree_probability(context, config, trajectory):
    """Players-only probability from the context, or the ball-aware one for this trajectory."""
    if not config.get("semi_markov_ballfree_enabled", False) or config.get("ballfree_model") != "sequence_ball":
        return context.ballfree_probability
    keys = ("ballfree_ball_weights_path", "ballfree_weights_path", "ballfree_ball_scope",
            "ballfree_minimum_unseen_frames", "broadcast_ball_fusion_enabled")
    key = (json.dumps({k: config.get(k) for k in keys}, sort_keys=True),
           hashlib.sha256(np.ascontiguousarray(trajectory.xy).tobytes() + trajectory.source.tobytes()).hexdigest())
    if key not in _BALLFREE_MEMO:
        _BALLFREE_MEMO.clear()
        _BALLFREE_MEMO[key] = configured_home_probability(
            context.players_xy, context.metadata.team_of_track, context.pitch, context.fps,
            [(first, last + 1) for _, first, last in half_video_blocks(GAME)], config, trajectory=trajectory,
        )
    return _BALLFREE_MEMO[key]


def infer(raw_ball_xy, context, config):
    trajectory = repair_ball_trajectory(
        raw_ball_xy,
        context.fps,
        max_gap_frames=int(config["max_interpolation_gap_frames"]),
        max_speed=float(config["max_ball_speed_metres_per_second"]),
        prediction_history_frames=int(config["prediction_history_frames"]),
        prediction_horizon_frames=int(config["prediction_horizon_frames"]),
        max_acceleration=float(
            config["max_ball_acceleration_metres_per_second_squared"]
        ),
        pitch=context.pitch,
        pitch_margin=float(config["prediction_pitch_margin_metres"]),
    )
    if config.get("video_ball_observations_path"):
        video_observation_path = config["video_ball_observations_path"]
        if not os.path.isabs(video_observation_path):
            video_observation_path = os.path.join(
                ROOT, video_observation_path
            )
        video_frames, video_xy, video_confidence = (
            load_video_ball_observations(
                video_observation_path
            )
        )
        trajectory = fuse_video_ball_observations(
            trajectory,
            video_frames,
            video_xy,
            video_confidence,
            fps=context.fps,
            max_speed=float(
                config["max_ball_speed_metres_per_second"]
            ),
            replace_source_codes=tuple(
                config["video_ball_fusion_source_codes"]
            ),
        )
    if config.get("broadcast_ball_fusion_enabled", False):
        mapped, broadcast_xy, broadcast_ball = context.broadcast_alignment
        frames, xy, confidence = broadcast_ball_observations(
            mapped, context.broadcast_view, broadcast_ball, broadcast_xy, trajectory.source,
            context.fps, config,
        )
        trajectory = fuse_video_ball_observations(
            trajectory, frames, xy, confidence, fps=context.fps,
            max_speed=float(config["max_ball_speed_metres_per_second"]), replace_source_codes=(0,),
        )
    control = extract_control_features(
        context.raw_path,
        context.metadata,
        trajectory.xy,
        context.pitch,
        distance_thresholds=tuple(config["distance_thresholds_metres"]),
        nearest_count=int(config["nearest_player_count"]),
        stationary_speed_threshold=float(
            config["stationary_speed_threshold_metres_per_second"]
        ),
        max_player_speed=float(config["max_player_speed_metres_per_second"]),
        fps=context.fps,
        frame_teams=context.frame_teams,
    )
    tags = localize_restarts(
        context.base_tags, trajectory, context.fps, config
    )
    tags = infer_restart_ownership(
        tags, trajectory, control, config
    )
    evidence = build_evidence(
        control, trajectory, tags, context.pitch, config, fps=context.fps,
        period_starts=HALF_STARTS, period_blocks=HALF_BLOCKS, broadcast_view=context.broadcast_view,
        ballfree_home_probability=ballfree_probability(context, config, trajectory),
        video_team=load_video_team(config, trajectory.source),
    )
    return Inference(
        labels=decode(evidence, tags, config),
        trajectory=trajectory,
        control=control,
        tags=tags,
        evidence=evidence,
    )


_SHARED = {}


def _infer_labels(raw_ball_xy):
    # Runs in forked workers; the shared context and config are inherited, not pickled.
    return infer(raw_ball_xy, _SHARED["context"], _SHARED["config"]).labels


def _infer_full(raw_ball_xy):
    return infer(raw_ball_xy, _SHARED["context"], _SHARED["config"])


def _redecode_labels(variant):
    return redecode(_SHARED["clean"], _SHARED["context"], variant).labels


def _labels(labels):
    return SimpleNamespace(labels=labels)


def redecode(clean, context, config):
    tags = localize_restarts(
        context.base_tags, clean.trajectory, context.fps, config
    )
    tags = infer_restart_ownership(
        tags, clean.trajectory, clean.control, config
    )
    evidence = build_evidence(
        clean.control, clean.trajectory, tags, context.pitch, config,
        fps=context.fps, period_starts=HALF_STARTS, period_blocks=HALF_BLOCKS, broadcast_view=context.broadcast_view,
        ballfree_home_probability=ballfree_probability(context, config, clean.trajectory),
        video_team=load_video_team(config, clean.trajectory.source),
    )
    return Inference(
        labels=decode(evidence, tags, config),
        trajectory=clean.trajectory,
        control=clean.control,
        tags=tags,
        evidence=evidence,
    )


def team_switches(labels):
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return changes[
        np.isin(labels[changes - 1], [HOME, AWAY])
        & np.isin(labels[changes], [HOME, AWAY])
    ]


def boundary_recall(reference, candidate, tolerance):
    if len(reference) == 0:
        return 1.0
    return float(np.mean([
        np.any(np.abs(candidate - frame) <= tolerance)
        for frame in reference
    ]))


def spell_metrics(labels, fps):
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(labels)]
    live = np.isin(labels[starts], [HOME, AWAY])
    lengths = ends[live] - starts[live]
    return {
        "possessions": int(len(lengths)),
        "shorter_than_one_second": int(np.count_nonzero(lengths < fps)),
        "shorter_than_two_seconds": int(np.count_nonzero(lengths < 2 * fps)),
        "mean_possession_seconds": (
            float(np.mean(lengths) / fps) if len(lengths) else 0.0
        ),
    }


def strong_event_persistence(inference):
    nearest = inference.control.nearest_distances[:, :, 0]
    values = []
    for team in (HOME, AWAY):
        strong = inference.evidence.strong_control[:, team]
        starts = np.flatnonzero(
            strong & np.r_[True, ~strong[:-1]]
        )
        for frame in starts:
            end = min(len(strong), frame + 26)
            team_distance = nearest[frame + 1:end, team]
            opponent_distance = nearest[frame + 1:end, 1 - team]
            valid = (
                np.isfinite(team_distance)
                & np.isfinite(opponent_distance)
            )
            if np.count_nonzero(valid) >= 5:
                values.append(float(np.mean(
                    team_distance[valid] < opponent_distance[valid]
                )))
    values = np.asarray(values)
    return {
        "events": int(len(values)),
        "majority_persistence": (
            float(np.mean(values >= 0.6)) if len(values) else 0.0
        ),
        "strong_persistence": (
            float(np.mean(values >= 0.8)) if len(values) else 0.0
        ),
    }


def decoded_boundary_persistence(inference):
    labels = inference.labels
    nearest = inference.control.nearest_distances[:, :, 0]
    switches = team_switches(labels)
    persistence = np.full(len(switches), np.nan, dtype=np.float32)
    for index, frame in enumerate(switches):
        team = labels[frame]
        end = min(len(labels), frame + 26)
        team_distance = nearest[frame:end, team]
        opponent_distance = nearest[frame:end, 1 - team]
        valid = (
            np.isfinite(team_distance)
            & np.isfinite(opponent_distance)
        )
        if np.count_nonzero(valid) >= 5:
            persistence[index] = float(np.mean(
                team_distance[valid] < opponent_distance[valid]
            ))
    usable = persistence[np.isfinite(persistence)]
    return {
        "team_switches": int(len(switches)),
        "proxy_events": int(len(usable)),
        "majority_persistence": (
            float(np.mean(usable >= 0.6))
            if len(usable) else 0.0
        ),
        "strong_persistence": (
            float(np.mean(usable >= 0.8))
            if len(usable) else 0.0
        ),
        "_switches": switches,
        "_persistence": persistence,
    }


def boundary_cohorts(labels, restart_frames, half_split_frame, restart_radius):
    switches = team_switches(labels)
    restart_frames = np.asarray(restart_frames, dtype=np.int64)
    if restart_frames.size:
        restart_proximal = np.any(
            np.abs(switches[:, None] - restart_frames[None, :])
            <= restart_radius,
            axis=1,
        )
    else:
        restart_proximal = np.zeros(len(switches), dtype=bool)
    return {
        "first_half": switches[switches < half_split_frame],
        "second_half": switches[switches >= half_split_frame],
        "home_to_away": switches[
            (labels[switches - 1] == HOME)
            & (labels[switches] == AWAY)
        ],
        "away_to_home": switches[
            (labels[switches - 1] == AWAY)
            & (labels[switches] == HOME)
        ],
        "restart_proximal": switches[restart_proximal],
        "open_play": switches[~restart_proximal],
    }


def decoded_boundary_cohort_persistence(decoded_boundaries, cohorts):
    switches = decoded_boundaries["_switches"]
    persistence = decoded_boundaries["_persistence"]
    by_frame = {
        int(frame): float(value)
        for frame, value in zip(switches, persistence)
        if np.isfinite(value)
    }
    metrics = {}
    for name, frames in cohorts.items():
        values = np.asarray(
            [by_frame[int(frame)] for frame in frames if int(frame) in by_frame]
        )
        metrics[name] = {
            "team_switches": int(len(frames)),
            "proxy_events": int(len(values)),
            "majority_persistence": (
                float(np.mean(values >= 0.6)) if len(values) else None
            ),
            "strong_persistence": (
                float(np.mean(values >= 0.8)) if len(values) else None
            ),
        }
    return metrics


def directional_boundary_recall(
    reference_labels,
    candidate_labels,
    reference_frames,
    tolerance,
):
    reference_frames = np.asarray(reference_frames, dtype=np.int64)
    if len(reference_frames) == 0:
        return 1.0
    candidate_switches = team_switches(candidate_labels)
    matched = []
    for frame in reference_frames:
        same_direction = candidate_switches[
            (candidate_labels[candidate_switches - 1]
             == reference_labels[frame - 1])
            & (candidate_labels[candidate_switches]
               == reference_labels[frame])
        ]
        matched.append(np.any(
            np.abs(same_direction - frame) <= tolerance
        ))
    return float(np.mean(matched))


def boundary_cohort_recall(
    reference_labels,
    candidate_labels,
    cohorts,
    tolerance,
):
    return {
        name: {
            "events": int(len(frames)),
            "directional_recall": directional_boundary_recall(
                reference_labels,
                candidate_labels,
                frames,
                tolerance,
            ),
        }
        for name, frames in cohorts.items()
    }


def restart_metrics(inference):
    tags = inference.tags
    localized = tags.restart_localized
    if localized is None or not np.any(localized):
        return {
            "events": len(tags.restart_events),
            "localized": 0,
            "median_pre_speed": None,
            "median_post_speed": None,
            "median_speed_gain": None,
        }
    pre_speed = []
    post_speed = []
    for frame in tags.restart_frames[localized]:
        pre = inference.trajectory.speed[max(0, frame - 10):frame]
        post = inference.trajectory.speed[frame:frame + 10]
        if np.isfinite(pre).any() and np.isfinite(post).any():
            pre_speed.append(float(np.nanmedian(pre)))
            post_speed.append(float(np.nanmedian(post)))
    return {
        "events": len(tags.restart_events),
        "localized": int(np.count_nonzero(localized)),
        "median_pre_speed": float(np.median(pre_speed)),
        "median_post_speed": float(np.median(post_speed)),
        "median_speed_gain": float(
            np.median(np.asarray(post_speed) - np.asarray(pre_speed))
        ),
    }


def restart_stability(reference, candidate):
    reference_localized = reference.tags.restart_localized
    candidate_localized = candidate.tags.restart_localized
    if (
        reference_localized is None
        or candidate_localized is None
        or len(reference_localized) == 0
    ):
        return None
    differences = np.abs(
        candidate.tags.restart_frames - reference.tags.restart_frames
    )
    return {
        "status_agreement": float(np.mean(
            candidate_localized == reference_localized
        )),
        "within_ten_frames": float(np.mean(differences <= 10)),
    }


def restart_ownership_metrics(inference, config):
    predictions, _, _ = estimate_restart_owners(
        inference.tags,
        inference.trajectory,
        inference.control,
        config,
    )
    organizer_teams = np.asarray(
        [event[3] for event in inference.tags.restart_events],
        dtype=np.int8,
    )
    known = organizer_teams >= 0
    covered = known & (predictions >= 0)
    coverage = (
        float(np.count_nonzero(covered) / np.count_nonzero(known))
        if np.count_nonzero(known)
        else 0.0
    )
    accuracy = (
        float(np.mean(predictions[covered] == organizer_teams[covered]))
        if np.any(covered)
        else 0.0
    )
    return {
        "known_events": int(np.count_nonzero(known)),
        "known_covered": int(np.count_nonzero(covered)),
        "known_coverage": coverage,
        "known_accuracy": accuracy,
        "unknown_events": int(np.count_nonzero(~known)),
        "unknown_inferred": int(np.count_nonzero(
            inference.tags.restart_team_inferred
        )),
    }


def restart_ownership_stability(reference, candidate):
    reference_inferred = reference.tags.restart_team_inferred
    candidate_inferred = candidate.tags.restart_team_inferred
    if (
        reference_inferred is None
        or candidate_inferred is None
        or len(reference_inferred) == 0
    ):
        return None
    status_agreement = float(np.mean(
        reference_inferred == candidate_inferred
    ))
    if np.any(reference_inferred):
        retained = candidate_inferred[reference_inferred] & (
            candidate.tags.restart_teams[reference_inferred]
            == reference.tags.restart_teams[reference_inferred]
        )
        clean_owner_retention = float(np.mean(retained))
    else:
        clean_owner_retention = 1.0
    return {
        "status_agreement": status_agreement,
        "clean_owner_retention": clean_owner_retention,
    }


def load_reference(path, n_frames):
    labels = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for expected_frame, row in enumerate(reader):
            if int(row["frame"]) != expected_frame:
                raise ValueError(
                    "{}: expected frame {}, found {}".format(
                        path, expected_frame, row["frame"]
                    )
                )
            labels.append(row["label"])
    if len(labels) != n_frames:
        raise ValueError(
            "{}: expected {} rows, found {}".format(
                path, n_frames, len(labels)
            )
        )
    mapping = {"H": HOME, "A": AWAY, "D": 2, "X": 3}
    return np.asarray([mapping[label] for label in labels], dtype=np.int8)


def make_perturbations(raw_ball_xy, policy):
    finite = np.isfinite(raw_ball_xy).all(axis=1)
    perturbations = []
    for index, sigma in enumerate((0.10, 0.25)):
        noisy = raw_ball_xy.copy()
        rng = np.random.default_rng(int(policy["seed"]) + index)
        noisy[finite] += rng.normal(
            0.0, sigma, size=(np.count_nonzero(finite), 2)
        ).astype(np.float32)
        perturbations.append(("noise_{:.2f}m".format(sigma), noisy))

    for width in (10, 25):
        masked = raw_ball_xy.copy()
        mask = np.zeros(len(masked), dtype=bool)
        for start in range(250, len(masked), 500):
            mask[start:min(start + width, len(masked))] = True
        masked[mask] = np.nan
        perturbations.append(("missing_{}_frames".format(width), masked))

    downsampled = raw_ball_xy.copy()
    observed = np.flatnonzero(finite)
    downsampled[observed[observed % 2 == 1]] = np.nan
    perturbations.append(("retain_50_percent", downsampled))
    return perturbations


def challenge_shaped_edge_stress(raw_ball_xy, context, config, homography_path,
                                 config_path):
    error, valid = load_homography_quality(
        homography_path, context.n_frames
    )
    blocks = half_video_blocks(GAME)
    samples, input_profile = select_input_ball_anchor_samples(
        raw_ball_xy, error, valid, blocks,
        fps=context.fps,
        minimum_gap_frames=50,
        maximum_homography_error=float(
            config["video_recovery_maximum_median_reprojection_error"]
        ),
        maximum_ball_speed=float(
            config["max_ball_speed_metres_per_second"]
        ),
        stable_sample_count=0,
    )
    clean = infer(raw_ball_xy, context, config)
    scenarios = {}
    for length in (269, 938):
        for width in (5, 25):
            masked, hidden, eligible = make_long_gap_edge_dropout(
                raw_ball_xy, samples, blocks, length, width
            )
            labels = infer(masked, context, config).labels
            changed = labels != clean.labels
            windows = np.zeros(len(labels), dtype=bool)
            for item in eligible:
                frame = item["frame"]
                windows[max(0, frame - 50):min(len(labels), frame + 51)] = True
            if not np.any(windows):
                raise ValueError("edge-dropout scenario has no endpoint windows")
            scenarios["gap_{}_drop_{}".format(length, width)] = {
                "eligible_endpoints": len(eligible),
                "newly_hidden_observed_frames": int(np.count_nonzero(hidden)),
                "changed_frames_global": int(np.count_nonzero(changed)),
                "changed_frames_near_endpoints": int(
                    np.count_nonzero(changed & windows)
                ),
                "frames_in_endpoint_windows": int(np.count_nonzero(windows)),
                "fraction_near_endpoints_changed": float(
                    np.mean(changed[windows])
                ),
            }
    return {
        "schema_version": 1,
        "scope": (
            "Challenge tracking/tagging input-only: compare the working "
            "model with itself under additional masked long-gap endpoints."
        ),
        "config_sha256": file_sha256(config_path),
        "homography_sha256": file_sha256(homography_path),
        "input_profile": input_profile,
        "gap_minimums_frames": [269, 938],
        "masked_widths_frames": [5, 25],
        "scenarios": scenarios,
        "decision": "diagnostic_only; no possession correctness or release claim",
        "limitations": [
            "Agreement with the clean prediction is not possession accuracy.",
            "Only homography-qualified same-half observed endpoints are perturbed.",
            "This tests tracking-only inference, not experimental video fusion.",
        ],
    }


def sensitivity_configs(config):
    variants = []
    if config["decoder_mode"] == "state_machine":
        for factor in (0.9, 1.1):
            variant = dict(config)
            variant["transition_confirm_frames"] = max(
                1, int(round(config["transition_confirm_frames"] * factor))
            )
            variant["strong_transition_confirm_frames"] = max(
                1,
                int(round(
                    config["strong_transition_confirm_frames"] * factor
                )),
            )
            variants.append((
                "decoder_{:.0f}pct".format(factor * 100),
                variant,
            ))
    elif config["decoder_mode"] == "semi_markov":
        for factor in (0.9, 1.1):
            variant = dict(config)
            variant["semi_markov_minimum_live_frames"] = max(
                2,
                int(round(
                    config["semi_markov_minimum_live_frames"] * factor
                )),
            )
            variant["semi_markov_minimum_dead_frames"] = max(
                2,
                int(round(
                    config["semi_markov_minimum_dead_frames"] * factor
                )),
            )
            variants.append((
                "decoder_duration_{:.0f}pct".format(factor * 100),
                variant,
            ))
        for factor in (0.8, 1.2):
            variant = dict(config)
            variant["semi_markov_live_transition_cost"] = (
                config["semi_markov_live_transition_cost"] * factor
            )
            variant["semi_markov_strong_transition_cost"] = (
                config["semi_markov_strong_transition_cost"] * factor
            )
            variant["semi_markov_dead_transition_cost"] = (
                config["semi_markov_dead_transition_cost"] * factor
            )
            variants.append((
                "decoder_cost_{:.0f}pct".format(factor * 100),
                variant,
            ))
    else:
        for factor in (0.8, 1.2):
            variant = dict(config)
            variant["hsmm_live_duration_mode_frames"] = max(
                2,
                int(round(
                    config["hsmm_live_duration_mode_frames"] * factor
                )),
            )
            variant["hsmm_dead_duration_mode_frames"] = max(
                2,
                int(round(
                    config["hsmm_dead_duration_mode_frames"] * factor
                )),
            )
            variants.append((
                "hsmm_duration_mode_{:.0f}pct".format(factor * 100),
                variant,
            ))
        for factor in (0.8, 1.2):
            variant = dict(config)
            variant["hsmm_duration_weight"] = (
                config["hsmm_duration_weight"] * factor
            )
            variants.append((
                "hsmm_duration_weight_{:.0f}pct".format(factor * 100),
                variant,
            ))
        for factor in (0.8, 1.2):
            variant = dict(config)
            variant["hsmm_boundary_evidence_weight"] = (
                config["hsmm_boundary_evidence_weight"] * factor
            )
            variants.append((
                "hsmm_boundary_weight_{:.0f}pct".format(factor * 100),
                variant,
            ))
    if config["restart_localization_enabled"]:
        for factor in (0.8, 1.2):
            variant = dict(config)
            variant["restart_localization_search_radius_frames"] = max(
                1,
                int(round(
                    config["restart_localization_search_radius_frames"]
                    * factor
                )),
            )
            variants.append((
                "restart_radius_{:.0f}pct".format(factor * 100),
                variant,
            ))
        for factor in (0.75, 1.25):
            variant = dict(config)
            variant[
                "restart_localization_min_post_speed_metres_per_second"
            ] = (
                config[
                    "restart_localization_min_post_speed_metres_per_second"
                ]
                * factor
            )
            variants.append((
                "restart_speed_{:.0f}pct".format(factor * 100),
                variant,
            ))
    return variants


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_ledger(path, report, policy, family, variant, decision, notes):
    perturbations = report["perturbations"]
    failed = [
        check["name"] for check in report["checks"]
        if not check["passed"]
    ]
    row = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "framework_version": policy["framework_version"],
        "family": family,
        "variant": variant,
        "decision": decision,
        "passed": str(report["passed"]).lower(),
        "checks_passed": sum(
            check["passed"] for check in report["checks"]
        ),
        "checks_total": len(report["checks"]),
        "failed_checks": "|".join(failed),
        "config_sha256": report["config_sha256"],
        "clean_label_sha256": report["clean_label_sha256"],
        "reference_label_sha256": report["reference_label_sha256"],
        "possessions": report["clean_spells"]["possessions"],
        "shorter_than_one_second": (
            report["clean_spells"]["shorter_than_one_second"]
        ),
        "strong_event_persistence": (
            report["strong_control"]["majority_persistence"]
        ),
        "restarts_localized": report["restarts"]["localized"],
        "restart_speed_gain": report["restarts"]["median_speed_gain"],
        "noise_0.10m_frame_agreement": (
            perturbations["noise_0.10m"]["frame_agreement"]
        ),
        "noise_0.10m_boundary_recall": (
            perturbations["noise_0.10m"]["boundary_recall"]
        ),
        "noise_0.25m_frame_agreement": (
            perturbations["noise_0.25m"]["frame_agreement"]
        ),
        "noise_0.25m_boundary_recall": (
            perturbations["noise_0.25m"]["boundary_recall"]
        ),
        "missing_25_frames_frame_agreement": (
            perturbations["missing_25_frames"]["frame_agreement"]
        ),
        "retain_50_percent_frame_agreement": (
            perturbations["retain_50_percent"]["frame_agreement"]
        ),
        "retain_50_percent_boundary_recall": (
            perturbations["retain_50_percent"]["boundary_recall"]
        ),
        "restart_owners_inferred": (
            report["restart_ownership"]["unknown_inferred"]
        ),
        "restart_owner_proxy_coverage": (
            report["restart_ownership"]["known_coverage"]
        ),
        "restart_owner_proxy_accuracy": (
            report["restart_ownership"]["known_accuracy"]
        ),
        "decoded_boundary_majority_persistence": (
            report["decoded_boundaries"]["majority_persistence"]
        ),
        "decoded_boundary_strong_persistence": (
            report["decoded_boundaries"]["strong_persistence"]
        ),
        "team_switches": (
            report["decoded_boundaries"]["team_switches"]
        ),
        "notes": notes,
    }
    ledger_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(ledger_dir, exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run label-free proxy validation and robustness checks. "
            "This never invokes the public evaluator."
        )
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument(
        "--challenge-edge-stress", action="store_true",
        help="separate input-only 269/938-frame endpoint-dropout diagnostic",
    )
    parser.add_argument("--record-family")
    parser.add_argument("--record-variant")
    parser.add_argument(
        "--decision",
        choices=["candidate", "promoted", "rejected", "retained"],
        default="candidate",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    args = parser.parse_args(argv)
    if bool(args.record_family) != bool(args.record_variant):
        parser.error(
            "--record-family and --record-variant must be supplied together"
        )
    if args.challenge_edge_stress and (
        args.record_family or args.skip_unit_tests
    ):
        parser.error(
            "--challenge-edge-stress cannot record an experiment or skip its separate checks"
        )

    config = load_config(args.config)
    with open(args.policy) as handle:
        policy = json.load(handle)
    context = prepare_context(args.data_dir, config)
    raw_ball_xy, _ = load_raw_ball(
        context.raw_path, context.n_frames, context.pitch
    )
    if args.challenge_edge_stress:
        homography_path = require(
            ["homography"], data_dir=args.data_dir
        )[0]
        report = challenge_shaped_edge_stress(
            raw_ball_xy, context, config, homography_path, args.config
        )
        report_dir = os.path.dirname(os.path.abspath(args.report))
        os.makedirs(report_dir, exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(report["scenarios"], indent=2, sort_keys=True))
        return 0
    checks = []

    def minimum(name, value, threshold):
        passed = value >= threshold
        checks.append({
            "name": name,
            "value": float(value),
            "minimum": float(threshold),
            "passed": passed,
        })
        print(
            "{} {:<52} {:.5f} >= {:.5f}".format(
                "PASS" if passed else "FAIL",
                name,
                value,
                threshold,
            )
        )

    def maximum(name, value, threshold):
        passed = value <= threshold
        checks.append({
            "name": name,
            "value": float(value),
            "maximum": float(threshold),
            "passed": passed,
        })
        print(
            "{} {:<52} {:.5f} <= {:.5f}".format(
                "PASS" if passed else "FAIL",
                name,
                value,
                threshold,
            )
        )

    if not args.skip_unit_tests:
        unit = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        checks.append({
            "name": "unit_tests",
            "passed": unit.returncode == 0,
        })
        print(
            "{} unit_tests".format(
                "PASS" if unit.returncode == 0 else "FAIL"
            )
        )
        if unit.returncode != 0:
            print(unit.stdout)
            print(unit.stderr, file=sys.stderr)

    print("Running clean, repeat and perturbed inference in parallel")
    # Independent full-pipeline runs: identical results, just not one after another.
    _SHARED.update(context=context, config=config)
    perturbations = make_perturbations(raw_ball_xy, policy)
    pool = ProcessPoolExecutor(max_workers=len(perturbations) + 1,
                               mp_context=multiprocessing.get_context("fork"))
    repeat_job = pool.submit(_infer_labels, raw_ball_xy)
    perturbation_jobs = [(name, pool.submit(_infer_full, raw)) for name, raw in perturbations]
    clean = infer(raw_ball_xy, context, config)
    clean_repeat = _labels(repeat_job.result())
    deterministic = np.array_equal(clean.labels, clean_repeat.labels)
    checks.append({"name": "deterministic_labels", "passed": deterministic})
    print(
        "{} deterministic_labels".format(
            "PASS" if deterministic else "FAIL"
        )
    )

    clean_spells = spell_metrics(clean.labels, context.fps)
    reference = load_reference(args.reference, context.n_frames)
    reference_spells = spell_metrics(reference, context.fps)
    maximum(
        "short_spell_increase_vs_reference",
        clean_spells["shorter_than_one_second"]
        - reference_spells["shorter_than_one_second"],
        policy["maximum_short_spell_increase"],
    )

    persistence = strong_event_persistence(clean)
    minimum(
        "strong_event_majority_persistence",
        persistence["majority_persistence"],
        policy["minimum_strong_event_persistence"],
    )
    restarts = restart_metrics(clean)
    if restarts["localized"]:
        minimum(
            "localized_restart_median_speed_gain",
            restarts["median_speed_gain"],
            policy["minimum_restart_speed_gain_metres_per_second"],
        )
    decoded_boundaries = decoded_boundary_persistence(clean)
    minimum(
        "decoded_boundary_majority_persistence",
        decoded_boundaries["majority_persistence"],
        policy["minimum_decoded_boundary_majority_persistence"],
    )
    minimum(
        "decoded_boundary_strong_persistence",
        decoded_boundaries["strong_persistence"],
        policy["minimum_decoded_boundary_strong_persistence"],
    )
    game = GAMES[GAME]
    half_split_frame = (
        game["halves"][2][0] - game["offsets"][2]
    )
    localized = clean.tags.restart_localized
    restart_frames = (
        clean.tags.restart_frames[localized]
        if localized is not None
        else np.asarray([], dtype=np.int64)
    )
    cohorts = boundary_cohorts(
        clean.labels,
        restart_frames,
        half_split_frame,
        int(policy["restart_cohort_radius_frames"]),
    )
    cohort_persistence = decoded_boundary_cohort_persistence(
        decoded_boundaries, cohorts
    )
    minimum_cohort_events = int(policy["minimum_boundary_cohort_events"])
    for name, metrics in cohort_persistence.items():
        if metrics["proxy_events"] < minimum_cohort_events:
            continue
        minimum(
            "{} decoded majority persistence".format(name),
            metrics["majority_persistence"],
            policy["minimum_cohort_decoded_boundary_majority_persistence"],
        )
        minimum(
            "{} decoded strong persistence".format(name),
            metrics["strong_persistence"],
            policy["minimum_cohort_decoded_boundary_strong_persistence"],
        )
    decoded_boundaries.pop("_switches")
    decoded_boundaries.pop("_persistence")
    reference_switches = team_switches(reference)
    switch_ratio = (
        decoded_boundaries["team_switches"] / float(len(reference_switches))
        if len(reference_switches) else 1.0
    )
    minimum(
        "team_switch_ratio_vs_reference",
        switch_ratio,
        policy["minimum_team_switch_ratio_vs_reference"],
    )
    maximum(
        "team_switch_ratio_vs_reference",
        switch_ratio,
        policy["maximum_team_switch_ratio_vs_reference"],
    )
    possession_ratio = (
        clean_spells["possessions"]
        / float(reference_spells["possessions"])
        if reference_spells["possessions"] else 1.0
    )
    minimum(
        "possession_ratio_vs_reference",
        possession_ratio,
        policy["minimum_possession_ratio_vs_reference"],
    )
    maximum(
        "possession_ratio_vs_reference",
        possession_ratio,
        policy["maximum_possession_ratio_vs_reference"],
    )
    dead_share_difference = abs(
        float(np.mean(clean.labels == 2))
        - float(np.mean(reference == 2))
    )
    maximum(
        "dead_share_difference_vs_reference",
        dead_share_difference,
        policy["maximum_dead_share_difference_vs_reference"],
    )
    ownership = restart_ownership_metrics(clean, config)
    if config["restart_ownership_enabled"]:
        minimum(
            "restart_owner_proxy_coverage",
            ownership["known_coverage"],
            policy["minimum_restart_owner_proxy_coverage"],
        )
        minimum(
            "restart_owner_proxy_accuracy",
            ownership["known_accuracy"],
            policy["minimum_restart_owner_proxy_accuracy"],
        )

    tolerance = int(policy["boundary_tolerance_frames"])
    clean_switches = team_switches(clean.labels)
    perturbation_results = {}
    for name, job in perturbation_jobs:
        print("Running {}".format(name))
        result = job.result()
        switches = team_switches(result.labels)
        metrics = {
            "frame_agreement": float(np.mean(
                result.labels == clean.labels
            )),
            "boundary_recall": boundary_recall(
                clean_switches, switches, tolerance
            ),
            "reverse_boundary_recall": boundary_recall(
                switches, clean_switches, tolerance
            ),
            "spells": spell_metrics(result.labels, context.fps),
            "restart_stability": restart_stability(clean, result),
            "restart_ownership_stability": (
                restart_ownership_stability(clean, result)
            ),
            "boundary_cohorts": boundary_cohort_recall(
                clean.labels,
                result.labels,
                cohorts,
                tolerance,
            ),
        }
        perturbation_results[name] = metrics
        thresholds = policy["perturbations"][name]
        minimum(
            "{} frame agreement".format(name),
            metrics["frame_agreement"],
            thresholds["minimum_frame_agreement"],
        )
        minimum(
            "{} boundary recall".format(name),
            metrics["boundary_recall"],
            thresholds["minimum_boundary_recall"],
        )
        for cohort_name, cohort_metrics in (
            metrics["boundary_cohorts"].items()
        ):
            if cohort_metrics["events"] < minimum_cohort_events:
                continue
            minimum(
                "{} {} directional boundary recall".format(
                    name, cohort_name
                ),
                cohort_metrics["directional_recall"],
                thresholds["minimum_boundary_recall"],
            )
        if metrics["restart_stability"] is not None:
            minimum(
                "{} restart status".format(name),
                metrics["restart_stability"]["status_agreement"],
                thresholds["minimum_restart_status_agreement"],
            )
            minimum(
                "{} restart within ten".format(name),
                metrics["restart_stability"]["within_ten_frames"],
                thresholds["minimum_restart_within_ten_frames"],
            )
        if metrics["restart_ownership_stability"] is not None:
            minimum(
                "{} ownership status".format(name),
                metrics["restart_ownership_stability"][
                    "status_agreement"
                ],
                thresholds["minimum_ownership_status_agreement"],
            )
            minimum(
                "{} clean owner retention".format(name),
                metrics["restart_ownership_stability"][
                    "clean_owner_retention"
                ],
                thresholds["minimum_clean_owner_retention"],
            )

    sensitivity_results = {}
    pool.shutdown()
    _SHARED["clean"] = clean
    variants = list(sensitivity_configs(config))
    with ProcessPoolExecutor(max_workers=max(1, len(variants)),
                             mp_context=multiprocessing.get_context("fork")) as sensitivity_pool:
        sensitivity_labels = list(sensitivity_pool.map(_redecode_labels, [v for _, v in variants]))
    for (name, variant), labels in zip(variants, sensitivity_labels):
        result = _labels(labels)
        switches = team_switches(result.labels)
        metrics = {
            "frame_agreement": float(np.mean(
                result.labels == clean.labels
            )),
            "boundary_recall": boundary_recall(
                clean_switches, switches, tolerance
            ),
            "boundary_cohorts": boundary_cohort_recall(
                clean.labels,
                result.labels,
                cohorts,
                tolerance,
            ),
        }
        sensitivity_results[name] = metrics
        minimum(
            "{} frame agreement".format(name),
            metrics["frame_agreement"],
            policy["minimum_sensitivity_frame_agreement"],
        )
        minimum(
            "{} boundary recall".format(name),
            metrics["boundary_recall"],
            policy["minimum_sensitivity_boundary_recall"],
        )
        for cohort_name, cohort_metrics in (
            metrics["boundary_cohorts"].items()
        ):
            if cohort_metrics["events"] < minimum_cohort_events:
                continue
            minimum(
                "{} {} directional boundary recall".format(
                    name, cohort_name
                ),
                cohort_metrics["directional_recall"],
                policy["minimum_sensitivity_boundary_recall"],
            )

    report = {
        "framework_version": policy["framework_version"],
        "config": os.path.abspath(args.config),
        "config_sha256": file_sha256(args.config),
        "reference": os.path.abspath(args.reference),
        "reference_label_sha256": hashlib.sha256(
            reference.tobytes()
        ).hexdigest(),
        "clean_label_sha256": hashlib.sha256(
            clean.labels.tobytes()
        ).hexdigest(),
        "clean_spells": clean_spells,
        "reference_spells": reference_spells,
        "strong_control": persistence,
        "decoded_boundaries": decoded_boundaries,
        "boundary_cohorts": {
            "half_split_frame": int(half_split_frame),
            "restart_radius_frames": int(
                policy["restart_cohort_radius_frames"]
            ),
            "clean": cohort_persistence,
        },
        "restarts": restarts,
        "restart_ownership": ownership,
        "perturbations": perturbation_results,
        "sensitivity": sensitivity_results,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
    report_dir = os.path.dirname(os.path.abspath(args.report))
    os.makedirs(report_dir, exist_ok=True)
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if args.record_family:
        append_ledger(
            args.ledger,
            report,
            policy,
            args.record_family,
            args.record_variant,
            args.decision,
            args.notes,
        )
        print("Ledger: {}".format(os.path.abspath(args.ledger)))
    print("Report: {}".format(os.path.abspath(args.report)))
    print(
        "{} ({} checks)".format(
            "PASS" if report["passed"] else "FAIL",
            len(checks),
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
