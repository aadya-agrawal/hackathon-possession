import argparse
import json
import os
import sys
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np

from games import GAME, GAMES, half_video_blocks, require
from possession.decoder import (
    Evidence, build_evidence, decode, validate_local_handover_config,
    validate_tracking_stoppage_config,
    validate_gap_handover_config,
    validate_core_reliability_config,
    validate_touch_motion_config, touch_motion_summary,
    validate_owner_flight_config,
    validate_contact_carry_config,
    validate_observed_handover_config,
    validate_player_reliability_config,
    validate_control_phase_config,
    validate_tracking_exit_bracket_config,
    validate_restart_dead_config,
)
from possession.broadcast import (
    align_broadcast, broadcast_ball_observations, load_player_array, validate_broadcast_config,
)
from possession.ballfree import configured_home_probability, validate_ballfree_config
from possession.submission import summarize, validate_submission_file, write_submission
from possession.tagging import (
    infer_restart_ownership,
    load_tag_features,
    localize_restarts,
)
from possession.tracking import (
    extract_control_features,
    goalkeeper_frame_teams,
    load_raw_ball,
    load_tracking_metadata,
)
from possession.trajectory import (
    fuse_video_ball_observations,
    load_video_ball_observations,
    repair_ball_trajectory,
)
from possession.video import (
    audit_input_ball_anchor_video,
    detect_interval_coverage,
    detect_interval_paths,
    inspect_video,
    load_homography_matrices,
    load_homography_quality,
    rank_counterfactual_boundary_influence,
    rank_recovery_intervals,
    select_balanced_long_gap_runs,
    select_input_ball_anchor_samples,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "submission_pre_kickoff.json")
DEFAULT_OUTPUT = os.path.join(
    ROOT, "submissions", "possession_compliant.csv"
)


def load_config(path):
    with open(path) as handle:
        config = json.load(handle)
    required = {
        "max_interpolation_gap_frames",
        "max_ball_speed_metres_per_second",
        "prediction_history_frames",
        "prediction_horizon_frames",
        "max_ball_acceleration_metres_per_second_squared",
        "prediction_pitch_margin_metres",
        "distance_thresholds_metres",
        "nearest_player_count",
        "stationary_speed_threshold_metres_per_second",
        "max_player_speed_metres_per_second",
        "restart_offset_frames",
        "restart_prior_frames",
        "dead_lookback_frames",
        "restart_localization_enabled",
        "restart_localization_search_radius_frames",
        "restart_localization_motion_window_frames",
        "restart_localization_min_post_speed_metres_per_second",
        "restart_localization_max_pre_speed_metres_per_second",
        "restart_localization_min_speed_gain_metres_per_second",
        "restart_localization_min_pre_observed_frames",
        "restart_localization_max_reappearance_pre_observed_frames",
        "restart_localization_min_post_reliable_frames",
        "restart_ownership_enabled",
        "restart_ownership_eligible_types",
        "restart_ownership_window_frames_by_type",
        "restart_ownership_default_window_frames",
        "restart_ownership_max_distance_metres",
        "restart_ownership_distance_margin_metres",
        "restart_ownership_minimum_votes",
        "restart_ownership_minimum_winner_fraction",
        "restart_ownership_phase_window_frames",
        "restart_ownership_minimum_phase_prior",
        "restart_ownership_prior_frames_by_type",
        "restart_ownership_default_prior_frames",
        "closeness_scale_metres",
        "nearby_two_weight",
        "nearby_four_weight",
        "tag_weight",
        "restart_weight",
        "set_piece_dead_weight",
        "low_player_dead_weight",
        "low_player_count",
        "missing_ball_dead_weight",
        "missing_ball_player_count",
        "stoppage_median_speed_metres_per_second",
        "stoppage_stationary_player_fraction",
        "stoppage_gap_tolerance_frames",
        "pitch_margin_metres",
        "outside_pitch_dead_weight",
        "minimum_control_score",
        "control_margin",
        "motion_coherence_scale_metres_per_second",
        "motion_coherence_radius_metres",
        "motion_coherence_weight",
        "strong_control_distance_metres",
        "strong_control_distance_margin_metres",
        "strong_control_motion_gate_enabled",
        "strong_control_relative_speed_metres_per_second",
        "dead_threshold",
        "decoder_mode",
        "semi_markov_minimum_live_frames",
        "semi_markov_minimum_dead_frames",
        "semi_markov_control_margin_weight",
        "semi_markov_live_dead_suppression_weight",
        "semi_markov_dead_control_weight",
        "semi_markov_dead_emission_threshold",
        "semi_markov_live_transition_cost",
        "semi_markov_strong_transition_cost",
        "semi_markov_strong_transition_window_frames",
        "semi_markov_strong_transition_minimum_frames",
        "semi_markov_dead_transition_cost",
        "semi_markov_start_dead_bonus",
        "hsmm_minimum_live_frames",
        "hsmm_minimum_dead_frames",
        "hsmm_maximum_segment_frames",
        "hsmm_boundary_search_radius_frames",
        "hsmm_live_duration_mode_frames",
        "hsmm_dead_duration_mode_frames",
        "hsmm_duration_log_sigma",
        "hsmm_duration_weight",
        "hsmm_boundary_evidence_window_frames",
        "hsmm_boundary_evidence_weight",
        "hsmm_boundary_evidence_maximum",
        "video_recovery_boundary_radius_frames",
        "video_recovery_minimum_missing_frames",
        "video_recovery_minimum_homography_valid_fraction",
        "video_recovery_maximum_median_reprojection_error",
        "video_recovery_context_frames",
        "video_recovery_maximum_intervals",
        "video_recovery_detector_confidence",
        "video_recovery_detector_class_id",
        "video_recovery_detector_image_size",
        "video_recovery_detector_tile_overlap_fraction",
        "video_recovery_detector_merge_radius_pixels",
        "video_recovery_detector_pitch_margin_normalized",
        "video_recovery_detector_maximum_intervals",
        "video_recovery_path_maximum_gap_frames",
        "video_recovery_path_maximum_speed_mps",
        "video_recovery_path_maximum_acceleration_mps2",
        "video_recovery_path_confidence_reward",
        "video_recovery_path_gap_penalty",
        "video_recovery_path_speed_weight",
        "video_recovery_path_acceleration_weight",
        "video_recovery_path_mode",
        "video_recovery_kalman_beam_width",
        "video_recovery_kalman_maximum_missing_frames",
        "video_recovery_kalman_process_acceleration_std_mps2",
        "video_recovery_kalman_measurement_noise_base_metres",
        "video_recovery_kalman_measurement_noise_minimum_metres",
        "video_recovery_kalman_measurement_noise_maximum_metres",
        "video_recovery_kalman_initial_velocity_std_mps",
        "video_recovery_kalman_innovation_gate",
        "video_recovery_kalman_missing_penalty",
        "video_recovery_kalman_confidence_reward",
        "video_ball_observations_path",
        "video_ball_fusion_source_codes",
        "video_influence_maximum_boundaries",
        "video_influence_window_frames",
        "video_influence_search_radius_frames",
        "video_influence_control_score",
        "video_influence_force_strong_control",
        "live_confirm_frames",
        "dead_confirm_frames",
        "transition_confirm_frames",
        "strong_transition_confirm_frames",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("{}: missing config keys {}".format(path, missing))
    if config["decoder_mode"] not in ("state_machine", "semi_markov", "hsmm"):
        raise ValueError(
            "decoder_mode must be state_machine, semi_markov, or hsmm"
        )
    if config["video_recovery_path_mode"] not in (
        "temporal", "kalman_rts", "edge_anchored"
    ):
        raise ValueError(
            "video_recovery_path_mode must be temporal, kalman_rts or edge_anchored"
        )
    thresholds = config["distance_thresholds_metres"]
    if len(thresholds) < 3 or thresholds != sorted(thresholds):
        raise ValueError("distance_thresholds_metres must contain three increasing values")
    provenance = config.get("compliance_provenance")
    if (
        isinstance(provenance, dict)
        and provenance.get("mode") == "external_calibration_tracking_only"
        and config["video_ball_observations_path"] is not None
    ):
        raise ValueError(
            "tracking-only compliance config cannot enable "
            "video_ball_observations_path"
        )
    validate_local_handover_config(config)
    validate_tracking_stoppage_config(config)
    validate_tracking_exit_bracket_config(config)
    validate_restart_dead_config(config)
    validate_gap_handover_config(config)
    validate_core_reliability_config(config)
    validate_touch_motion_config(config)
    validate_owner_flight_config(config)
    validate_observed_handover_config(config)
    validate_contact_carry_config(config)
    validate_broadcast_config(config)
    validate_ballfree_config(config)
    validate_player_reliability_config(config)
    validate_control_phase_config(config)
    if config.get("tag_team_prior_mode", "all") not in ("all", "exclude_defending", "signed"):
        raise ValueError("tag_team_prior_mode must be all, exclude_defending or signed")
    half_local = config.get("decode_halves_separately_enabled", False)
    if not isinstance(half_local, bool):
        raise ValueError("decode_halves_separately_enabled must be boolean")
    if half_local and (
        config["decoder_mode"] != "semi_markov"
        or config.get("semi_markov_phase_enabled", False)
        or config.get("semi_markov_owner_flight_enabled", False)
    ):
        raise ValueError("half-local decoding requires semi-Markov without cross-half owner phases")
    return config


def decode_half_local(evidence, tags, config, blocks):
    if (
        config.get("decoder_mode") != "semi_markov"
        or config.get("semi_markov_phase_enabled", False)
        or config.get("semi_markov_owner_flight_enabled", False)
    ):
        raise ValueError("half-local decoding requires semi-Markov without cross-half owner phases")
    n = len(evidence.dead)
    boundaries = [0]
    previous_end = 0
    for _, first, last in blocks:
        if not 0 <= previous_end <= first <= last < n:
            raise ValueError("half-local blocks must be ordered, disjoint, and inside the video")
        boundaries.extend((first, last + 1))
        previous_end = last + 1
    boundaries.append(n)
    if len(blocks) != 2:
        raise ValueError("half-local decoding requires two authoritative half blocks")
    labels = np.empty(n, dtype=np.int8)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop == start:
            continue
        arrays = {}
        for field in fields(Evidence):
            value = getattr(evidence, field.name)
            if isinstance(value, np.ndarray):
                if len(value) != n:
                    raise ValueError("half-local evidence array is not aligned to the video")
                arrays[field.name] = value[start:stop]
        labels[start:stop] = decode(replace(evidence, **arrays), tags, config)
    return labels


def broadcast_view_state(data_dir, tactical_raw_path, tactical_metadata, n_frames, pitch, config,
                         tactical_xy=None, with_alignment=False):
    # Broadcast tracking is aligned automatically from player formations; nothing is frame-specific.
    meta_path, raw_path, homography_path = require(
        ["atd_meta", "atd_raw", "homography"], data_dir=data_dir, source="broadcast",
    )
    metadata = load_tracking_metadata(meta_path)
    if tuple(metadata.team_names) != tuple(tactical_metadata.team_names):
        raise ValueError("broadcast and tactical tracking disagree on home/away teams")
    broadcast_xy = load_player_array(raw_path, metadata.team_of_track, pitch)
    _, valid = load_homography_quality(homography_path, len(broadcast_xy))
    if tactical_xy is None:
        tactical_xy = load_player_array(tactical_raw_path, tactical_metadata.team_of_track, pitch)
    if len(tactical_xy) != n_frames:
        raise ValueError("tactical tracking does not match the submission timeline")
    mapped, state, diagnostics = align_broadcast(
        tactical_xy, tactical_metadata.team_of_track, broadcast_xy, metadata.team_of_track, valid, config,
    )
    if with_alignment:
        broadcast_ball, _ = load_raw_ball(raw_path, len(broadcast_xy), pitch)
        return state, diagnostics, (mapped, broadcast_xy, broadcast_ball)
    return state, diagnostics


def _require_evidence(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "{} is missing: regenerate it from the tactical video with tools/ "
            "(see docs/REPRODUCIBILITY.md, section 4 'Video outputs')".format(path)
        )


def load_video_team(config, source):
    """Frames and teams seen at the ball in the video, restricted to frames without a tracked ball."""
    path = config.get("video_team_evidence_path")
    if path is None:
        return None
    if not isinstance(path, str):
        raise ValueError("video_team_evidence_path must be a path or null")
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    _require_evidence(path)
    with np.load(path) as evidence:
        frames = evidence["frame"].astype(np.int64)
        teams = evidence["team"].astype(np.int64)
    keep = (frames >= 0) & (frames < len(source))
    keep[keep] = source[frames[keep]] != 1
    return frames[keep], teams[keep]


def configured_frame_teams(players_xy, metadata, pitch, config):
    """Per-frame track teams with goalkeeper identity swaps corrected, or None when disabled."""
    enabled = config.get("goalkeeper_end_correction_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("goalkeeper_end_correction_enabled must be boolean")
    override_path = config.get("track_team_override_path")
    if not enabled:
        if override_path is None:
            return None
        teams = np.tile(metadata.team_of_track.astype(np.int8), (len(players_xy), 1))
        return _apply_track_team_override(teams, override_path)
    radius = config.get("goalkeeper_end_radius_metres")
    if not isinstance(radius, (int, float)) or isinstance(radius, bool) or not 0 < radius < pitch[0] / 2:
        raise ValueError("goalkeeper_end_radius_metres must be positive and under half the pitch")
    if players_xy is None:
        raise ValueError("goalkeeper end correction needs player positions")
    from possession.ballfree import attack_directions
    blocks = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    team = metadata.team_of_track
    direction = np.zeros(len(players_xy))
    within = np.zeros(len(players_xy), dtype=bool)
    for start, stop in blocks:
        within[start:stop] = True
    direction[within] = attack_directions(
        players_xy[:, team == 0], players_xy[:, team == 1], blocks, pitch)[within]
    teams = goalkeeper_frame_teams(players_xy, metadata, direction, pitch, float(radius))
    return teams if override_path is None else _apply_track_team_override(teams, override_path)


def _apply_track_team_override(teams, path):
    # Video-derived per-frame track teams (-1 keeps the tracking/metadata team).
    if not isinstance(path, str):
        raise ValueError("track_team_override_path must be a path or null")
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    _require_evidence(path)
    with np.load(path) as stored:
        override = stored["override"] if "override" in stored.files else stored[stored.files[0]]
    if override.shape != teams.shape or not np.isin(override, (-1, 0, 1)).all():
        raise ValueError("track team override must be -1/0/1 per frame and track")
    return np.where(override >= 0, override, teams).astype(np.int8)


def run_pipeline(
    data_dir,
    output_path,
    config_path,
    video_report_path=None,
    video_detection_report_path=None,
    video_model_path=None,
    video_device="cpu",
    video_path_report_path=None,
    video_influence_report_path=None,
    video_path_mode=None,
    video_class_id=None,
    video_confidence=None,
):
    if (
        video_path_mode is not None
        or video_class_id is not None
        or video_confidence is not None
    ) and not video_path_report_path:
        raise ValueError("video path overrides require --video-path-report")
    game = GAMES[GAME]
    n_frames = game["n_video_frames"]
    fps = game["fps"]
    pitch = game["pitch"]
    config = load_config(config_path)

    meta_path, raw_path, tagging_path = require(
        ["atd_meta", "atd_raw", "smart_tagging"], data_dir=data_dir
    )
    metadata = load_tracking_metadata(meta_path)
    print("teams: H={}  A={}".format(*metadata.team_names))

    print("loading ball observations")
    raw_ball_xy, _ = load_raw_ball(raw_path, n_frames, pitch)
    observed_count = int(np.count_nonzero(np.isfinite(raw_ball_xy).all(axis=1)))

    print("repairing short ball gaps")
    trajectory = repair_ball_trajectory(
        raw_ball_xy,
        fps,
        max_gap_frames=int(config["max_interpolation_gap_frames"]),
        max_speed=float(config["max_ball_speed_metres_per_second"]),
        prediction_history_frames=int(config["prediction_history_frames"]),
        prediction_horizon_frames=int(config["prediction_horizon_frames"]),
        max_acceleration=float(
            config["max_ball_acceleration_metres_per_second_squared"]
        ),
        pitch=pitch,
        pitch_margin=float(config["prediction_pitch_margin_metres"]),
    )
    video_observation_count = 0
    if config["video_ball_observations_path"]:
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
            fps=fps,
            max_speed=float(
                config["max_ball_speed_metres_per_second"]
            ),
            replace_source_codes=tuple(
                config["video_ball_fusion_source_codes"]
            ),
        )
        video_observation_count = int(
            np.count_nonzero(trajectory.source == 4)
        )
    broadcast_view = broadcast_report = alignment = players_xy = None
    if config.get("broadcast_stoppage_enabled", False) or config.get("semi_markov_ballfree_enabled", False):
        players_xy = load_player_array(raw_path, metadata.team_of_track, pitch)
    if config.get("broadcast_stoppage_enabled", False):
        print("aligning broadcast tracking to the tactical timeline")
        broadcast_view, broadcast_report, alignment = broadcast_view_state(
            data_dir, raw_path, metadata, n_frames, pitch, config, tactical_xy=players_xy,
            with_alignment=True,
        )
    broadcast_ball_count = 0
    if config.get("broadcast_ball_fusion_enabled", False):
        if broadcast_view is None:
            raise ValueError("broadcast ball fusion requires the broadcast alignment")
        frames, xy, confidence = broadcast_ball_observations(
            alignment[0], broadcast_view, alignment[2], alignment[1], trajectory.source, fps, config,
        )
        trajectory = fuse_video_ball_observations(
            trajectory, frames, xy, confidence, fps=fps,
            max_speed=float(config["max_ball_speed_metres_per_second"]), replace_source_codes=(0,),
        )
        broadcast_ball_count = int(len(frames))
    frame_teams = configured_frame_teams(players_xy, metadata, pitch, config)
    interpolated_count = int(np.count_nonzero(trajectory.source == 2))
    predicted_count = int(np.count_nonzero(trajectory.source == 3))

    print("extracting team-control features")
    control = extract_control_features(
        raw_path,
        metadata,
        trajectory.xy,
        pitch,
        distance_thresholds=tuple(config["distance_thresholds_metres"]),
        nearest_count=int(config["nearest_player_count"]),
        stationary_speed_threshold=float(
            config["stationary_speed_threshold_metres_per_second"]
        ),
        max_player_speed=float(config["max_player_speed_metres_per_second"]),
        fps=fps,
        player_reliability_resume_gap_frames=config.get("player_reliability_resume_gap_frames", 0),
        frame_teams=frame_teams,
    )

    print("loading tactical tags")
    tags = load_tag_features(
        tagging_path,
        metadata.team_names,
        n_frames,
        fps,
        restart_offset_frames=int(config["restart_offset_frames"]),
        restart_prior_frames=int(config["restart_prior_frames"]),
        dead_lookback_frames=int(config["dead_lookback_frames"]),
        team_prior_mode=config.get("tag_team_prior_mode", "all"),
    )
    tags = localize_restarts(tags, trajectory, fps, config)
    tags = infer_restart_ownership(tags, trajectory, control, config)

    print("decoding possession states")
    ballfree_probability = None
    if config.get("semi_markov_ballfree_enabled", False):
        print("estimating ball-free possession from player tracking")
        ballfree_probability = configured_home_probability(
            players_xy, metadata.team_of_track, pitch, fps,
            [(first, last + 1) for _, first, last in half_video_blocks(GAME)], config, trajectory=trajectory,
        )
    evidence = build_evidence(
        control, trajectory, tags, pitch, config, fps=fps,
        period_starts=[first for _, first, _ in half_video_blocks(GAME)],
        period_blocks=[(first, last + 1) for _, first, last in half_video_blocks(GAME)],
        broadcast_view=broadcast_view, ballfree_home_probability=ballfree_probability,
        video_team=load_video_team(config, trajectory.source),
    )
    labels = (
        decode_half_local(evidence, tags, config, half_video_blocks(GAME))
        if config.get("decode_halves_separately_enabled", False)
        else decode(evidence, tags, config)
    )
    write_submission(output_path, labels)
    validate_submission_file(output_path, n_frames)

    report = summarize(labels, fps)
    report.update({
        "frames": n_frames,
        "observed_ball_frames": observed_count,
        "interpolated_ball_frames": interpolated_count,
        "predicted_ball_frames": predicted_count,
        "video_ball_frames": video_observation_count,
        "remaining_missing_ball_frames": int(np.count_nonzero(trajectory.source == 0)),
        "set_piece_hint_frames": int(np.count_nonzero(tags.dead_hint)),
        "localized_restarts": int(np.count_nonzero(tags.restart_localized)),
        "inferred_restart_owners": int(
            np.count_nonzero(tags.restart_team_inferred)
        ),
        "output": os.path.abspath(output_path),
    })
    if config.get("semi_markov_touch_motion_gate_enabled", False):
        report["touch_motion"] = {
            **touch_motion_summary(evidence, config),
            "unknown_behavior": "retain_original_support",
            "mode": config.get("touch_motion_mode", "gate"),
        }
    if config.get("semi_markov_owner_flight_enabled", False):
        report["owner_flight"] = evidence.owner_flight_diagnostics
    if config.get("semi_markov_phase_enabled", False):
        report["control_phase"] = evidence.control_phase.diagnostics
    if config.get("tracking_exit_bracket_enabled", False):
        report["exit_bracket"] = evidence.exit_bracket_diagnostics
    if evidence.restart_dead_diagnostics is not None:
        report["restart_dead"] = evidence.restart_dead_diagnostics
    if broadcast_report is not None:
        report["broadcast"] = {**broadcast_report, **(evidence.broadcast_stoppage_diagnostics or {})}
        if config.get("broadcast_ball_fusion_enabled", False):
            report["broadcast"]["broadcast_ball_frames"] = broadcast_ball_count
    if evidence.contact_carry_diagnostics is not None:
        report["contact_carry"] = evidence.contact_carry_diagnostics
    if config.get("decode_halves_separately_enabled", False):
        report["half_local_decoding"] = True
    if config.get("tag_team_prior_mode", "all") != "all":
        report["tag_team_prior_mode"] = config["tag_team_prior_mode"]
    if config.get("player_reliability_enabled", False):
        report["player_reliability"] = evidence.player_reliability_diagnostics
    if video_report_path:
        video_path, homography_path = require(
            ["video", "homography"], data_dir=data_dir
        )
        metadata = inspect_video(video_path)
        if metadata.frames != n_frames:
            raise ValueError(
                "{}: expected {} video frames, found {}".format(
                    video_path, n_frames, metadata.frames
                )
            )
        if abs(metadata.fps - fps) > 1e-6:
            raise ValueError(
                "{}: expected {} fps, found {}".format(
                    video_path, fps, metadata.fps
                )
            )
        homography_error, homography_valid = load_homography_quality(
            homography_path, n_frames
        )
        intervals = rank_recovery_intervals(
            trajectory.source,
            labels,
            homography_error,
            homography_valid,
            boundary_radius=int(
                config["video_recovery_boundary_radius_frames"]
            ),
            minimum_missing_frames=int(
                config["video_recovery_minimum_missing_frames"]
            ),
            minimum_valid_fraction=float(
                config[
                    "video_recovery_minimum_homography_valid_fraction"
                ]
            ),
            maximum_median_error=float(
                config[
                    "video_recovery_maximum_median_reprojection_error"
                ]
            ),
            context_frames=int(config["video_recovery_context_frames"]),
            maximum_intervals=int(
                config["video_recovery_maximum_intervals"]
            ),
        )
        video_report = {
            "video": {
                "path": os.path.abspath(video_path),
                "frames": metadata.frames,
                "fps": metadata.fps,
                "width": metadata.width,
                "height": metadata.height,
            },
            "homography": {
                "path": os.path.abspath(homography_path),
                "valid_frames": int(np.count_nonzero(homography_valid)),
                "median_reprojection_error": float(np.nanmedian(
                    homography_error[homography_valid]
                )),
            },
            "candidate_count": len(intervals),
            "candidate_missing_frames": int(sum(
                interval.missing_frames for interval in intervals
            )),
            "candidates": [interval.to_dict() for interval in intervals],
        }
        report_dir = os.path.dirname(os.path.abspath(video_report_path))
        os.makedirs(report_dir, exist_ok=True)
        with open(video_report_path, "w") as handle:
            json.dump(video_report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        report["video_recovery_report"] = os.path.abspath(
            video_report_path
        )
        report["video_recovery_candidates"] = len(intervals)
        if video_influence_report_path:
            influence = rank_counterfactual_boundary_influence(
                evidence,
                tags,
                labels,
                trajectory.source,
                intervals,
                config,
                window_frames=config[
                    "video_influence_window_frames"
                ],
                search_radius=int(
                    config["video_influence_search_radius_frames"]
                ),
                control_score=float(
                    config["video_influence_control_score"]
                ),
                force_strong_control=bool(
                    config["video_influence_force_strong_control"]
                ),
                maximum_boundaries=int(
                    config["video_influence_maximum_boundaries"]
                ),
            )
            influence_report = {
                "control_score": config[
                    "video_influence_control_score"
                ],
                "force_strong_control": config[
                    "video_influence_force_strong_control"
                ],
                "search_radius_frames": config[
                    "video_influence_search_radius_frames"
                ],
                "window_frames": config[
                    "video_influence_window_frames"
                ],
                "boundaries": influence,
            }
            influence_dir = os.path.dirname(os.path.abspath(
                video_influence_report_path
            ))
            os.makedirs(influence_dir, exist_ok=True)
            with open(video_influence_report_path, "w") as handle:
                json.dump(
                    influence_report, handle, indent=2, sort_keys=True
                )
                handle.write("\n")
            report["video_influence_report"] = os.path.abspath(
                video_influence_report_path
            )
        if video_detection_report_path:
            if not video_model_path:
                raise ValueError(
                    "--video-model is required with "
                    "--video-detection-report"
                )
            if not os.path.isfile(video_model_path):
                raise ValueError(
                    "{}: video model not found".format(video_model_path)
                )
            matrices = load_homography_matrices(
                homography_path, n_frames
            )
            detector_intervals = [
                interval for interval in intervals
                if interval.boundary_distance_frames == 0
            ][:int(config["video_recovery_detector_maximum_intervals"])]
            detector_reports = detect_interval_coverage(
                video_path,
                video_model_path,
                detector_intervals,
                matrices,
                class_id=int(
                    config["video_recovery_detector_class_id"]
                ),
                confidence=float(
                    config["video_recovery_detector_confidence"]
                ),
                image_size=int(
                    config["video_recovery_detector_image_size"]
                ),
                device=video_device,
                overlap_fraction=float(
                    config[
                        "video_recovery_detector_tile_overlap_fraction"
                    ]
                ),
                merge_radius_pixels=float(
                    config[
                        "video_recovery_detector_merge_radius_pixels"
                    ]
                ),
                pitch_margin=float(
                    config[
                        "video_recovery_detector_pitch_margin_normalized"
                    ]
                ),
            )
            detection_report = {
                "model": os.path.abspath(video_model_path),
                "device": video_device,
                "confidence": config[
                    "video_recovery_detector_confidence"
                ],
                "class_id": config[
                    "video_recovery_detector_class_id"
                ],
                "image_size": config[
                    "video_recovery_detector_image_size"
                ],
                "tile_overlap_fraction": config[
                    "video_recovery_detector_tile_overlap_fraction"
                ],
                "intervals": detector_reports,
            }
            detection_dir = os.path.dirname(os.path.abspath(
                video_detection_report_path
            ))
            os.makedirs(detection_dir, exist_ok=True)
            with open(video_detection_report_path, "w") as handle:
                json.dump(
                    detection_report, handle, indent=2, sort_keys=True
                )
                handle.write("\n")
            report["video_detection_report"] = os.path.abspath(
                video_detection_report_path
            )
        if video_path_report_path:
            if not video_model_path:
                raise ValueError(
                    "--video-model is required with --video-path-report"
                )
            if not os.path.isfile(video_model_path):
                raise ValueError(
                    "{}: video model not found".format(video_model_path)
                )
            path_mode = (
                video_path_mode if video_path_mode is not None
                else config["video_recovery_path_mode"]
            )
            if path_mode not in ("temporal", "kalman_rts", "edge_anchored"):
                raise ValueError("unknown video path mode {}".format(path_mode))
            class_id = (
                int(video_class_id) if video_class_id is not None
                else int(config["video_recovery_detector_class_id"])
            )
            confidence = (
                float(video_confidence) if video_confidence is not None
                else float(config["video_recovery_detector_confidence"])
            )
            if class_id < 0 or not 0 < confidence <= 1:
                raise ValueError("invalid video class ID or confidence")
            matrices = load_homography_matrices(
                homography_path, n_frames
            )
            path_intervals = [
                interval for interval in intervals
                if interval.boundary_distance_frames == 0
            ][:int(config["video_recovery_detector_maximum_intervals"])]
            path_reports = detect_interval_paths(
                video_path,
                video_model_path,
                path_intervals,
                trajectory.source,
                trajectory.xy,
                matrices,
                fps=fps,
                pitch=pitch,
                class_id=class_id,
                confidence=confidence,
                image_size=int(
                    config["video_recovery_detector_image_size"]
                ),
                device=video_device,
                overlap_fraction=float(
                    config[
                        "video_recovery_detector_tile_overlap_fraction"
                    ]
                ),
                merge_radius_pixels=float(
                    config[
                        "video_recovery_detector_merge_radius_pixels"
                    ]
                ),
                pitch_margin=float(
                    config[
                        "video_recovery_detector_pitch_margin_normalized"
                    ]
                ),
                maximum_gap_frames=int(
                    config["video_recovery_path_maximum_gap_frames"]
                ),
                maximum_speed=float(
                    config["video_recovery_path_maximum_speed_mps"]
                ),
                maximum_acceleration=float(
                    config[
                        "video_recovery_path_maximum_acceleration_mps2"
                    ]
                ),
                confidence_reward=float(
                    config["video_recovery_path_confidence_reward"]
                ),
                gap_penalty=float(
                    config["video_recovery_path_gap_penalty"]
                ),
                speed_weight=float(
                    config["video_recovery_path_speed_weight"]
                ),
                acceleration_weight=float(
                    config["video_recovery_path_acceleration_weight"]
                ),
                path_mode=path_mode,
                kalman_beam_width=int(
                    config["video_recovery_kalman_beam_width"]
                ),
                kalman_maximum_missing_frames=int(
                    config[
                        "video_recovery_kalman_maximum_missing_frames"
                    ]
                ),
                kalman_process_acceleration_std=float(
                    config[
                        "video_recovery_kalman_process_acceleration_std_mps2"
                    ]
                ),
                kalman_measurement_noise_base=float(
                    config[
                        "video_recovery_kalman_measurement_noise_base_metres"
                    ]
                ),
                kalman_measurement_noise_minimum=float(
                    config[
                        "video_recovery_kalman_measurement_noise_minimum_metres"
                    ]
                ),
                kalman_measurement_noise_maximum=float(
                    config[
                        "video_recovery_kalman_measurement_noise_maximum_metres"
                    ]
                ),
                kalman_initial_velocity_std=float(
                    config[
                        "video_recovery_kalman_initial_velocity_std_mps"
                    ]
                ),
                kalman_innovation_gate=float(
                    config["video_recovery_kalman_innovation_gate"]
                ),
                kalman_missing_penalty=float(
                    config["video_recovery_kalman_missing_penalty"]
                ),
                kalman_confidence_reward=float(
                    config["video_recovery_kalman_confidence_reward"]
                ),
            )
            path_report = {
                "model": os.path.abspath(video_model_path),
                "device": video_device,
                "confidence": confidence,
                "class_id": class_id,
                "path_mode": path_mode,
                "runs": path_reports,
            }
            path_dir = os.path.dirname(os.path.abspath(
                video_path_report_path
            ))
            os.makedirs(path_dir, exist_ok=True)
            with open(video_path_report_path, "w") as handle:
                json.dump(path_report, handle, indent=2, sort_keys=True)
                handle.write("\n")
            report["video_path_report"] = os.path.abspath(
                video_path_report_path
            )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def run_ball_anchor_input_audit(
    data_dir, config_path, output_path, model_path, device,
):
    game = GAMES[GAME]
    n_frames = game["n_video_frames"]
    config = load_config(config_path)
    raw_path, homography_path, video_path = require(
        ["atd_raw", "homography", "video"], data_dir=data_dir
    )
    raw_ball_xy, _ = load_raw_ball(raw_path, n_frames, game["pitch"])
    homography_error, homography_valid = load_homography_quality(
        homography_path, n_frames
    )
    samples, tracking = select_input_ball_anchor_samples(
        raw_ball_xy, homography_error, homography_valid,
        half_video_blocks(GAME),
        fps=game["fps"],
        minimum_gap_frames=50,
        maximum_homography_error=float(
            config["video_recovery_maximum_median_reprojection_error"]
        ),
        maximum_ball_speed=float(
            config["max_ball_speed_metres_per_second"]
        ),
        stable_sample_count=120,
    )
    matrices = load_homography_matrices(homography_path, n_frames)
    video = audit_input_ball_anchor_video(
        video_path, model_path, samples, raw_ball_xy, matrices,
        game["pitch"], fps=game["fps"], device=device,
    )
    report = {
        "schema_version": 1,
        "scope": "input-only; no possession labels, predictions or manual video review",
        "model": os.path.abspath(model_path),
        "tracking": tracking,
        "video": video,
        "limitations": [
            "Detector agreement is not ball ground truth or absolute tracking error.",
            "The nearest detection can be a distractor; missing detections do not prove a bad anchor.",
            "Airborne balls and projective video geometry can shift pixel/ground alignment.",
            "Motion support is an input consistency test, not independently verified ball identity.",
        ],
    }
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def run_ball_anchor_path_audit(
    data_dir, config_path, output_path, model_path, device,
):
    game = GAMES[GAME]
    n_frames = game["n_video_frames"]
    config = load_config(config_path)
    raw_path, homography_path, video_path = require(
        ["atd_raw", "homography", "video"], data_dir=data_dir
    )
    raw_ball_xy, _ = load_raw_ball(raw_path, n_frames, game["pitch"])
    error, valid = load_homography_quality(homography_path, n_frames)
    samples, input_profile = select_input_ball_anchor_samples(
        raw_ball_xy, error, valid, half_video_blocks(GAME),
        fps=game["fps"], minimum_gap_frames=50,
        maximum_homography_error=float(
            config["video_recovery_maximum_median_reprojection_error"]
        ),
        maximum_ball_speed=float(
            config["max_ball_speed_metres_per_second"]
        ),
        stable_sample_count=0,
    )
    selected = select_balanced_long_gap_runs(samples)
    source = np.isfinite(raw_ball_xy).all(axis=1).astype(np.int8)
    matrices = load_homography_matrices(homography_path, n_frames)
    intervals = [
        SimpleNamespace(boundary_frame=item["start_frame"])
        for item in selected
    ]
    reports = detect_interval_paths(
        video_path, model_path, intervals, source, raw_ball_xy, matrices,
        fps=game["fps"], pitch=game["pitch"], class_id=32,
        confidence=0.03, image_size=1280, device=device,
        overlap_fraction=0.1, merge_radius_pixels=20,
        pitch_margin=0.02, maximum_gap_frames=5,
        maximum_speed=float(config["max_ball_speed_metres_per_second"]),
        maximum_acceleration=float(
            config["video_recovery_path_maximum_acceleration_mps2"]
        ),
        confidence_reward=2, gap_penalty=1,
        speed_weight=0.25, acceleration_weight=0.5,
        path_mode="edge_anchored",
        kalman_beam_width=24, kalman_maximum_missing_frames=5,
        kalman_process_acceleration_std=15,
        kalman_measurement_noise_base=0.5,
        kalman_measurement_noise_minimum=0.1,
        kalman_measurement_noise_maximum=2,
        kalman_initial_velocity_std=30,
        kalman_innovation_gate=4,
        kalman_missing_penalty=3,
        kalman_confidence_reward=2,
        require_generic_coco=True,
    )
    if len(reports) != len(selected):
        raise ValueError("input-selected long gaps were silently skipped")
    cohorts = {}
    for item, path in zip(selected, reports):
        name = "half_{}_gap_{}".format(item["half"], item["band"])
        group = cohorts.setdefault(name, {
            "runs": 0, "missing_frames": 0,
            "edge_frames_with_candidates": 0,
            "selected_observation_frames": 0,
            "tracklets": 0, "runs_with_any_path": 0,
            "runs_with_both_sides": 0,
            "_position_std": [],
        })
        group["runs"] += 1
        group["missing_frames"] += path["frames"]
        group["edge_frames_with_candidates"] += len(path["candidate_frames"])
        group["selected_observation_frames"] += path["selected_frames"]
        group["tracklets"] += len(path["edge_tracklets"])
        group["runs_with_any_path"] += int(bool(path["selected_frames"]))
        group["runs_with_both_sides"] += int(
            len(path["edge_tracklets"]) == 2
        )
        if path["maximum_position_std"] is not None:
            group["_position_std"].append(path["maximum_position_std"])
    for group in cohorts.values():
        std = group.pop("_position_std")
        group["maximum_position_std_p90_metres_when_selected"] = (
            float(np.percentile(std, 90)) if std else None
        )
    report = {
        "schema_version": 1,
        "scope": (
            "Input-only automatic video paths on balanced long raw-ball gaps; "
            "no possession output, protected truth or manual video review."
        ),
        "model": os.path.abspath(model_path),
        "input_profile": input_profile,
        "sampled_runs": len(selected),
        "sampling": (
            "Four evenly spaced paired-anchor gaps per half and "
            "269-937/938+ frame duration cohort."
        ),
        "cohorts": cohorts,
        "limitations": [
            "Candidate activation and Kalman uncertainty are not ball ground truth.",
            "Only the first and last 25 frames have detections; the center stays missing.",
            "Human-ball validation used cleaner anchors and has no possession references.",
        ],
    }
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a possession submission from tactical tracking and Smart Tagging."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="data bundle root; defaults to MTAH_DATA_DIR or ./data",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="JSON configuration file",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="output frame,label CSV",
    )
    parser.add_argument(
        "--video-report",
        help=(
            "optional JSON report ranking missing-ball intervals near "
            "possession changes; does not alter labels"
        ),
    )
    parser.add_argument(
        "--video-detection-report",
        help=(
            "optional JSON report for local tiled ball detection; "
            "requires --video-report and --video-model"
        ),
    )
    parser.add_argument(
        "--video-path-report",
        help=(
            "optional detector-path diagnostic near missing-ball runs; "
            "requires --video-report and --video-model; does not alter labels"
        ),
    )
    parser.add_argument(
        "--video-path-mode",
        choices=("temporal", "kalman_rts", "edge_anchored"),
        help="optional detector-path mode override for --video-path-report",
    )
    parser.add_argument(
        "--video-class-id", type=int,
        help="optional detector class override for --video-path-report",
    )
    parser.add_argument(
        "--video-confidence", type=float,
        help="optional detector confidence override for --video-path-report",
    )
    parser.add_argument(
        "--video-influence-report",
        help=(
            "optional JSON counterfactual boundary-influence report; "
            "requires --video-report"
        ),
    )
    parser.add_argument(
        "--ball-anchor-audit-report",
        help=(
            "input-only JSON summary of tracking gap endpoints and automatic "
            "video-detection agreement; requires --video-model; does not decode labels"
        ),
    )
    parser.add_argument(
        "--ball-anchor-path-audit-report",
        help=(
            "input-only balanced long-gap edge-path activation report; "
            "requires a generic COCO --video-model; does not decode labels"
        ),
    )
    parser.add_argument(
        "--video-model",
        help="path to public local detector weights",
    )
    parser.add_argument(
        "--video-device",
        default="cpu",
        help="Ultralytics inference device, such as cpu or 0",
    )
    args = parser.parse_args(argv)
    if args.video_detection_report and not args.video_report:
        parser.error(
            "--video-detection-report requires --video-report"
        )
    if args.video_path_report and not args.video_report:
        parser.error("--video-path-report requires --video-report")
    if (
        args.video_path_mode is not None
        or args.video_class_id is not None
        or args.video_confidence is not None
    ) and not args.video_path_report:
        parser.error("video path overrides require --video-path-report")
    if args.video_influence_report and not args.video_report:
        parser.error("--video-influence-report requires --video-report")
    if args.ball_anchor_audit_report or args.ball_anchor_path_audit_report:
        if not args.video_model:
            parser.error("ball-anchor audits require --video-model")
        if (
            args.video_report or args.video_detection_report
            or args.video_path_report or args.video_influence_report
            or (
                args.ball_anchor_audit_report
                and args.ball_anchor_path_audit_report
            )
        ):
            parser.error("ball-anchor audits cannot be combined with other video diagnostics")
        try:
            if args.ball_anchor_path_audit_report:
                report = run_ball_anchor_path_audit(
                    args.data_dir, args.config,
                    args.ball_anchor_path_audit_report,
                    args.video_model, args.video_device,
                )
            else:
                report = run_ball_anchor_input_audit(
                    args.data_dir, args.config,
                    args.ball_anchor_audit_report,
                    args.video_model, args.video_device,
                )
        except (IOError, OSError, RuntimeError, ValueError, KeyError) as error:
            parser.exit(1, "error: {}\n".format(error))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    try:
        run_pipeline(
            args.data_dir,
            args.output,
            args.config,
            args.video_report,
            args.video_detection_report,
            args.video_model,
            args.video_device,
            args.video_path_report,
            args.video_influence_report,
            args.video_path_mode,
            args.video_class_id,
            args.video_confidence,
        )
    except (IOError, OSError, RuntimeError, ValueError, KeyError) as error:
        parser.exit(1, "error: {}\n".format(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
