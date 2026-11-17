import json
from dataclasses import replace
import hashlib
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from possession import AWAY, DEAD, HOME, UNKNOWN
from possession.cli import DEFAULT_CONFIG, DEFAULT_OUTPUT, decode_half_local, load_config, run_pipeline
from possession.decoder import (
    Evidence,
    _hsmm_duration_scores,
    _restart_stoppage_hint,
    _semi_markov_emissions,
    _semi_markov_strong_transition_support,
    _semi_markov_emissions,
    CORE_CALIBRATION_KEYS,
    _tracking_stoppage_hint,
    tracking_exit_brackets,
    validate_tracking_exit_bracket_config,
    kickoff_dead_intervals,
    pre_kickoff_forced_dead,
    restart_anchored_dead_intervals,
    validate_restart_dead_config,
    _anchor_teams,
    _corroborated_stoppage,
    _gap_handover_evidence,
    build_evidence,
    decode,
    decode_hsmm,
    decode_semi_markov,
    owner_flight_evidence,
    validate_owner_flight_config,
    validate_observed_handover_config,
    contact_carry_evidence,
    validate_contact_carry_config,
    validate_control_phase_config,
    validate_player_reliability_config,
    touch_motion_summary, validate_touch_motion_config,
)
from possession.external_soccernet import (
    build_soccernet_gamestate_manifest,
    load_gamestate_ball_annotations,
    load_soccernet_config,
)
from possession.external_soccernet_metrics import (
    _anchor_perturbations,
    benchmark_soccernet_anchored_gaps,
    build_soccernet_geometry_metrics,
    score_annotated_ball_candidates,
)
from possession.external_parameter_selection import (
    build_compliance_outputs,
    select_external_parameters,
)
from possession.external_metrica_validation import (
    _candidate_labels,
    _decoded_event_metrics,
    _event_team_metrics,
    _gap_metrics,
    _load_epts_game,
    _load_events,
    _paired_event_delta,
    _paired_gap_delta,
    _scoring_events,
    _select_windows,
    _to_metres,
    config_at_fps,
    load_validation_protocol,
)
from possession.external_temporal_parity import (
    gap_time_correction, load_frozen_diagnostic, select_owner_flight,
    select_player_reliability,
    support_horizon_repair_values, temporal_settings,
)
from possession.control_phase import (
    CONTACT, FLIGHT, RECEPTION, STOPPAGE, TURNOVER, UNSEEN, ControlPhase,
    infer_control_phase, sustained_player_contacts,
)
from possession.external_possession_events import (
    possession_chain_references,
    recovery_masks,
    recovery_references,
    score_censored_dead_frames,
    score_possession_references,
    score_recovery_transitions,
)
from possession.external_handover_calibration import (
    _evaluate as evaluate_handover,
    handover_candidates,
    load_handover_protocol,
    stoppage_candidates,
)
from possession.external_transfer_check import (
    gap_release_spell_summary,
    independent_reference_mask,
    live_missing_stress,
    load_frozen_transfer_protocol,
    restart_only_tags,
)
from possession.release_check import _exit_candidate_changes, _plan_evidence, release_decision
from possession.external_core_calibration import choose_core_candidate, load_core_protocol
from possession.external_reliability_calibration import (
    load_reliability_protocol,
    reliability_overrides,
)
from possession.external_guarded_calibration import (
    evaluate_case,
    guarded_candidates,
    guarded_checks,
    load_guarded_protocol,
)
from possession.external_dfl_validation import (
    audit_dfl_spell_structure, load_dfl_match, score_dfl_labels,
)
from possession.external_prediction_reselection import choose_prediction_policy
from possession.external_decoder_reselection import choose_decoder_policy
from possession.submission import summarize, validate_submission_file, write_submission
from possession.tagging import (
    TagFeatures,
    infer_restart_ownership,
    load_tag_features,
    localize_restarts,
)
from possession.tracking import (
    TrackingMetadata,
    control_features_from_arrays,
    extract_control_features,
    load_player_positions_at_frames,
    load_raw_ball,
    load_tracking_metadata,
    player_velocity_steps,
)
from possession.trajectory import (
    BallTrajectory,
    _prune_prediction_jumps,
    fuse_video_ball_observations,
    repair_ball_trajectory,
)
from possession.video import (
    _project,
    _tile_regions,
    audit_input_ball_anchor_video,
    detect_interval_paths,
    detect_tiled_ball_candidates,
    detect_tiled_ball_pixels,
    load_homography_quality,
    make_long_gap_edge_dropout,
    rank_recovery_intervals,
    score_player_relative_candidates,
    select_anchored_kalman_tracklet,
    select_edge_anchored_ball_paths,
    select_balanced_long_gap_runs,
    select_input_ball_anchor_samples,
    select_kalman_ball_path,
    select_kalman_ball_tracklets,
    select_temporal_ball_path,
)


ROOT = Path(__file__).resolve().parents[1]


def default_config():
    with (ROOT / "configs" / "default.json").open() as handle:
        return json.load(handle)


def empty_tags(n_frames):
    return TagFeatures(
        team_prior=np.zeros(n_frames, dtype=np.float32),
        dead_hint=np.zeros(n_frames, dtype=bool),
        pre_restart_window=np.zeros(n_frames, dtype=bool),
        restart_team=np.full(n_frames, -1, dtype=np.int8),
    )




class ExternalSoccerNetTests(unittest.TestCase):
    def test_anchor_perturbations_are_bounded_paired_and_reproducible(self):
        first = list(_anchor_perturbations("SNGS-test", 2.0))
        second = list(_anchor_perturbations("SNGS-test", 2.0))
        self.assertEqual(len(first), 4)
        for observed, repeated in zip(first, second):
            np.testing.assert_array_equal(observed, repeated)
            np.testing.assert_allclose(
                np.linalg.norm(observed, axis=1), [2.0, 2.0]
            )
        self.assertFalse(np.allclose(first[0][0], first[0][1]))
        for radius in (-1.0, float("nan"), 6.0):
            with self.assertRaisesRegex(ValueError, "anchor noise radii"):
                benchmark_soccernet_anchored_gaps(
                    "unused.json", "unused.pt", split="development",
                    maximum_clips=1, confidence=0.03, image_size=1280,
                    device="cpu", gap_frames=50,
                    anchor_noise_metres=(radius,),
                )

    def test_detector_scoring_distinguishes_oracle_from_highest_confidence(self):
        score = score_annotated_ball_candidates(
            np.asarray([10.0, 10.0]),
            [
                {"pixel": np.asarray([11.0, 10.0]), "confidence": 0.2},
                {"pixel": np.asarray([90.0, 10.0]), "confidence": 0.9},
            ],
        )
        self.assertEqual(score["candidate_count"], 2)
        self.assertAlmostEqual(score["nearest_error_pixels"], 1.0)
        self.assertAlmostEqual(score["highest_confidence_error_pixels"], 80.0)
        self.assertIsNone(
            score_annotated_ball_candidates(
                np.asarray([10.0, 10.0]), []
            )["highest_confidence_error_pixels"]
        )

    def _write_clip(self, root, split, clip, version="1.3"):
        clip_root = root / split / clip
        clip_root.mkdir(parents=True)
        label_path = clip_root / "Labels-GameState.json"
        label_path.write_text(json.dumps({
            "info": {
                "version": version,
                "frame_rate": 25,
            },
            "images": [
                {
                    "image_id": "frame-a",
                    "file_name": "000001.jpg",
                    "width": 1920,
                    "height": 1080,
                },
                {
                    "image_id": "frame-b",
                    "file_name": "000002.jpg",
                    "width": 1920,
                    "height": 1080,
                },
            ],
            "annotations": [
                {
                    "image_id": "frame-a",
                    "track_id": 7,
                    "supercategory": "object",
                    "attributes": {
                        "role": "ball",
                        "team": None,
                        "jersey": None,
                    },
                    "bbox_image": {
                        "x_center": 100.5,
                        "y_center": 200.5,
                        "w": 8,
                        "h": 9,
                    },
                    "bbox_pitch": None,
                },
                {
                    "image_id": "frame-a",
                    "track_id": 2,
                    "supercategory": "object",
                    "attributes": {
                        "role": "player",
                        "team": "left",
                        "jersey": "10",
                    },
                    "bbox_image": {
                        "x_center": 400,
                        "y_center": 500,
                        "w": 30,
                        "h": 80,
                    },
                    "bbox_pitch": {
                        "x_bottom_left": 10,
                        "y_bottom_left": 20,
                    },
                },
                {
                    "image_id": "frame-a",
                    "supercategory": "pitch",
                    "lines": {},
                },
            ],
            "categories": [],
        }))
        return label_path

    def _write_config(self, root, eligible=None, overlap="unresolved"):
        path = root / "config.json"
        path.write_text(json.dumps({
            "minimum_dataset_version": "1.3",
            "split_mapping": {
                "development": ["train"],
                "validation": ["valid"],
                "test": ["test"],
            },
            "detector_evaluation": {
                "checkpoint": "ball.pt",
                "training_dataset": "unknown",
                "overlap_status": overlap,
                "eligible_source_splits": eligible or [],
            },
        }))
        return path

    def test_extracts_only_human_ball_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label_path = self._write_clip(root, "train", "001")

            labels = load_gamestate_ball_annotations(label_path)

            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0]["frame"], 0)
            self.assertEqual(labels[0]["pixel_x"], 100.5)
            self.assertEqual(
                labels[0]["source"],
                "soccernet_gamestate_human_ground_truth",
            )

    def test_builds_official_split_isolated_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_clip(root, "train", "001")
            self._write_clip(root, "valid", "002")
            self._write_clip(root, "test", "003")
            config_path = self._write_config(root)

            manifest = build_soccernet_gamestate_manifest(
                root, config_path
            )

            self.assertEqual(manifest["summary"]["clips"], 3)
            self.assertEqual(
                manifest["summary"]["splits"]["validation"][
                    "source_splits"
                ],
                ["valid"],
            )
            self.assertEqual(
                manifest["summary"]["ball_annotations"], 3
            )
            self.assertFalse(manifest["detector_evaluation"]["enabled"])
            self.assertFalse(
                manifest["split_policy"]["challenge_split_used"]
            )

    def test_rejects_old_dataset_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label_path = self._write_clip(
                root, "train", "001", version="1.2"
            )
            with self.assertRaisesRegex(ValueError, "older than required"):
                load_gamestate_ball_annotations(label_path)

    def test_marks_multiple_ball_boxes_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label_path = self._write_clip(root, "train", "001")
            data = json.loads(label_path.read_text())
            duplicate = dict(data["annotations"][0])
            duplicate["track_id"] = 99
            duplicate["bbox_image"] = dict(duplicate["bbox_image"])
            duplicate["bbox_image"]["x_center"] = 150
            data["annotations"].append(duplicate)
            label_path.write_text(json.dumps(data))

            labels = load_gamestate_ball_annotations(label_path)

            self.assertEqual(len(labels), 2)
            self.assertTrue(all(label["ambiguous"] for label in labels))

    def test_blocks_detector_evaluation_until_overlap_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_config(
                root, eligible=["valid"], overlap="unresolved"
            )
            with self.assertRaisesRegex(
                ValueError, "cannot be enabled"
            ):
                load_soccernet_config(config_path)

    def test_allows_absent_deferred_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_clip(root, "train", "001")
            self._write_clip(root, "valid", "002")
            config_path = self._write_config(root)
            config = json.loads(config_path.read_text())
            config["deferred_target_splits"] = ["test"]
            config_path.write_text(json.dumps(config))

            manifest = build_soccernet_gamestate_manifest(
                root, config_path
            )

            test_summary = manifest["summary"]["splits"]["test"]
            self.assertFalse(test_summary["available"])
            self.assertTrue(test_summary["deferred"])
            self.assertEqual(test_summary["clips"], 0)

    def test_builds_label_only_geometry_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_clip(root, "train", "001")
            self._write_clip(root, "valid", "002")
            config_path = self._write_config(root)
            config = json.loads(config_path.read_text())
            config["deferred_target_splits"] = ["test"]
            config_path.write_text(json.dumps(config))
            manifest = build_soccernet_gamestate_manifest(
                root, config_path
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            metrics = build_soccernet_geometry_metrics(manifest_path)

            development = metrics["splits"]["development"]
            self.assertEqual(development["frames"], 2)
            self.assertEqual(development["unambiguous_ball_frames"], 1)
            self.assertEqual(development["missing_ball_frames"], 1)
            self.assertEqual(
                development["distributions"][
                    "missing_runs_seconds"
                ]["p50"],
                0.04,
            )
            self.assertEqual(
                development["distributions"][
                    "nearest_player_distances_m"
                ]["count"],
                0,
            )
            self.assertFalse(metrics["policy"]["test_split_used"])
            self.assertFalse(metrics["policy"]["detector_outputs_used"])

    def test_selects_external_parameters_and_passes_validation(self):
        metrics = {
            "splits": {
                "development": {
                    "distributions": {
                        "ball_speeds_mps": {"p99": 56.7},
                        "ball_accelerations_mps2": {
                            "p75": 14.1,
                            "p90": 36.6,
                            "p99": 169.1,
                        },
                        "missing_runs_seconds": {"p75": 0.75},
                    },
                },
                "validation": {
                    "distributions": {
                        "ball_speeds_mps": {"p99": 60.5},
                        "ball_accelerations_mps2": {
                            "p75": 16.9,
                            "p90": 42.4,
                        },
                        "missing_runs_seconds": {"p75": 0.72},
                    },
                },
            },
        }

        selected, checks = select_external_parameters(metrics, 50)

        self.assertEqual(
            selected["max_ball_speed_metres_per_second"], 65.0
        )
        self.assertEqual(
            selected["max_interpolation_gap_frames"], 38
        )
        self.assertEqual(
            selected[
                "video_recovery_kalman_process_acceleration_std_mps2"
            ],
            15.0,
        )
        self.assertTrue(all(check["passed"] for check in checks))

    def test_compliance_config_disables_match_specific_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            config_path = root / "base.json"
            metrics_path.write_text(json.dumps({
                "policy": {
                    "test_split_used": False,
                    "detector_outputs_used": False,
                },
                "splits": {
                    "development": {
                        "distributions": {
                            "ball_speeds_mps": {"p99": 40},
                            "ball_accelerations_mps2": {
                                "p75": 10,
                                "p90": 20,
                                "p99": 100,
                            },
                            "missing_runs_seconds": {"p75": 0.5},
                        },
                    },
                    "validation": {
                        "distributions": {
                            "ball_speeds_mps": {"p99": 42},
                            "ball_accelerations_mps2": {
                                "p75": 11,
                                "p90": 21,
                            },
                            "missing_runs_seconds": {"p75": 0.4},
                        },
                    },
                },
            }))
            config_path.write_text(json.dumps({
                "video_ball_observations_path": "match-specific.json",
            }))

            config, report = build_compliance_outputs(
                metrics_path, config_path, 50
            )

            self.assertIsNone(config["video_ball_observations_path"])
            self.assertFalse(
                config["compliance_provenance"][
                    "match_specific_video_observations_enabled"
                ]
            )
            self.assertTrue(report["passed"])

    def test_compliance_config_cannot_reenable_video_observations(self):
        config = json.loads(
            (ROOT / "configs" / "compliant_tracking.json").read_text()
        )
        config["video_ball_observations_path"] = "match-specific.json"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-compliance.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(
                ValueError, "cannot enable"
            ):
                load_config(path)


class ExternalMetricaTests(unittest.TestCase):
    def test_current_validation_protocol_matches_deployed_frame_counts(self):
        protocol = load_validation_protocol(ROOT / "configs" / "external_transfer_validation_v2.json")
        baseline = load_config(ROOT / protocol["frozen_config"])
        self.assertEqual(protocol["frozen_config_source_fps"], 25)
        self.assertEqual(config_at_fps(baseline, 25, 25), baseline)
        self.assertEqual(protocol["metrica"]["games"]["untouched_test"], [])
        self.assertEqual(protocol["candidate_comparison"]["reference"], "deployment_semi_markov")
        official = next(c for c in protocol["candidate_comparison"]["candidates"]
                        if c["name"] == "organizer_smooth50_2s")
        self.assertEqual(official["baseline_half_window_seconds"], 2.0)

    def test_historical_validation_protocol_is_not_rewritten(self):
        path = ROOT / "configs" / "external_transfer_validation.json"
        self.assertEqual(load_validation_protocol(path), json.loads(path.read_text()))
        self.assertTrue(all(candidate["source_fps"] == 50
                            for candidate in load_validation_protocol(path)["candidate_comparison"]["candidates"]))

    def test_runtime_parity_rejects_calibration_fps_as_deployment_fps(self):
        protocol = json.loads((ROOT / "configs" / "external_transfer_validation_v2.json").read_text())
        protocol["candidate_comparison"]["candidates"][-1]["source_fps"] = 50
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-parity.json"
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "deployment candidate source fps"):
                load_validation_protocol(path)
            protocol["candidate_comparison"]["candidates"][-1]["source_fps"] = 25
            protocol["runtime_parity"]["fps"] = 50
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "challenge's declared"):
                load_validation_protocol(path)

    def test_gap_time_correction_does_not_rescale_inherited_windows_or_costs(self):
        baseline = load_config(ROOT / "configs" / "compliant_tracking.json")
        selection = {"config_key": "max_interpolation_gap_frames",
                     "selection_source_fps": 50, "selection_frames": 38}
        corrected = gap_time_correction(baseline, selection, 25)
        self.assertEqual([key for key in baseline if baseline[key] != corrected[key]],
                         ["max_interpolation_gap_frames"])
        self.assertEqual(corrected["max_interpolation_gap_frames"], 19)
        self.assertEqual(corrected["prediction_horizon_frames"], 10)
        self.assertEqual(corrected["semi_markov_live_transition_cost"], 30.0)
        self.assertEqual(baseline["max_interpolation_gap_frames"], 38)
        self.assertEqual(temporal_settings(baseline, 25)["windows"]
                         ["max_interpolation_gap_frames"]["seconds"], 1.52)
        with self.assertRaisesRegex(ValueError, "calibration lineage"):
            gap_time_correction(baseline, {**selection, "selection_source_fps": 25}, 25)

    def test_rescales_frame_windows_and_emission_sum_costs(self):
        config = default_config()
        config["max_interpolation_gap_frames"] = 38
        scaled = config_at_fps(config, 50, 25)
        self.assertEqual(scaled["max_interpolation_gap_frames"], 19)
        self.assertEqual(scaled["prediction_horizon_frames"], 5)
        self.assertEqual(scaled["semi_markov_live_transition_cost"], 15.0)
        self.assertEqual(config["max_interpolation_gap_frames"], 38)
        self.assertEqual(config_at_fps(config, 50, 50), config)
        for fps in (0, -1, np.nan):
            with self.assertRaisesRegex(ValueError, "fps"):
                config_at_fps(config, 50, fps)

    def test_masks_span_the_match_without_crossing_periods(self):
        ball = np.zeros((1000, 2))
        periods = np.r_[np.ones(500), np.full(500, 2)]
        starts = _select_windows(ball, 19, count=20, periods=periods)
        self.assertEqual(len(starts), 20)
        self.assertLess(starts[0], 50)
        self.assertGreater(starts[-1], 900)
        for start in starts:
            self.assertTrue(np.all(
                periods[start - 1:start + 20] == periods[start]
            ))

    def test_long_gaps_are_not_interpolated_at_wrong_frame_rate(self):
        config = config_at_fps(default_config(), 50, 25)
        config["prediction_horizon_frames"] = 0
        ball = np.column_stack((np.linspace(10, 11, 150), np.full(150, 20)))
        short = _gap_metrics(ball, 25, 0.2, config)
        long = _gap_metrics(ball, 25, 0.75, config)
        self.assertEqual(short["recovery_fraction"], 1.0)
        self.assertEqual(long["recovery_fraction"], 0.0)
        self.assertEqual(long["within_2m_fraction_of_masked"], 0.0)
        self.assertIsNone(long["p90_error_metres"])

    def test_excludes_conflicting_and_unobserved_event_anchors(self):
        data = {
            "frames": np.asarray([1, 2, 3]),
            "periods": np.ones(3, dtype=np.int8),
            "ball": np.asarray([[0, 0], [np.nan, np.nan], [0, 0]]),
            "home": np.zeros((3, 1, 2)),
            "away": np.ones((3, 1, 2)),
        }
        events = [
            {"frame": 1, "team": "home", "type": "PASS", "period": 1},
            {"frame": 1, "team": "away", "type": "RECOVERY", "period": 1},
            {"frame": 2, "team": "home", "type": "PASS", "period": 1},
            {"frame": 3, "team": "home", "type": "PASS", "period": 1},
        ]
        scored = _scoring_events(data, events)
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0]["row"], 2)

    def test_scores_decoded_labels_not_nearest_team_proxy(self):
        labels = np.asarray([AWAY, DEAD, HOME], dtype=np.int8)
        events = [
            {"row": 0, "team": "home", "type": "PASS", "period": 1},
            {"row": 1, "team": "away", "type": "RECOVERY", "period": 1},
            {"row": 2, "team": "home", "type": "PASS", "period": 1},
        ]
        result, correct = _decoded_event_metrics(labels, events)
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertEqual(correct.tolist(), [False, False, True])
        with self.assertRaisesRegex(ValueError, "anchor"):
            _decoded_event_metrics(labels, [])

    def test_paired_comparison_is_deterministic_and_signed(self):
        events = [
            {"row": 0, "period": 1}, {"row": 1500, "period": 1},
        ]
        first = np.asarray([True, True])
        second = np.asarray([False, True])
        result = _paired_event_delta(first, second, events, 25)
        self.assertEqual(result["delta_percentage_points"], 50.0)
        self.assertEqual(result["gained_events"], 1)
        self.assertEqual(
            result, _paired_event_delta(first, second, events, 25)
        )

    def test_gap_improvements_include_abstained_frames(self):
        candidate = {
            "mask_start_rows": [1, 10],
            "per_window": [
                {"recovery_fraction": 1.0, "within_2m_fraction_of_masked": 0.8},
                {"recovery_fraction": 1.0, "within_2m_fraction_of_masked": 0.8},
            ],
        }
        reference = {
            "mask_start_rows": [1, 10],
            "per_window": [
                {"recovery_fraction": 0.5, "within_2m_fraction_of_masked": 0.5},
                {"recovery_fraction": 0.5, "within_2m_fraction_of_masked": 0.5},
            ],
        }
        result = _paired_gap_delta(candidate, reference)
        self.assertEqual(result["recovery_fraction"]["delta_percentage_points"], 50)
        self.assertAlmostEqual(
            result["within_2m_fraction_of_masked"]["delta_percentage_points"], 30
        )
        with self.assertRaisesRegex(ValueError, "identical"):
            _paired_gap_delta(candidate, {**reference, "mask_start_rows": [2, 11]})

    def test_array_control_features_match_tracking_adapter(self):
        config = default_config()
        home = np.asarray([[[10., 20.]], [[10.04, 20.]]])
        away = np.asarray([[[30., 20.]], [[30.04, 20.]]])
        ball = np.asarray([[10., 20.], [10.04, 20.]])
        array_result = control_features_from_arrays(home, away, ball, 25, config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracking.txt"
            path.write_text(
                "1:0.1,0.2,NaN;0.3,0.2,NaN:0.1,0.2,0\n"
                "2:0.1004,0.2,1;0.3004,0.2,1:0.1004,0.2,0\n"
            )
            from possession.tracking import TrackingMetadata
            standard = extract_control_features(
                path, TrackingMetadata(np.asarray([HOME, AWAY]), ("H", "A")),
                ball, (100, 100), fps=25,
            )
        for field in (
            "nearest_distances", "nearby_counts", "nearest_player_xy",
            "nearest_player_velocity", "nearest_player_slot", "nearest_player_age_frames",
            "player_counts", "median_player_speed",
            "stationary_player_fraction",
        ):
            np.testing.assert_allclose(
                getattr(array_result, field), getattr(standard, field), atol=1e-4
            )

    def test_decodes_periods_separately_without_event_inputs(self):
        n_frames = 80
        data = {
            "ball": np.tile([20., 20.], (n_frames, 1)),
            "home": np.tile([[[20., 20.]]], (n_frames, 1, 1)),
            "away": np.tile([[[60., 20.]]], (n_frames, 1, 1)),
            "periods": np.r_[np.ones(40), np.full(40, 2)],
            "fps": 25,
        }
        labels = _candidate_labels(
            data, config_at_fps(default_config(), 50, 25)
        )
        self.assertEqual(len(labels), n_frames)
        self.assertTrue(np.all(labels == HOME))

    def test_converts_normalized_coordinates_to_pitch_metres(self):
        converted = _to_metres(np.asarray([
            [0.5, 0.5],
            [1.0, 0.0],
        ]))
        np.testing.assert_allclose(
            converted,
            np.asarray([
                [0.0, 0.0],
                [52.5, 34.0],
            ]),
        )

    def test_scores_event_team_from_nearest_player(self):
        frames = np.asarray([1, 2])
        ball = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        home = np.asarray([
            [[0.1, 0.0]],
            [[10.0, 0.0]],
        ])
        away = np.asarray([
            [[10.0, 0.0]],
            [[1.1, 0.0]],
        ])
        events = [
            {"frame": 1, "team": "home", "type": "PASS"},
            {"frame": 2, "team": "away", "type": "RECOVERY"},
        ]

        result = _event_team_metrics(
            frames, ball, home, away, events
        )

        self.assertEqual(result["events"], 2)
        self.assertEqual(result["accuracy"], 1.0)


class ExternalRecoveryTests(unittest.TestCase):
    def _events(self):
        return [
            {"type": "BALL LOST", "frame": 11, "end_frame": 20,
             "team": "away", "period": 1, "subtype": "INTERCEPTION"},
            {"type": "RECOVERY", "frame": 21, "end_frame": 21,
             "team": "home", "period": 1, "subtype": "INTERCEPTION"},
            {"type": "BALL LOST", "frame": 24, "end_frame": 25,
             "team": "home", "period": 1, "subtype": "INTERCEPTION"},
            {"type": "RECOVERY", "frame": 26, "end_frame": 26,
             "team": "away", "period": 1, "subtype": "INTERCEPTION"},
        ]

    def _settings(self):
        return {
            "boundary_search_seconds": 2.0,
            "boundary_tolerance_seconds": [0.0, 0.2, 0.5, 1.0],
            "short_reference_seconds": 1.0,
            "post_recovery_seconds": [0.2, 0.5, 1.0, 2.0],
        }

    def _references(self):
        frames = np.arange(1, 101)
        periods = np.ones(100, dtype=np.int8)
        references, counts = recovery_references(
            self._events(), frames, periods, 10, 0.2
        )
        return references, counts, periods

    def test_loss_recovery_pair_preserves_boundary_uncertainty(self):
        references, counts, _ = self._references()
        self.assertEqual(counts["paired_recoveries"], 2)
        self.assertEqual(references[0]["row"], 20)
        self.assertEqual(references[0]["boundary_start_row"], 19)
        self.assertEqual(references[0]["boundary_end_row"], 20)
        self.assertEqual(references[0]["outgoing"], AWAY)
        self.assertEqual(references[0]["incoming"], HOME)
        self.assertAlmostEqual(references[0]["reference_support_seconds"], 0.4)

    def test_ambiguous_or_unpaired_recoveries_are_not_invented(self):
        events = self._events()[:2]
        events.append({**events[0], "frame": 12})
        events.append({
            "type": "RECOVERY", "frame": 60, "end_frame": 60,
            "team": "home", "period": 1, "subtype": "",
        })
        references, counts = recovery_references(
            events, np.arange(1, 101), np.ones(100), 10, 0.2
        )
        self.assertEqual(references, [])
        self.assertEqual(counts["ambiguous_loss_matches"], 1)
        self.assertEqual(counts["unpaired_recoveries"], 1)

    def test_recovery_timing_and_censoring_are_not_anchor_accuracy(self):
        references, _, periods = self._references()
        labels = np.full(100, AWAY, dtype=np.int8)
        labels[20:25] = HOME
        result = score_recovery_transitions(
            labels, periods, references, 10, self._settings()
        )
        self.assertEqual(result["accuracy_at_recovery"], 1.0)
        self.assertEqual(result["same_direction_boundary_recall"]["0.0"], 1.0)
        self.assertEqual(result["cohorts"]["short_reference"]["events"], 1)
        self.assertEqual(
            result["event_derived_post_recovery_support"]["0.5"]["events"], 1
        )

    def test_same_boundary_cannot_match_two_references(self):
        references, _, periods = self._references()
        duplicate = {**references[0], "row": 22, "boundary_start_row": 22,
                     "boundary_end_row": 22}
        labels = np.full(100, AWAY, dtype=np.int8)
        labels[20:] = HOME
        result = score_recovery_transitions(
            labels, periods, [references[0], duplicate], 10, self._settings()
        )
        self.assertEqual(result["same_direction_boundary_recall"]["1.0"], 0.5)

    def test_recovery_masks_are_observed_nonoverlapping_and_period_local(self):
        references, _, periods = self._references()
        ball = np.zeros((100, 2))
        masked, starts, length = recovery_masks(
            ball, periods, references, 10, 0.75
        )
        self.assertEqual(length, 8)
        self.assertEqual(len(starts), 1)
        self.assertEqual(int(np.isnan(masked[:, 0]).sum()), 8)
        self.assertTrue(np.isfinite(ball).all())


class ExternalPossessionReferenceTests(unittest.TestCase):
    def _events(self):
        values = [
            ("SET PIECE", 1, 0, "home", "KICK OFF"),
            ("PASS", 2, 20, "home", ""),
            ("BALL OUT", 25, 30, "home", ""),
            ("SET PIECE", 50, 50, "away", "THROW IN"),
            ("PASS", 50, 70, "away", ""),
            ("BALL LOST", 75, 80, "away", "INTERCEPTION"),
            ("RECOVERY", 81, 81, "home", "INTERCEPTION"),
            ("PASS", 81, 95, "home", ""),
            ("BALL OUT", 96, 100, "home", ""),
        ]
        return [
            {"type": kind, "frame": start, "end_frame": stop,
             "team": team, "subtype": subtype, "period": 1}
            for kind, start, stop, team, subtype in values
        ]

    def _references(self, events=None):
        return possession_chain_references(
            self._events() if events is None else events,
            np.arange(1, 111), np.ones(110, dtype=np.int8), 10,
            {"loss_pair_tolerance_seconds": 0.2, "boundary_guard_seconds": 0},
        )

    def test_builds_live_dead_unknown_references_and_complete_spells(self):
        truth, report = self._references()
        self.assertEqual(truth[10], HOME)
        self.assertEqual(truth[35], DEAD)
        self.assertEqual(truth[60], AWAY)
        self.assertEqual(truth[79], UNKNOWN)
        self.assertEqual(truth[80], UNKNOWN)
        self.assertEqual(truth[85], HOME)
        self.assertEqual(truth[105], UNKNOWN)
        self.assertEqual(len(report["complete_live_spells"]), 3)
        self.assertFalse(report["challenge_equivalent_ground_truth"])

    def test_conflicting_events_do_not_create_complete_truth(self):
        events = self._events()
        events.append({
            "type": "PASS", "frame": 10, "end_frame": 15,
            "team": "away", "subtype": "", "period": 1,
        })
        truth, report = self._references(events)
        self.assertEqual(truth[12], UNKNOWN)
        self.assertGreater(report["summary"]["conflicting_claim_frames"], 0)
        self.assertLess(len(report["complete_live_spells"]), 3)

    def test_scores_known_frames_and_full_extent_spells_without_clipping(self):
        truth, references = self._references()
        labels = np.full(110, HOME, dtype=np.int8)
        labels[29:49] = DEAD
        labels[49:80] = AWAY
        labels[99:] = DEAD
        result = score_possession_references(labels, truth, references, np.ones(110))
        self.assertEqual(result["known_frame_accuracy"], 1.0)
        self.assertEqual(result["reference_conditioned_spell_scores"]["0.5"]["f1"], 1.0)
        wrong = np.full(110, HOME, dtype=np.int8)
        result = score_possession_references(wrong, truth, references, np.ones(110))
        self.assertEqual(result["reference_conditioned_spell_scores"]["0.75"]["matched"], 0)
        with self.assertRaisesRegex(ValueError, "H/A/D"):
            score_possession_references(truth, truth, references, np.ones(110))

    def test_dead_diagnostic_excludes_censored_frames_and_denominators_are_explicit(self):
        labels = np.asarray([HOME, DEAD, DEAD, DEAD, AWAY, DEAD], dtype=np.int8)
        truth = np.asarray([HOME, DEAD, DEAD, UNKNOWN, AWAY, HOME], dtype=np.int8)
        result = score_censored_dead_frames(labels, truth)
        self.assertEqual(result["known_dead_frames"], 2)
        self.assertEqual(result["known_live_frames"], 3)
        self.assertEqual(result["unknown_frames_excluded"], 1)
        self.assertEqual(result["correct_dead_frames"], 2)
        self.assertEqual(result["false_dead_on_known_live_frames"], 1)
        self.assertEqual(result["dead_precision_on_known_frames"], 2 / 3)
        self.assertEqual(result["dead_recall_on_known_frames"], 1)
        self.assertEqual(result["false_dead_fraction_of_known_live"], 1 / 3)
        with self.assertRaisesRegex(ValueError, "both known dead and known live"):
            score_censored_dead_frames(labels, np.full(len(labels), HOME, dtype=np.int8))
        with self.assertRaisesRegex(ValueError, "aligned H/A/D"):
            score_censored_dead_frames(labels[:-1], truth)

    def test_rejects_cross_period_live_events(self):
        events = self._events()
        periods = np.r_[np.ones(10), np.full(100, 2)]
        with self.assertRaises(ValueError):
            possession_chain_references(
                events, np.arange(1, 111), periods, 10,
                {"loss_pair_tolerance_seconds": 0.2, "boundary_guard_seconds": 0},
            )

    def test_calibration_protocol_and_grid_are_split_isolated(self):
        path = ROOT / "configs" / "metrica_handover_calibration.json"
        protocol = load_handover_protocol(path)
        self.assertEqual(protocol["splits"]["development"], ["Sample_Game_1"])
        self.assertEqual(len(handover_candidates(protocol)), 13)
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.json"
            protocol["splits"]["diagnostic_validation"] = ["Sample_Game_1"]
            invalid.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "split-isolated"):
                load_handover_protocol(invalid)


class LocalHandoverTests(unittest.TestCase):
    def _config(self):
        config = default_config()
        config.update({
            "semi_markov_local_handover_enabled": True,
            "semi_markov_local_handover_window_frames": 5,
            "semi_markov_local_handover_minimum_frames": 2,
            "semi_markov_local_handover_distance_metres": 1.5,
            "semi_markov_local_handover_distance_margin_metres": 0.5,
            "semi_markov_local_handover_minimum_confidence": 0.5,
        })
        return config

    def test_discount_requires_current_contact_and_past_support(self):
        contact = np.zeros((40, 2), dtype=bool)
        contact[20:30, AWAY] = True
        evidence = Evidence(
            control=np.zeros((40, 2)), dead=np.zeros(40),
            strong_control=contact, handover_control=contact,
        )
        support = _semi_markov_strong_transition_support(evidence, self._config())
        self.assertFalse(support[:21].any())
        self.assertTrue(support[21:30, AWAY].all())
        self.assertFalse(support[30:].any())

    def test_enabled_mode_cannot_silently_fall_back(self):
        evidence = Evidence(control=np.zeros((10, 2)), dead=np.zeros(10))
        with self.assertRaisesRegex(ValueError, "contact evidence"):
            _semi_markov_strong_transition_support(evidence, self._config())

    def test_only_reliable_observed_or_interpolated_ball_can_support_handover(self):
        n = 6
        from possession.tracking import ControlFeatures
        from possession.trajectory import BallTrajectory
        nearest = np.full((n, 2, 3), 4.0)
        nearest[:, HOME, 0] = 0.2
        control = ControlFeatures(
            nearest_distances=nearest, nearby_counts=np.zeros((n, 2, 3)),
            nearest_player_xy=np.zeros((n, 2, 2)),
            nearest_player_speed=np.zeros((n, 2)),
            nearest_player_velocity=np.zeros((n, 2, 2)),
            player_counts=np.full((n, 2), 11),
            median_player_speed=np.ones(n),
            stationary_player_fraction=np.zeros(n),
        )
        trajectory = BallTrajectory(
            xy=np.tile([50., 34.], (n, 1)), velocity=np.zeros((n, 2)), speed=np.zeros(n),
            confidence=np.asarray([1., .55, .9, 0., .9, .2]),
            source=np.asarray([1, 2, 3, 0, 4, 2]),
        )
        evidence = build_evidence(control, trajectory, empty_tags(n), (105, 68), self._config())
        self.assertEqual(evidence.handover_control[:, HOME].tolist(),
                         [True, True, False, False, False, False])
        self.assertFalse(evidence.handover_control[:, AWAY].any())

    def test_load_config_rejects_incomplete_local_mode(self):
        config = self._config()
        del config["semi_markov_local_handover_minimum_confidence"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "missing"):
                load_config(path)


class TrackingStoppageTests(unittest.TestCase):
    def _config(self):
        config = default_config()
        config.update({
            "tracking_stoppage_enabled": True,
            "tracking_stoppage_minimum_missing_frames": 5,
            "tracking_stoppage_motion_window_frames": 4,
            "tracking_stoppage_minimum_slow_fraction": 0.75,
            "tracking_stoppage_maximum_median_speed_mps": 1.5,
            "tracking_stoppage_minimum_stationary_fraction": 0.5,
            "tracking_stoppage_dead_weight": 0.9,
        })
        return config

    def _hint(self, source, speed, stationary):
        from types import SimpleNamespace
        features = SimpleNamespace(
            median_player_speed=np.asarray(speed),
            stationary_player_fraction=np.asarray(stationary),
        )
        trajectory = SimpleNamespace(source=np.asarray(source))
        return _tracking_stoppage_hint(features, trajectory, self._config())

    def test_missing_ball_alone_does_not_imply_dead_ball(self):
        hint = self._hint([0] * 15, [4.] * 15, [0.1] * 15)
        self.assertFalse(hint.any())

    def test_low_motion_requires_a_sustained_missing_run(self):
        source = [1] * 4 + [0] * 10 + [2] * 5
        hint = self._hint(source, [0.5] * len(source), [0.8] * len(source))
        self.assertFalse(hint[:8].any())
        self.assertTrue(hint[8:14].all())
        self.assertFalse(hint[14:].any())

    def test_unknown_motion_does_not_create_stoppage_evidence(self):
        hint = self._hint([0] * 15, [np.nan] * 15, [np.nan] * 15)
        self.assertFalse(hint.any())

    def test_stoppage_frame_parameters_scale_to_dataset_fps(self):
        scaled = config_at_fps(self._config(), 50, 25)
        self.assertEqual(scaled["tracking_stoppage_motion_window_frames"], 2)
        self.assertEqual(scaled["tracking_stoppage_minimum_missing_frames"], 3)

    def test_dead_evidence_is_opt_in_and_does_not_require_event_tags(self):
        from possession.trajectory import repair_ball_trajectory
        n = 30
        ball = np.full((n, 2), np.nan)
        home = np.tile(np.asarray([[[10., 20.]] * 11]), (n, 1, 1))
        away = home + 30
        config = self._config()
        control = control_features_from_arrays(home, away, ball, 25, config)
        trajectory = repair_ball_trajectory(ball, 25)
        baseline = build_evidence(control, trajectory, empty_tags(n), (105, 68), default_config())
        enhanced = build_evidence(control, trajectory, empty_tags(n), (105, 68), config)
        self.assertTrue(np.all(baseline.dead == 0))
        self.assertTrue(np.all(enhanced.dead[5:] >= 0.9))
        np.testing.assert_array_equal(baseline.control, enhanced.control)

    def test_protocol_has_six_stoppage_hypotheses_plus_baseline(self):
        protocol = load_handover_protocol(
            ROOT / "configs" / "metrica_stoppage_handover_calibration.json"
        )
        self.assertEqual(len(stoppage_candidates(protocol)), 7)

    def test_exit_bracket_requires_observed_outward_crossing_and_inside_return(self):
        config = {**default_config(), "tracking_exit_bracket_enabled": True,
                  "tracking_exit_bracket_minimum_frames": 5,
                  "tracking_exit_bracket_maximum_frames": 50,
                  "tracking_exit_bracket_minimum_outward_speed_mps": 2.}
        xy = np.column_stack((np.full(30, 50.), np.full(30, 34.))).astype(np.float32)
        xy[1, 0] = 104.8
        xy[2:17, 0] = 106.2
        xy[17:, 0] = 104.5
        source = np.ones(30, dtype=np.int8)
        source[4:16] = 0
        trajectory = SimpleNamespace(source=source, xy=xy)
        mask, diagnostics = tracking_exit_brackets(trajectory, (105., 68.), 25, config)
        self.assertTrue(mask[2:17].all())
        self.assertFalse(mask[:2].any())
        self.assertFalse(mask[17:].any())
        self.assertEqual(diagnostics["qualified_exit_to_return_intervals"], 1)
        shallow = trajectory.xy.copy()
        shallow[2:17, 0] = 105.2
        self.assertFalse(tracking_exit_brackets(
            SimpleNamespace(source=trajectory.source, xy=shallow),
            (105., 68.), 25, config,
        )[0].any())
        trajectory.source[17:] = 0
        with self.subTest("no observed return"):
            self.assertFalse(tracking_exit_brackets(trajectory, (105., 68.), 25, config)[0].any())
        trajectory.source[17:] = 1
        trajectory.xy[17, 0] = 105.2
        trajectory.xy[20:, 0] = 104.5
        config["tracking_exit_bracket_minimum_frames"] = 20
        with self.subTest("too brief"):
            self.assertFalse(tracking_exit_brackets(trajectory, (105., 68.), 25, config)[0].any())
        config["tracking_exit_bracket_minimum_frames"] = 5
        config["tracking_exit_bracket_maximum_frames"] = 10
        with self.subTest("too long"):
            self.assertFalse(tracking_exit_brackets(trajectory, (105., 68.), 25, config)[0].any())
        trajectory.xy[2] = np.nan
        with self.assertRaisesRegex(ValueError, "observed ball coordinates must be finite"):
            tracking_exit_brackets(trajectory, (105., 68.), 25, config)

    def test_exit_bracket_configuration_is_opt_in_and_validated(self):
        self.assertIsNone(validate_tracking_exit_bracket_config(default_config()))
        config = {**default_config(), "tracking_exit_bracket_enabled": True,
                  "tracking_exit_bracket_minimum_frames": 25,
                  "tracking_exit_bracket_maximum_frames": 3000,
                  "tracking_exit_bracket_minimum_outward_speed_mps": 2.}
        self.assertEqual(config_at_fps(config, 25, 50)["tracking_exit_bracket_minimum_frames"], 50)
        self.assertEqual(config_at_fps(config, 25, 50)["tracking_exit_bracket_maximum_frames"], 6000)
        for key, bad in (
            ("tracking_exit_bracket_minimum_frames", 1),
            ("tracking_exit_bracket_maximum_frames", 25),
            ("tracking_exit_bracket_minimum_outward_speed_mps", float("nan")),
            ("tracking_exit_bracket_enabled", "yes"),
            ("outside_pitch_dead_weight", float("nan")),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_tracking_exit_bracket_config({**config, key: bad})


class RestartDeadBallTests(unittest.TestCase):
    def _config(self):
        return {
            **default_config(),
            "tracking_kickoff_dead_enabled": True,
            "tracking_kickoff_formation_minimum_frames": 20,
            "tracking_kickoff_minimum_players": 16,
            "tracking_kickoff_halfway_tolerance_metres": 1.0,
            "tracking_kickoff_maximum_wrong_side_players": 1,
            "tracking_kickoff_merge_gap_frames": 10,
            "tracking_kickoff_corroboration_frames": 10,
            "tracking_kickoff_centre_radius_metres": 9.15,
            "tracking_kickoff_goal_mouth_half_width_metres": 4.66,
            "tracking_kickoff_goal_lookback_frames": 100,
            "tracking_kickoff_period_start_maximum_frames": 50,
            "tracking_restart_extension_enabled": True,
            "tracking_restart_maximum_frames": 100,
            "tracking_restart_bridge_frames": 15,
            "tracking_restart_touchline_distance_metres": 1.5,
            "tracking_restart_goal_area_margin_metres": 1.0,
        }

    def _kickoff_fixture(self, n=260):
        extremes = np.zeros((n, 2, 2, 3), dtype=np.float32)
        # Teams mixed across halfway by default (live play).
        extremes[:, HOME, 0] = 10.
        extremes[:, HOME, 1] = 80.
        extremes[:, AWAY, 0] = 20.
        extremes[:, AWAY, 1] = 95.
        for first, stop in ((30, 80), (84, 100), (150, 190), (220, 250)):
            extremes[first:stop, HOME, 1] = (60., 52., 50.)  # one stray home track allowed
            extremes[first:stop, AWAY, 0] = (54., 55., 56.)
        control = SimpleNamespace(
            team_x_extremes=extremes,
            player_counts=np.full((n, 2), 10, dtype=np.int16),
        )
        trajectory = SimpleNamespace(
            source=np.zeros(n, dtype=np.int8),
            xy=np.full((n, 2), np.nan, dtype=np.float32),
        )
        return trajectory, control

    def test_kickoff_dead_needs_corroborated_formation(self):
        config = self._config()
        trajectory, control = self._kickoff_fixture()
        trajectory.source[185] = 1
        trajectory.xy[185] = (55., 34.)
        tags = replace(
            empty_tags(260), restart_events=((70, 90, 75, HOME),),
            restart_types=("kick off (start)",),
        )
        dead, quality = kickoff_dead_intervals(
            trajectory, control, tags, (105., 68.), config, period_starts=(10,),
        )
        np.testing.assert_array_equal(np.flatnonzero(dead[:100]), np.arange(10, 100))
        self.assertTrue(dead[150:190].all())
        self.assertFalse(dead[100:150].any())
        self.assertFalse(dead[220:].any())
        self.assertEqual(quality["kickoffs_tag_corroborated"], 1)
        self.assertEqual(quality["kickoffs_ball_corroborated"], 1)
        self.assertEqual(quality["kickoffs_from_period_start"], 1)

        control.team_x_extremes[150:190, HOME, 1] = (60., 58., 50.)
        dead, _ = kickoff_dead_intervals(
            trajectory, control, tags, (105., 68.), config, period_starts=(10,),
        )
        self.assertFalse(dead[150:190].any())

    def test_kickoff_after_goal_starts_at_goal_mouth_entry(self):
        config = self._config()
        trajectory, control = self._kickoff_fixture()
        trajectory.source[100:112] = 1
        trajectory.xy[100:110] = (100., 34.)
        trajectory.xy[110:112] = (105.5, 35.)
        tags = replace(
            empty_tags(260), restart_events=((160, 170, 165, AWAY),),
            restart_types=("kick off (after goal)",),
        )
        dead, quality = kickoff_dead_intervals(
            trajectory, control, tags, (105., 68.), config, period_starts=(),
        )
        self.assertTrue(dead[110:190].all())
        self.assertFalse(dead[100:110].any())
        self.assertEqual(quality["kickoffs_after_goal_mouth_exit"], 1)

    def _restart_fixture(self):
        n = 200
        trajectory = SimpleNamespace(
            source=np.zeros(n, dtype=np.int8),
            xy=np.full((n, 2), np.nan, dtype=np.float32),
        )
        trajectory.source[:50] = 1
        trajectory.xy[:50] = (40., 20.)
        trajectory.source[60:80] = 1
        trajectory.xy[60:80] = (40., -1.)
        tags = replace(
            empty_tags(n), restart_events=((100, 120, 110, UNKNOWN),),
            restart_types=("throw-in",),
        )
        return trajectory, tags

    def test_restart_anchor_requires_boundary_sighting(self):
        config = self._config()
        trajectory, tags = self._restart_fixture()
        dead, quality = restart_anchored_dead_intervals(trajectory, tags, (105., 68.), config)
        np.testing.assert_array_equal(np.flatnonzero(dead), np.arange(50, 100))
        self.assertEqual(quality["restart_anchored_stoppages"], 1)

        config["tracking_restart_bridge_frames"] = 5
        dead, _ = restart_anchored_dead_intervals(trajectory, tags, (105., 68.), config)
        np.testing.assert_array_equal(np.flatnonzero(dead), np.arange(60, 100))

        trajectory.source[60:80] = 0
        dead, quality = restart_anchored_dead_intervals(trajectory, tags, (105., 68.), config)
        self.assertFalse(dead.any())
        self.assertEqual(quality["restart_anchor_abstentions"], 1)

        trajectory, tags = self._restart_fixture()
        tags = replace(tags, restart_types=("free kick",))
        self.assertFalse(restart_anchored_dead_intervals(
            trajectory, tags, (105., 68.), config,
        )[0].any())

    def test_goal_kick_anchor_accepts_ball_placed_in_goal_area(self):
        config = self._config()
        trajectory, tags = self._restart_fixture()
        trajectory.xy[60:80] = (3., 30.)
        tags = replace(tags, restart_types=("goal kick",))
        dead, _ = restart_anchored_dead_intervals(trajectory, tags, (105., 68.), config)
        np.testing.assert_array_equal(np.flatnonzero(dead), np.arange(50, 100))
        tags = replace(tags, restart_types=("throw-in",))
        self.assertFalse(restart_anchored_dead_intervals(
            trajectory, tags, (105., 68.), config,
        )[0].any())

    def test_restart_dead_config_is_opt_in_and_validated(self):
        validate_restart_dead_config(default_config())
        config = self._config()
        validate_restart_dead_config(config)
        self.assertEqual(config_at_fps(config, 25, 50)["tracking_restart_maximum_frames"], 200)
        for invalid in (
            {"tracking_kickoff_dead_enabled": "yes"},
            {"tracking_kickoff_formation_minimum_frames": 0},
            {"tracking_kickoff_maximum_wrong_side_players": 3},
            {"tracking_kickoff_centre_radius_metres": float("nan")},
            {"tracking_restart_bridge_frames": -1},
            {"tracking_restart_touchline_distance_metres": 0},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_restart_dead_config({**config, **invalid})

    def test_pre_kickoff_forced_dead_covers_block_start_to_first_kickoff(self):
        config = self._config()
        trajectory, control = self._kickoff_fixture()
        trajectory.source[185] = 1
        trajectory.xy[185] = (55., 34.)
        tags = replace(
            empty_tags(260), restart_events=((70, 90, 75, HOME),),
            restart_types=("kick off (start)",),
        )
        _, _, kickoffs = kickoff_dead_intervals(
            trajectory, control, tags, (105., 68.), config, period_starts=(10,),
            return_kickoffs=True,
        )
        self.assertEqual(kickoffs, [100, 190])
        # Pre-kickoff block, then a block without a detected kickoff.
        forced, quality = pre_kickoff_forced_dead(260, kickoffs, [(10, 140), (200, 260)])
        np.testing.assert_array_equal(np.flatnonzero(forced), np.arange(10, 100))
        self.assertEqual(quality["pre_kickoff_block_kickoffs"], [[10, 100], [200, None]])
        # A block graded from its kickoff is unchanged.
        forced, quality = pre_kickoff_forced_dead(260, kickoffs, [(100, 140)])
        self.assertFalse(forced.any())
        self.assertEqual(quality["pre_kickoff_block_kickoffs"], [[100, 100]])
        forced, _ = pre_kickoff_forced_dead(260, kickoffs, [(0, 140), (140, 260)])
        np.testing.assert_array_equal(np.flatnonzero(forced), np.r_[np.arange(0, 100), np.arange(140, 190)])
        with self.assertRaises(ValueError):
            pre_kickoff_forced_dead(260, kickoffs, [(200, 300)])

    def test_pre_kickoff_forced_dead_overrides_live_evidence_in_every_pass(self):
        config = default_config()
        n = 120
        control = np.zeros((n, 2), dtype=np.float32)
        control[:, HOME] = 2.
        dead = np.zeros(n, dtype=np.float32)
        free = decode(Evidence(control, dead), empty_tags(n), config)
        self.assertTrue((free == HOME).all())
        forced = np.zeros(n, dtype=bool)
        np.testing.assert_array_equal(
            decode(Evidence(control, dead, forced_dead=forced), empty_tags(n), config), free,
        )
        forced[:40] = True
        labels = decode(Evidence(control, dead, forced_dead=forced), empty_tags(n), config)
        self.assertTrue((labels[:40] == DEAD).all())
        self.assertTrue((labels[40:] == HOME).all())
        penalty = np.zeros((n, 3), dtype=np.float32)
        penalty[:, AWAY] = 1.
        labels = decode(
            Evidence(control, dead, forced_dead=forced, ballfree_penalty=penalty), empty_tags(n), config,
        )
        self.assertTrue((labels[:40] == DEAD).all())
        self.assertTrue((labels[40:] == HOME).all())

    def test_pre_kickoff_forced_dead_requires_kickoff_detection(self):
        config = {**self._config(), "pre_kickoff_forced_dead_enabled": True}
        validate_restart_dead_config(config)
        for invalid in (
            {"pre_kickoff_forced_dead_enabled": "yes"},
            {"tracking_kickoff_dead_enabled": False},
            {"decoder_mode": "hsmm"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_restart_dead_config({**config, **invalid})


class FrozenTransferCheckTests(unittest.TestCase):
    def test_protocol_locks_the_candidate_and_baseline(self):
        protocol, configs = load_frozen_transfer_protocol(
            ROOT / "configs" / "frozen_stoppage_transfer_check.json"
        )
        self.assertTrue(configs["candidate"]["tracking_stoppage_enabled"])
        self.assertIsNone(configs["candidate"]["video_ball_observations_path"])
        self.assertEqual(protocol["games"]["final_test"], ["Sample_Game_3"])

    def test_restart_tags_do_not_use_possession_or_recovery_labels(self):
        periods = np.ones(200, dtype=np.int8)
        events = [
            {"type": "SET PIECE", "frame": 101, "period": 1, "team": "home", "subtype": "THROW IN"},
            {"type": "RECOVERY", "frame": 90, "period": 1, "team": "away", "subtype": ""},
        ]
        settings = {"drop_fraction": 0.0, "jitter_seconds": 0.0, "unknown_team_fraction": 0.0}
        config = config_at_fps(default_config(), 50, 25)
        first = restart_only_tags(events, periods, 25, config, settings, 2027)
        changed = [events[0], {**events[1], "frame": 20, "team": "home"}]
        second = restart_only_tags(changed, periods, 25, config, settings, 2027)
        self.assertEqual(first.restart_events, second.restart_events)
        np.testing.assert_array_equal(first.dead_hint, second.dead_hint)
        np.testing.assert_array_equal(first.team_prior, second.team_prior)
        self.assertEqual(first.restart_frames.tolist(), [100])

    def test_tag_drop_and_unknown_team_are_explicit(self):
        event = {"type": "SET PIECE", "frame": 101, "period": 1,
                 "team": "home", "subtype": "THROW IN"}
        settings = {"drop_fraction": 1.0, "jitter_seconds": 0.0, "unknown_team_fraction": 0.0}
        config = config_at_fps(default_config(), 50, 25)
        tags = restart_only_tags([event], np.ones(200), 25, config, settings, 2027)
        self.assertFalse(tags.dead_hint.any())
        self.assertEqual(tags.restart_events, ())
        settings.update(drop_fraction=0.0, unknown_team_fraction=1.0)
        tags = restart_only_tags([event], np.ones(200), 25, config, settings, 2027)
        self.assertEqual(tags.restart_teams.tolist(), [UNKNOWN])
        self.assertTrue(tags.dead_hint.any())
        self.assertFalse(tags.team_prior.any())

    def test_stress_windows_are_known_live_and_do_not_cross_team_boundaries(self):
        ball = np.tile([50., 30.], (1000, 1))
        truth = np.r_[np.full(100, DEAD), np.full(400, HOME), np.full(500, AWAY)]
        periods = np.ones(1000, dtype=np.int8)
        masked, starts, length = live_missing_stress(ball, truth, periods, 25, 2, 20, 10)
        for start in starts:
            self.assertTrue(np.all(truth[start:start + length] == truth[start]))
            self.assertIn(truth[start], (HOME, AWAY))
            self.assertTrue(np.isnan(masked[start:start + length]).all())
        self.assertTrue(np.isfinite(ball).all())

    def test_primary_scores_exclude_shared_restart_endpoints(self):
        truth = np.full(500, HOME, dtype=np.int8)
        references = {"complete_live_spells": [
            {"start_row": 100, "stop_row": 200},
            {"start_row": 300, "stop_row": 400},
        ]}
        events = [{"type": "SET PIECE", "frame": 101, "period": 1}]
        filtered, refs = independent_reference_mask(
            truth, references, events, np.ones(500), 10, 3.0
        )
        self.assertTrue(np.all(filtered[70:131] == UNKNOWN))
        self.assertEqual(len(refs["complete_live_spells"]), 1)
        self.assertEqual(refs["complete_live_spells"][0]["start_row"], 300)

    def test_per_type_restart_windows_and_votes_rescale(self):
        config = config_at_fps(default_config(), 50, 25)
        self.assertEqual(config["restart_ownership_window_frames_by_type"]["throw-in"], 13)
        self.assertEqual(config["restart_ownership_minimum_votes"], 5)

    def test_epts_formats_map_substitutes_by_player_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            game = "Sample_Game_3"
            metadata = """<main><Metadata><GlobalConfig><FrameRate>25</FrameRate>
<ProviderGlobalParameters>
<ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
<ProviderParameter><Name>first_half_end</Name><Value>2</Value></ProviderParameter>
<ProviderParameter><Name>second_half_start</Name><Value>3</Value></ProviderParameter>
<ProviderParameter><Name>second_half_end</Name><Value>4</Value></ProviderParameter>
</ProviderGlobalParameters></GlobalConfig>
<FieldSize><Width>105</Width><Height>68</Height></FieldSize>
<Teams><Team id="H"><Name>Home</Name></Team><Team id="A"><Name>Away</Name></Team></Teams>
<Players><Player id="h1" teamId="H"/><Player id="h2" teamId="H"/><Player id="a1" teamId="A"/></Players>
<Sensor id="position"><Channels><Channel id="x"><Unit>normalized</Unit></Channel>
<Channel id="y"><Unit>normalized</Unit></Channel></Channels></Sensor><PlayerChannels>
<PlayerChannel id="h1x" channelId="x" playerId="h1"/><PlayerChannel id="h1y" channelId="y" playerId="h1"/>
<PlayerChannel id="h2x" channelId="x" playerId="h2"/><PlayerChannel id="h2y" channelId="y" playerId="h2"/>
<PlayerChannel id="a1x" channelId="x" playerId="a1"/><PlayerChannel id="a1y" channelId="y" playerId="a1"/>
</PlayerChannels></Metadata><DataFormatSpecifications>"""
            for start, stop, player in ((1, 2, "h1"), (3, 4, "h2")):
                metadata += """<DataFormatSpecification startFrame="{}" endFrame="{}" separator=":">
<StringRegister name="frameCount"/><SplitRegister separator=";">
<SplitRegister separator=","><PlayerChannelRef playerChannelId="{}x"/>
<PlayerChannelRef playerChannelId="{}y"/></SplitRegister>
<SplitRegister separator=","><PlayerChannelRef playerChannelId="a1x"/>
<PlayerChannelRef playerChannelId="a1y"/></SplitRegister></SplitRegister>
<SplitRegister separator=","><BallChannelRef channelId="x"/>
<BallChannelRef channelId="y"/></SplitRegister></DataFormatSpecification>""".format(start, stop, player, player)
            metadata += "</DataFormatSpecifications></main>"
            (path / (game + "_metadata.xml")).write_text(metadata)
            (path / (game + "_tracking.txt")).write_text(
                "1:0.1,0.2;0.9,0.8:0.5,0.5\n"
                "2:0.11,0.2;0.9,0.8:0.5,0.5\n"
                "3:0.5,0.4;0.9,0.8:NaN,NaN\n"
                "4:0.51,0.4;0.9,0.8:0.5,0.5\n"
            )
            (path / (game + "_events.json")).write_text(json.dumps({"data": [{
                "type": {"name": "CARRY"}, "team": {"id": "H", "name": "Home"},
                "start": {"frame": 3}, "end": {"frame": 4},
                "period": 2, "subtypes": None,
            }]}))
            data = _load_epts_game(path, game)
            self.assertEqual(data["home"].shape, (4, 2, 2))
            self.assertTrue(np.isnan(data["home"][2:, 0]).all())
            self.assertTrue(np.isnan(data["home"][:2, 1]).all())
            np.testing.assert_allclose(data["home"][2, 1], [52.5, 27.2], atol=1e-5)
            self.assertEqual(data["periods"].tolist(), [1, 1, 2, 2])
            events = _load_events(data["event_path"], {"CARRY"})
            self.assertEqual(events[0]["team"], "home")
            self.assertEqual(events[0]["type"], "CARRY")


class GuardedPossessionTests(unittest.TestCase):
    def _gap_config(self, mode="contact_guarded"):
        config = default_config()
        config.update({
            "semi_markov_gap_handover_enabled": True,
            "semi_markov_gap_handover_maximum_gap_frames": 5,
            "semi_markov_gap_handover_reappearance_frames": 2,
            "semi_markov_gap_handover_window_frames": 4,
            "semi_markov_gap_handover_minimum_frames": 2,
            "semi_markov_gap_handover_distance_metres": 1.5,
            "semi_markov_gap_handover_distance_margin_metres": 1.0,
            "semi_markov_gap_handover_minimum_confidence": 0.5,
            "semi_markov_gap_handover_mode": mode,
            "semi_markov_strong_transition_window_frames": 4,
            "semi_markov_strong_transition_minimum_frames": 2,
        })
        return config

    def _gap_data(self):
        from types import SimpleNamespace
        source = np.r_[np.ones(4), np.full(4, 2), np.ones(4)].astype(np.int8)
        nearest = np.full((12, 2), 5.0)
        nearest[:6, HOME] = 0.2
        nearest[6:, AWAY] = 0.2
        margin = nearest[:, ::-1] - nearest
        scores = np.full((12, 2), 0.1)
        scores[:6, HOME] = 0.8
        scores[6:, AWAY] = 0.8
        trajectory = SimpleNamespace(
            source=source, confidence=np.where(source == 2, 0.55, 1.0)
        )
        return trajectory, nearest, margin, scores

    def test_gap_context_requires_opposing_trusted_observed_anchors(self):
        trajectory, nearest, margin, scores = self._gap_data()
        contact, context = _gap_handover_evidence(
            trajectory, scores, nearest, margin, self._gap_config()
        )
        self.assertTrue(context[4:10, AWAY].all())
        self.assertFalse(context[:4].any())
        self.assertFalse(context[10:].any())
        self.assertFalse(context[:, HOME].any())
        self.assertFalse(contact[4:6].any())
        self.assertTrue(contact[6:10, AWAY].all())
        nearest[8] = [0.2, 5.0]
        margin = nearest[:, ::-1] - nearest
        _, context = _gap_handover_evidence(
            trajectory, scores, nearest, margin, self._gap_config()
        )
        self.assertFalse(context.any())

    def test_predicted_and_low_confidence_points_cannot_support_handover(self):
        trajectory, nearest, margin, scores = self._gap_data()
        trajectory.source[6] = 3
        trajectory.confidence[7] = 0.2
        contact, _ = _gap_handover_evidence(
            trajectory, scores, nearest, margin, self._gap_config()
        )
        self.assertFalse(contact[6:8].any())

    def test_normal_transition_policy_is_unchanged_outside_gap_context(self):
        trajectory, nearest, margin, scores = self._gap_data()
        contact, context = _gap_handover_evidence(
            trajectory, scores, nearest, margin, self._gap_config()
        )
        strong = np.zeros((12, 2), dtype=bool)
        strong[:4, HOME] = True
        strong[8:, AWAY] = True
        evidence = Evidence(
            control=scores, dead=np.zeros(12), strong_control=strong,
            gap_handover_control=contact, gap_handover_context=context,
        )
        config = self._gap_config()
        original = _semi_markov_strong_transition_support(
            evidence, {**config, "semi_markov_gap_handover_enabled": False}
        )
        changed = _semi_markov_strong_transition_support(evidence, config)
        np.testing.assert_array_equal(changed[~context], original[~context])
        self.assertTrue(original[6, AWAY])
        self.assertFalse(changed[6, AWAY])
        self.assertTrue(changed[7, AWAY])
        augmented = _semi_markov_strong_transition_support(
            evidence, self._gap_config("augment")
        )
        self.assertTrue(np.all(augmented[original]))

    def _corroboration_config(self):
        return {
            "pitch_margin_metres": 1.0,
            "tracking_stoppage_corroboration_lookback_frames": 20,
            "tracking_stoppage_exit_minimum_observations": 2,
            "tracking_stoppage_continuity_max_gap_frames": 20,
        }

    def test_same_team_reappearance_blocks_new_stoppage_without_a_restart(self):
        from types import SimpleNamespace
        trajectory = SimpleNamespace(
            source=np.r_[np.ones(4), np.zeros(10), np.ones(6)],
            xy=np.tile([50., 30.], (20, 1)),
        )
        tags = empty_tags(20)
        tags.pre_restart_window[:] = True
        anchors = np.full(20, HOME)
        supported = _corroborated_stoppage(
            trajectory, tags, (105, 68), anchors, self._corroboration_config()
        )
        self.assertFalse(supported[4:14].any())
        tags.restart_frames = np.asarray([10])
        supported = _corroborated_stoppage(
            trajectory, tags, (105, 68), anchors, self._corroboration_config()
        )
        self.assertTrue(supported[4:14].all())

    def test_exit_corrob_requires_observations_and_expires_on_in_pitch_reappearance(self):
        from types import SimpleNamespace
        trajectory = SimpleNamespace(
            source=np.r_[np.ones(2), np.zeros(8)],
            xy=np.r_[np.tile([108., 30.], (2, 1)), np.full((8, 2), np.nan)],
        )
        supported = _corroborated_stoppage(
            trajectory, empty_tags(10), (105, 68), np.full(10, UNKNOWN),
            self._corroboration_config(),
        )
        self.assertTrue(supported[2:].all())
        trajectory.source[2] = 1
        trajectory.xy[2] = [50., 30.]
        supported = _corroborated_stoppage(
            trajectory, empty_tags(10), (105, 68), np.full(10, UNKNOWN),
            self._corroboration_config(),
        )
        self.assertFalse(supported[3:].any())
        trajectory.source[:2] = 3
        supported = _corroborated_stoppage(
            trajectory, empty_tags(10), (105, 68), np.full(10, UNKNOWN),
            self._corroboration_config(),
        )
        self.assertFalse(supported.any())

    def test_a_restart_invalidates_previous_exit_evidence(self):
        from types import SimpleNamespace
        trajectory = SimpleNamespace(
            source=np.r_[np.ones(2), np.zeros(8)],
            xy=np.r_[np.tile([108., 30.], (2, 1)), np.full((8, 2), np.nan)],
        )
        tags = empty_tags(10)
        tags.restart_frames = np.asarray([6])
        tags.pre_restart_window[:6] = True
        supported = _corroborated_stoppage(
            trajectory, tags, (105, 68), np.full(10, UNKNOWN),
            self._corroboration_config(),
        )
        self.assertTrue(supported[2:6].all())
        self.assertFalse(supported[6:].any())

    def test_anchor_ambiguity_is_not_silently_assigned(self):
        nearest = np.asarray([[0.2, 5.], [0.2, 0.3], [np.inf, np.inf]])
        margins = np.asarray([[4.8, -4.8], [0.1, -0.1], [-np.inf, -np.inf]])
        teams = _anchor_teams(nearest, margins, np.ones(3, dtype=bool), 1.5, 1.0)
        self.assertEqual(teams.tolist(), [HOME, UNKNOWN, UNKNOWN])

    def test_protocol_has_long_live_negatives_and_excludes_consumed_games(self):
        path = ROOT / "configs" / "metrica_guarded_development.json"
        protocol, _ = load_guarded_protocol(path)
        self.assertEqual(protocol["live_missing_durations_seconds"], [2.0, 5.0])
        self.assertEqual(len(guarded_candidates(protocol)), 7)
        with tempfile.TemporaryDirectory() as directory:
            protocol["development_game"] = "Sample_Game_3"
            invalid = Path(directory) / "invalid.json"
            invalid.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "Game 1"):
                load_guarded_protocol(invalid)

    def test_false_dead_guard_overrides_an_apparent_f1_gain(self):
        protocol, _ = load_guarded_protocol(ROOT / "configs" / "metrica_guarded_development.json")
        metrics = {
            "possession": {
                "reference_conditioned_spell_scores": {"0.5": {"f1": 0.6}, "0.75": {"f1": 0.5}},
                "known_live_macro_f1": 0.8,
            },
            "recoveries": {"accuracy_at_recovery": 0.8,
                           "same_direction_boundary_recall": {"0.5": 0.6}},
            "spells": {"shorter_than_one_second": 2},
            "live_stress": {"false_dead_fraction": 0.0,
                            "windows_with_false_dead_fraction": 0.0, "split_window_fraction": 0.0},
        }
        baseline = {"cases": {"live_missing_2.0s:none": metrics}}
        changed = json.loads(json.dumps(baseline))
        changed["cases"]["live_missing_2.0s:none"]["live_stress"]["false_dead_fraction"] = 0.02
        checks = guarded_checks(changed, baseline, protocol)
        self.assertTrue(any(not check["passed"] and "false_dead" in check["name"] for check in checks))


class GapReleaseTests(unittest.TestCase):
    def test_exit_candidate_scope_rejects_unpinned_active_changes(self):
        baseline = {"decoder_mode": "semi_markov", "video_ball_observations_path": None}
        overrides = {"tracking_exit_bracket_enabled": True}
        candidate = {**baseline, **overrides, "exit_bracket_release_provenance": {}}
        self.assertEqual(
            _exit_candidate_changes(baseline, candidate, overrides),
            ["exit_bracket_release_provenance", "tracking_exit_bracket_enabled"],
        )
        with self.assertRaisesRegex(ValueError, "more than its frozen evidence"):
            _exit_candidate_changes(
                baseline, {**candidate, "video_ball_observations_path": "match.mp4"},
                overrides,
            )

    def test_provenance_audit_reads_archived_implementation_history(self):
        evidence = _plan_evidence(ROOT, "seven-frame quadratic fit outperformed")
        self.assertTrue(evidence["source"].startswith("docs/archive/IMPLEMENTATION_PLAN.md:"))
        self.assertIn("seven-frame quadratic fit outperformed", evidence["text"])

    def test_missing_historical_phrase_is_explicitly_unverified(self):
        evidence = _plan_evidence(ROOT, "this phrase does not appear in the historical log")
        self.assertEqual(evidence, {
            "source": "docs/archive/IMPLEMENTATION_PLAN.md", "status": "lineage_not_found",
        })

    def test_frozen_gap_release_accepts_only_compatible_preprocessing_changes(self):
        protocol, configs = load_frozen_transfer_protocol(
            ROOT / "configs" / "guarded_gap_release.json"
        )
        self.assertEqual(protocol["candidate_scope"], "gap_release")
        self.assertEqual(protocol["games"], {
            "reused_diagnostic": ["Sample_Game_2", "Sample_Game_3"]
        })
        self.assertTrue(configs["candidate"]["semi_markov_gap_handover_enabled"])
        self.assertFalse(configs["candidate"].get("tracking_stoppage_enabled", False))

    def test_release_spell_summary_uses_both_iou_thresholds(self):
        def case(first, second):
            return {
                "scenario": "clean", "tag_mode": "coarse_restart_tags",
                "models": {
                    name: {"primary_possession": {"reference_conditioned_spell_scores": {
                        "0.5": {"f1": values[0]}, "0.75": {"f1": values[1]},
                    }}}
                    for name, values in (("baseline", first), ("candidate", second))
                },
            }
        results = [
            {"game": "Sample_Game_2", "cases": [case((0.5, 0.3), (0.52, 0.3))]},
            {"game": "Sample_Game_3", "cases": [case((0.6, 0.4), (0.61, 0.41))]},
        ]
        summary = gap_release_spell_summary(results, {"spell_ious": [0.5, 0.75]})
        self.assertAlmostEqual(summary[0]["gain"], 0.01)
        self.assertAlmostEqual(summary[1]["gain"], 0.01)

    def test_unresolved_provenance_blocks_promotion_even_when_metrics_pass(self):
        result = release_decision(
            {"complete": True, "passed": True}, {"passed": True},
            {"active_tuned_parameter_provenance_cleared": False}, True,
        )
        self.assertEqual(result["decision"], "retain_current_submission")
        self.assertEqual(result["blockers"], ["active_parameter_provenance"])

    def test_every_required_gate_must_pass_for_promotion(self):
        audit = {"active_tuned_parameter_provenance_cleared": True}
        result = release_decision(
            {"complete": True, "passed": True}, {"passed": True}, audit, True,
        )
        self.assertEqual(result["decision"], "promote")
        incomplete = release_decision(
            {"complete": False, "passed": True}, {"passed": True}, audit, True,
        )
        self.assertEqual(incomplete["decision"], "retain_current_submission")


class CoreCalibrationTests(unittest.TestCase):
    def test_interaction_pass_preserves_the_original_thresholds_and_bound(self):
        base, _ = load_core_protocol(ROOT / "configs" / "metrica_core_development.json")
        extended, _ = load_core_protocol(
            ROOT / "configs" / "metrica_core_interaction_development.json"
        )
        self.assertEqual(extended["guards"], base["guards"])
        self.assertEqual(extended["selection_weights"], base["selection_weights"])
        self.assertEqual(len(extended["interaction_profiles"]), 6)
        self.assertEqual(extended["development_game"], "Sample_Game_1")

    def test_core_protocol_excludes_dead_restart_and_preprocessing_changes(self):
        path = ROOT / "configs" / "metrica_core_development.json"
        protocol, baseline = load_core_protocol(path)
        self.assertEqual(protocol["development_game"], "Sample_Game_1")
        self.assertEqual(len(protocol["evidence_profiles"]), 4)
        self.assertEqual(len(protocol["transition_profiles"]), 6)
        self.assertIsNone(baseline["video_ball_observations_path"])
        with tempfile.TemporaryDirectory() as directory:
            protocol["evidence_profiles"][1]["values"]["set_piece_dead_weight"] = 0.5
            invalid = Path(directory) / "invalid.json"
            invalid.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "fixed dead/restart"):
                load_core_protocol(invalid)

    def test_near_ties_prefer_fewer_parameter_changes(self):
        baseline = default_config()
        values = {key: baseline[key] for key in CORE_CALIBRATION_KEYS}
        experiments = [
            {"name": "baseline", "safe": True, "values": values,
             "evaluation": {"objective": 0.5}},
            {"name": "changed", "safe": True,
             "values": {**values, "nearby_two_weight": 0.06},
             "evaluation": {"objective": 0.5001}},
        ]
        self.assertEqual(
            choose_core_candidate(experiments, baseline, 0.0005)["name"], "baseline"
        )
        experiments[1]["evaluation"]["objective"] = 0.51
        self.assertEqual(
            choose_core_candidate(experiments, baseline, 0.0005)["name"], "changed"
        )
        experiments[1]["safe"] = False
        self.assertEqual(
            choose_core_candidate(experiments, baseline, 0.0005)["name"], "baseline"
        )

    def test_vectorized_duration_recurrence_matches_exhaustive_small_sequences(self):
        rng = np.random.default_rng(2030)
        config = default_config()
        config.update({
            "semi_markov_minimum_live_frames": 2,
            "semi_markov_minimum_dead_frames": 2,
            "semi_markov_live_transition_cost": 1.0,
            "semi_markov_strong_transition_cost": 0.25,
            "semi_markov_dead_transition_cost": 0.75,
            "semi_markov_strong_transition_window_frames": 3,
            "semi_markov_strong_transition_minimum_frames": 2,
        })
        for _ in range(6):
            evidence = Evidence(
                control=rng.uniform(0, 1.5, (8, 2)).astype(np.float32),
                dead=rng.uniform(0, 1, 8).astype(np.float32),
                strong_control=rng.random((8, 2)) > 0.5,
            )
            emissions = _semi_markov_emissions(evidence, config)
            support = _semi_markov_strong_transition_support(evidence, config)
            best_score, best_path = -np.inf, None
            for candidate in itertools.product((HOME, AWAY, DEAD), repeat=8):
                path = np.asarray(candidate)
                edges = np.r_[0, np.flatnonzero(np.diff(path)) + 1, 8]
                if np.any(np.diff(edges) < 2):
                    continue
                score = sum(float(emissions[frame, state]) for frame, state in enumerate(path))
                if path[0] == DEAD:
                    score += config["semi_markov_start_dead_bonus"]
                for frame in range(1, 8):
                    if path[frame] == path[frame - 1]:
                        continue
                    if DEAD in (path[frame], path[frame - 1]):
                        score -= config["semi_markov_dead_transition_cost"]
                    elif support[frame, path[frame]]:
                        score -= config["semi_markov_strong_transition_cost"]
                    else:
                        score -= config["semi_markov_live_transition_cost"]
                if score > best_score:
                    best_score, best_path = score, path
            np.testing.assert_array_equal(decode_semi_markov(evidence, config), best_path)


class CoreReliabilityTests(unittest.TestCase):
    def test_observed_strong_cost_restore_keeps_uncertain_cost_and_horizon(self):
        baseline, source, _ = self._configs("transitions")
        values = {key: source[key] for key in CORE_CALIBRATION_KEYS}
        repair = support_horizon_repair_values(baseline, values, restore_observed_strong_cost=True)
        self.assertEqual(repair["semi_markov_strong_transition_cost"], 10)
        self.assertEqual(repair["semi_markov_live_transition_cost"], 20)
        self.assertEqual(repair["core_uncertain_strong_transition_cost"], 5)
        self.assertEqual(repair["core_uncertain_live_transition_cost"], 20)
        self.assertEqual(repair["core_uncertain_strong_transition_window_frames"], 20)
        self.assertEqual(repair["core_uncertain_strong_transition_minimum_frames"], 4)
        self.assertEqual(values["semi_markov_strong_transition_cost"], 5)

    def test_horizon_only_repair_preserves_geometry_costs_and_observed_support(self):
        baseline, source, _ = self._configs("transitions")
        values = {key: source[key] for key in CORE_CALIBRATION_KEYS}
        overrides = support_horizon_repair_values(baseline, values)
        config = {**source, **overrides}
        self.assertEqual(config["core_uncertain_strong_transition_window_frames"], 20)
        self.assertEqual(config["core_uncertain_strong_transition_minimum_frames"], 4)
        self.assertEqual(config["core_uncertain_live_transition_cost"], 20)
        self.assertEqual(config["core_uncertain_strong_transition_cost"], 5)
        self.assertEqual(config["core_uncertain_strong_control_distance_metres"], 1)
        fraction = support_horizon_repair_values(baseline, values, preserve_fraction=True)
        self.assertEqual(fraction["core_uncertain_strong_transition_minimum_frames"], 8)
        self.assertEqual(fraction["core_uncertain_strong_transition_window_frames"], 20)
        self.assertEqual(fraction["semi_markov_strong_transition_minimum_frames"], 4)
        strong = np.zeros((30, 2), dtype=bool)
        strong[12:18, AWAY] = True
        observed = np.ones(30, dtype=bool)
        observed[3] = False
        evidence = Evidence(
            control=np.zeros((30, 2)), dead=np.zeros(30), strong_control=strong,
            reliable_observation=observed, uncertain_strong_control=strong.copy(),
        )
        normal = _semi_markov_strong_transition_support(evidence, {**source, "core_reliability_mode": "none"})
        repaired = _semi_markov_strong_transition_support(evidence, config)
        self.assertFalse(normal[3, AWAY])
        self.assertTrue(repaired[3, AWAY])
        np.testing.assert_array_equal(normal[observed], repaired[observed])
        with self.assertRaisesRegex(ValueError, "shorter candidate"):
            support_horizon_repair_values(baseline, {**values, "semi_markov_strong_transition_window_frames": 20})

    def _write_protocol_fixture(self, project):
        extension = json.loads((ROOT / "configs" / "metrica_reliability_development.json").read_text())
        base_bytes = (ROOT / extension["base_protocol"]).read_bytes()
        base_protocol, baseline = load_core_protocol(ROOT / extension["base_protocol"])
        values = {key: baseline[key] for key in CORE_CALIBRATION_KEYS}
        for group, name in (("evidence_profiles", "proximity_focused"),
                            ("transition_profiles", "strict_responsive")):
            values.update(next(profile["values"] for profile in base_protocol[group]
                               if profile["name"] == name))
        report = {
            "development_game": "Sample_Game_1",
            "baseline_sha256": base_protocol["baseline_sha256"],
            "excluded_games": ["Sample_Game_2", "Sample_Game_3"],
            "experiments": [{"name": extension["source_candidate"], "values": values}],
        }
        report_path = project / extension["source_report"]
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps(report))
        extension["source_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        base_path = project / extension["base_protocol"]
        base_path.parent.mkdir(parents=True)
        base_path.write_bytes(base_bytes)
        protocol_path = project / "configs" / "reliability_fixture.json"
        protocol_path.write_text(json.dumps(extension))
        return protocol_path, report_path

    def _configs(self, mode):
        baseline = default_config()
        candidate = {
            **baseline,
            "closeness_scale_metres": 2.5,
            "nearby_two_weight": 0.06,
            "nearby_four_weight": 0.02,
            "semi_markov_live_transition_cost": 20.0,
            "semi_markov_strong_transition_cost": 5.0,
            "semi_markov_strong_transition_window_frames": 10,
            "semi_markov_strong_transition_minimum_frames": 4,
            "strong_control_distance_metres": 1.0,
            "strong_control_distance_margin_metres": 1.5,
        }
        return baseline, candidate, {**candidate, **reliability_overrides(baseline, mode)}

    def _inputs(self):
        from possession.tracking import ControlFeatures
        from possession.trajectory import BallTrajectory
        n = 6
        nearest = np.full((n, 2, 3), 3.0)
        nearest[:, HOME, 0] = 0.3
        counts = np.zeros((n, 2, 3), dtype=np.int16)
        counts[:, HOME, 1:] = [2, 4]
        counts[:, AWAY, 1:] = [1, 2]
        features = ControlFeatures(
            nearest_distances=nearest, nearby_counts=counts,
            nearest_player_xy=np.zeros((n, 2, 2)),
            nearest_player_speed=np.zeros((n, 2)),
            nearest_player_velocity=np.zeros((n, 2, 2)),
            player_counts=np.full((n, 2), 11),
            median_player_speed=np.ones(n),
            stationary_player_fraction=np.zeros(n),
        )
        trajectory = BallTrajectory(
            xy=np.tile([50., 30.], (n, 1)),
            velocity=np.zeros((n, 2)), speed=np.zeros(n),
            source=np.asarray([1, 2, 3, 0, 4, 1]),
            confidence=np.asarray([1., .55, .3, 0., .6, 1.]),
        )
        return features, trajectory

    def test_geometry_fallback_restores_baseline_only_on_nonobserved_sources(self):
        baseline, candidate, conditioned = self._configs("geometry")
        features, trajectory = self._inputs()
        tags = empty_tags(6)
        original = build_evidence(features, trajectory, tags, (105, 68), baseline)
        changed = build_evidence(features, trajectory, tags, (105, 68), candidate)
        hybrid = build_evidence(features, trajectory, tags, (105, 68), conditioned)
        reliable = trajectory.source == 1
        np.testing.assert_array_equal(hybrid.control[reliable], changed.control[reliable])
        np.testing.assert_array_equal(hybrid.control[~reliable], original.control[~reliable])
        np.testing.assert_array_equal(hybrid.dead, original.dead)
        self.assertEqual(hybrid.reliable_observation.tolist(), reliable.tolist())

    def test_transition_only_fallback_keeps_new_evidence(self):
        _, candidate, conditioned = self._configs("transitions")
        features, trajectory = self._inputs()
        changed = build_evidence(features, trajectory, empty_tags(6), (105, 68), candidate)
        hybrid = build_evidence(features, trajectory, empty_tags(6), (105, 68), conditioned)
        np.testing.assert_array_equal(hybrid.control, changed.control)
        self.assertIsNotNone(hybrid.uncertain_strong_control)

    def test_transition_costs_and_support_match_baseline_when_all_input_is_uncertain(self):
        baseline, candidate, conditioned = self._configs("transitions")
        for config in (baseline, candidate, conditioned):
            config["semi_markov_minimum_live_frames"] = 3
            config["semi_markov_minimum_dead_frames"] = 2
        rng = np.random.default_rng(2070)
        control = rng.uniform(0, 1.5, (40, 2)).astype(np.float32)
        dead = rng.uniform(0, 0.5, 40).astype(np.float32)
        strong = rng.random((40, 2)) > 0.5
        fallback = rng.random((40, 2)) > 0.5
        original = decode_semi_markov(
            Evidence(control, dead, fallback), baseline
        )
        hybrid = decode_semi_markov(
            Evidence(
                control, dead, strong,
                reliable_observation=np.zeros(40, dtype=bool),
                uncertain_strong_control=fallback,
            ), conditioned,
        )
        np.testing.assert_array_equal(hybrid, original)
        trusted = decode_semi_markov(Evidence(control, dead, strong), candidate)
        hybrid = decode_semi_markov(
            Evidence(
                control, dead, strong,
                reliable_observation=np.ones(40, dtype=bool),
                uncertain_strong_control=fallback,
            ), conditioned,
        )
        np.testing.assert_array_equal(hybrid, trusted)

    def test_enabled_fallback_cannot_silently_ignore_missing_reliability(self):
        _, _, conditioned = self._configs("transitions")
        with self.assertRaisesRegex(ValueError, "aligned observation"):
            decode_semi_markov(Evidence(np.zeros((30, 2)), np.zeros(30)), conditioned)

    def test_uncertain_frame_counts_scale_with_dataset_fps(self):
        _, _, conditioned = self._configs("both")
        scaled = config_at_fps(conditioned, 50, 25)
        self.assertEqual(scaled["core_uncertain_strong_transition_window_frames"], 10)
        self.assertEqual(scaled["core_uncertain_strong_transition_minimum_frames"], 4)
        self.assertEqual(scaled["core_uncertain_live_transition_cost"], 15.0)

    def test_rescue_protocol_keeps_the_original_guards_and_three_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            fixture, _ = self._write_protocol_fixture(project)
            with mock.patch("possession.external_reliability_calibration.__file__",
                            str(project / "possession" / "external_reliability_calibration.py")):
                extension, protocol, baseline, _, source = load_reliability_protocol(fixture)
        self.assertEqual(extension["modes"], ["geometry", "transitions", "both"])
        self.assertEqual(protocol["development_game"], "Sample_Game_1")
        self.assertEqual(protocol["guards"]["maximum_case_spell_f1_drop"], 0.01)
        self.assertEqual(source["name"], "proximity_focused+strict_responsive")
        self.assertEqual(
            reliability_overrides(baseline, "both")["core_uncertain_live_transition_cost"],
            baseline["semi_markov_live_transition_cost"],
        )

    def test_replay_still_requires_hash_verified_report_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            fixture, report = self._write_protocol_fixture(project)
            with mock.patch("possession.external_reliability_calibration.__file__",
                            str(project / "possession" / "external_reliability_calibration.py")):
                report.write_text(report.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "source report changed"):
                    load_reliability_protocol(fixture)
                report.unlink()
                with self.assertRaises(FileNotFoundError):
                    load_reliability_protocol(fixture)






class ControlPhaseTests(unittest.TestCase):
    def _motion_fixture(self):
        n = 36
        x = np.arange(n, dtype=np.float32) * .4
        x[:6] = np.arange(6, dtype=np.float32) * .08
        x[6:] = x[5] + np.arange(1, n - 5, dtype=np.float32) * .4
        xy = np.column_stack((x, np.zeros(n, dtype=np.float32)))
        source = np.ones(n, dtype=np.int8)
        source[12:14] = 2
        nearest = np.full((n, 2, 1), 4., dtype=np.float32)
        nearest[2:6, HOME, 0] = .5
        nearest[18:22, AWAY, 0] = .5
        slots = np.full((n, 2), -1, dtype=np.int16)
        slots[2:6, HOME] = 0
        slots[18:22, AWAY] = 0
        trajectory = SimpleNamespace(
            xy=xy, source=source,
            speed=np.linalg.norm(np.gradient(xy, axis=0) * 25, axis=1),
        )
        control = SimpleNamespace(
            nearest_distances=nearest, nearest_player_slot=slots,
        )
        config = {
            "phase_contact_minimum_frames": 4,
            "phase_flight_max_frames": 20,
            "phase_minimum_ball_speed_mps": 2.,
            "phase_contact_distance_metres": 1.5,
            "phase_motion_continuity_enabled": True,
            "phase_contact_motion_minimum": .25,
            "phase_bridge_max_frames": 3,
            "max_ball_speed_metres_per_second": 65.,
            "max_ball_acceleration_metres_per_second_squared": 45.,
        }
        return trajectory, control, empty_tags(n), nearest[:, :, 0] < 1.5, config

    def test_motion_contact_bridges_short_straight_gap_not_opponent_proximity(self):
        trajectory, control, tags, strong, config = self._motion_fixture()
        phase = infer_control_phase(
            trajectory, control, tags, strong, config, fps=25,
        )
        self.assertEqual(phase.kind[5], CONTACT)
        self.assertEqual(phase.kind[12], FLIGHT)
        self.assertEqual(phase.kind[13], FLIGHT)
        self.assertEqual(phase.team[12], HOME)
        self.assertEqual(phase.kind[21], UNSEEN)
        self.assertEqual(phase.diagnostics["physics_qualified_gap_frames"], 2)
        self.assertEqual(phase.diagnostics["physics_bridge_frames"], 2)
        self.assertEqual(phase.diagnostics["motion_supported_contact_frames"], 1)

        trajectory.xy[18:] += np.arange(
            1, len(trajectory.xy) - 17, dtype=np.float32,
        )[:, None] * np.array([-.3, 0], dtype=np.float32)
        trajectory.speed = np.linalg.norm(
            np.gradient(trajectory.xy, axis=0) * 25, axis=1,
        )
        turnover = infer_control_phase(
            trajectory, control, tags, strong, config, fps=25,
        )
        self.assertEqual(turnover.kind[21], TURNOVER)
        self.assertEqual(turnover.team[21], AWAY)
        self.assertEqual(turnover.kind[22], FLIGHT)
        self.assertEqual(turnover.team[22], AWAY)

    def test_motion_bridge_abstains_on_unqualified_gaps(self):
        trajectory, control, tags, strong, config = self._motion_fixture()
        trajectory.xy[14:] += np.array([2., 0], dtype=np.float32)
        phase = infer_control_phase(
            trajectory, control, tags, strong, config, fps=25,
        )
        self.assertEqual(phase.kind[12], UNSEEN)
        self.assertEqual(phase.kind[14], UNSEEN)
        trajectory, control, tags, strong, config = self._motion_fixture()
        trajectory.source[10:15] = 2
        phase = infer_control_phase(
            trajectory, control, tags, strong, config, fps=25,
        )
        self.assertEqual(phase.kind[12], UNSEEN)
        trajectory, control, tags, strong, config = self._motion_fixture()
        tags.dead_hint[12] = True
        phase = infer_control_phase(
            trajectory, control, tags, strong, config, fps=25,
        )
        self.assertEqual(phase.kind[12], STOPPAGE)
        self.assertEqual(phase.kind[13], UNSEEN)
        with self.assertRaisesRegex(ValueError, "positive physical fps"):
            infer_control_phase(trajectory, control, tags, strong, config)

    def test_motion_continuity_pins_only_complete_supported_flights(self):
        config = default_config()
        config.update({
            "semi_markov_phase_enabled": True,
            "phase_decode_mode": "bounded",
            "phase_motion_continuity_enabled": True,
            "phase_contact_minimum_frames": 4,
            "phase_contact_motion_minimum": .25,
            "phase_bridge_max_frames": 3,
            "phase_flight_max_frames": 20,
            "phase_minimum_ball_speed_mps": 2.,
            "phase_contact_distance_metres": 1.5,
            "phase_flight_weight": .25,
            "phase_turnover_transition_cost": 8.,
            "semi_markov_minimum_live_frames": 5,
            "semi_markov_live_transition_cost": 8.,
            "semi_markov_strong_transition_cost": 8.,
        })
        control = np.zeros((90, 2), dtype=np.float32)
        control[:, HOME] = 1.
        control[20:30, HOME] = 0.
        control[20:30, AWAY] = 1.
        phase = ControlPhase(
            np.full(90, UNSEEN, dtype=np.int8),
            np.full(90, UNKNOWN, dtype=np.int8), {},
            np.full(90, -1, dtype=np.int32),
        )
        phase.kind[15] = CONTACT
        phase.team[15] = HOME
        phase.kind[16:34] = FLIGHT
        phase.team[16:34] = HOME
        phase.anchor_rows[16:34] = 15
        phase.kind[31:34] = UNSEEN
        phase.team[31:34] = UNKNOWN
        phase.anchor_rows[31:34] = -1
        phase.kind[34] = RECEPTION
        phase.team[34] = HOME
        strong = np.zeros((90, 2), dtype=bool)
        strong[31:34, HOME] = True
        evidence = Evidence(
            control, np.zeros(90, dtype=np.float32),
            strong_control=strong, control_phase=phase,
        )
        baseline = decode_semi_markov(
            evidence, {**config, "semi_markov_phase_enabled": False,
                       "phase_motion_continuity_enabled": False},
        )
        self.assertTrue(np.all(baseline[20:30] == AWAY))
        pinned = decode(evidence, empty_tags(90), config)
        self.assertTrue(np.all(pinned[16:31] == HOME))
        np.testing.assert_array_equal(pinned[31:], baseline[31:])
        evidence.dead[21:28] = 100.
        dead_baseline = decode_semi_markov(
            evidence, {**config, "semi_markov_phase_enabled": False,
                       "phase_motion_continuity_enabled": False},
        )
        self.assertTrue(np.any(dead_baseline[21:28] == DEAD))
        np.testing.assert_array_equal(
            decode(evidence, empty_tags(90), config), dead_baseline,
        )
        evidence.dead[21:28] = 0.
        phase.kind[23] = UNSEEN
        phase.team[23] = UNKNOWN
        phase.anchor_rows[23] = -1
        np.testing.assert_array_equal(
            decode(evidence, empty_tags(90), config), baseline,
        )
        phase.kind[23] = FLIGHT
        phase.team[23] = HOME
        phase.anchor_rows[23] = 15
        phase.kind[34] = TURNOVER
        phase.team[34] = AWAY
        strong[31:34, HOME] = False
        strong[31:34, AWAY] = True
        control[34:, HOME] = 0.
        control[34:, AWAY] = 1.
        baseline = decode_semi_markov(
            evidence, {**config, "semi_markov_phase_enabled": False,
                       "phase_motion_continuity_enabled": False},
        )
        self.assertEqual(baseline[15], HOME)
        self.assertEqual(baseline[34], AWAY)
        switched = decode(evidence, empty_tags(90), config)
        self.assertTrue(np.all(switched[16:31] == HOME))
        self.assertEqual(switched[34], AWAY)
        np.testing.assert_array_equal(switched[31:], baseline[31:])
        strong[31:39, AWAY] = True
        supported_baseline = decode_semi_markov(
            evidence, {**config, "semi_markov_phase_enabled": False,
                       "phase_motion_continuity_enabled": False},
        )
        self.assertTrue(np.any(supported_baseline[16:31] == AWAY))
        np.testing.assert_array_equal(
            decode(evidence, empty_tags(90), config), supported_baseline,
        )
        for invalid in (
            {"semi_markov_phase_enabled": False},
            {"phase_contact_motion_minimum": 0.},
            {"phase_bridge_max_frames": 21},
            {"phase_decode_mode": "global"},
            {"phase_observed_reception_bridge_enabled": True,
             "phase_observed_reception_bridge_max_frames": 30},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_control_phase_config({**config, **invalid})

    def _fixture(self):
        n = 14
        source = np.ones(n, dtype=np.int8)
        speed = np.full(n, 5., dtype=np.float32)
        nearest = np.full((n, 2, 1), 4., dtype=np.float32)
        slots = np.full((n, 2), -1, dtype=np.int16)
        for start, stop, team, slot in (
            (1, 4, HOME, 0), (6, 8, AWAY, 0), (10, 12, AWAY, 1),
        ):
            nearest[start:stop, team, 0] = .5
            slots[start:stop, team] = slot
        trajectory = SimpleNamespace(source=source, speed=speed)
        control = SimpleNamespace(nearest_distances=nearest, nearest_player_slot=slots)
        config = {
            "phase_contact_minimum_frames": 2, "phase_flight_max_frames": 5,
            "phase_minimum_ball_speed_mps": 2.,
            "phase_contact_distance_metres": 1.,
        }
        return trajectory, control, empty_tags(n), nearest[:, :, 0] < 1, config

    def _long_reception_fixture(self):
        n = 45
        nearest = np.full((n, 2, 1), 4., dtype=np.float32)
        slots = np.full((n, 2), -1, dtype=np.int16)
        nearest[1:4, HOME, 0] = .5
        nearest[32:35, HOME, 0] = .5
        slots[1:4, HOME] = 0
        slots[32:35, HOME] = 1
        velocities = np.zeros((n, 2, 2), dtype=np.float32)
        velocities[32:35, HOME, 0] = 2.5
        control = SimpleNamespace(
            nearest_distances=nearest, nearest_player_slot=slots,
            nearest_player_velocity=velocities,
        )
        ball_velocity = np.zeros((n, 2), dtype=np.float32)
        ball_velocity[:, 0] = 2.5
        trajectory = SimpleNamespace(
            source=np.ones(n, dtype=np.int8),
            speed=np.full(n, 2.5, dtype=np.float32),
            xy=np.column_stack((np.arange(n) * .1, np.zeros(n))).astype(np.float32),
            velocity=ball_velocity,
        )
        config = {
            "phase_contact_minimum_frames": 3, "phase_flight_max_frames": 5,
            "phase_minimum_ball_speed_mps": 2.,
            "phase_contact_distance_metres": 1.5,
            "phase_observed_reception_bridge_enabled": True,
            "phase_observed_reception_bridge_max_frames": 38,
            "strong_control_relative_speed_metres_per_second": 2.,
        }
        return trajectory, control, empty_tags(n), nearest[:, :, 0] < 1.5, config

    def test_long_reception_requires_observed_flight_and_same_track_motion(self):
        trajectory, control, tags, strong, config = self._long_reception_fixture()
        short = infer_control_phase(
            trajectory, control, tags, strong,
            {**config, "phase_observed_reception_bridge_enabled": False},
        )
        self.assertEqual(short.kind[34], CONTACT)
        linked = infer_control_phase(trajectory, control, tags, strong, config)
        self.assertEqual(linked.kind[34], RECEPTION)
        self.assertEqual(linked.team[34], HOME)
        self.assertEqual(linked.diagnostics["observed_long_receptions"], 1)
        self.assertGreater(linked.diagnostics["observed_long_flight_frames"], 0)
        self.assertEqual(linked.kind[20], FLIGHT)
        self.assertEqual(short.kind[20], UNSEEN)
        self.assertEqual(linked.anchor_rows[20], 3)
        self.assertEqual(linked.kind[35], FLIGHT)

    def test_long_reception_abstains_on_missing_contested_or_incoherent_motion(self):
        changes = (
            lambda t, c, tags: t.source.__setitem__(20, 0),
            lambda t, c, tags: c.nearest_distances.__setitem__((20, AWAY, 0), .5),
            lambda t, c, tags: tags.dead_hint.__setitem__(20, True),
            lambda t, c, tags: c.nearest_player_velocity.__setitem__(
                (slice(32, 35), HOME, 0), 0.,
            ),
            lambda t, c, tags: t.xy.__setitem__((slice(None), 0), -t.xy[:, 0]),
        )
        for change in changes:
            trajectory, control, tags, strong, config = self._long_reception_fixture()
            change(trajectory, control, tags)
            with self.subTest(change=change):
                phase = infer_control_phase(trajectory, control, tags, strong, config)
                self.assertEqual(phase.kind[34], CONTACT)
                self.assertEqual(phase.diagnostics["observed_long_receptions"], 0)
        trajectory, control, tags, strong, config = self._long_reception_fixture()
        control.nearest_player_velocity = None
        with self.assertRaisesRegex(ValueError, "same-track player motion"):
            infer_control_phase(trajectory, control, tags, strong, config)

    def test_long_reception_can_repair_a_bounded_opponent_flight_pulse(self):
        trajectory, control, tags, strong, profile = self._long_reception_fixture()
        config = {
            **default_config(), **profile,
            "semi_markov_phase_enabled": True, "phase_decode_mode": "bounded",
            "phase_flight_weight": 4., "phase_turnover_transition_cost": 8.,
            "semi_markov_minimum_live_frames": 5,
            "semi_markov_live_transition_cost": 8.,
            "semi_markov_strong_transition_cost": 8.,
        }
        team_scores = np.zeros((len(trajectory.source), 2), dtype=np.float32)
        team_scores[:, HOME] = 1.
        team_scores[20:29, HOME] = 0.
        team_scores[20:29, AWAY] = 1.
        evidence = Evidence(
            team_scores, np.zeros(len(team_scores), dtype=np.float32),
            strong_control=strong,
            control_phase=infer_control_phase(trajectory, control, tags, strong, config),
        )
        baseline = decode_semi_markov(
            evidence, {**config, "semi_markov_phase_enabled": False},
        )
        self.assertTrue(np.all(baseline[20:29] == AWAY))
        repaired = decode(evidence, tags, config)
        self.assertTrue(np.all(repaired[20:29] == HOME))
        np.testing.assert_array_equal(repaired[:4], baseline[:4])
        np.testing.assert_array_equal(repaired[34:], baseline[34:])
        old_phase = infer_control_phase(
            trajectory, control, tags, strong,
            {**config, "phase_observed_reception_bridge_enabled": False},
        )
        old_evidence = Evidence(
            team_scores, np.zeros(len(team_scores), dtype=np.float32),
            strong_control=strong, control_phase=old_phase,
        )
        np.testing.assert_array_equal(decode(old_evidence, tags, {
            **config, "phase_observed_reception_bridge_enabled": False,
        }), baseline)

    def test_explicit_contact_flight_turnover_reception_and_restart(self):
        trajectory, control, tags, strong, config = self._fixture()
        phases = infer_control_phase(trajectory, control, tags, strong, config)
        self.assertEqual(phases.kind[2], CONTACT)
        self.assertEqual(phases.kind[4], FLIGHT)
        self.assertEqual(phases.team[4], HOME)
        self.assertEqual(phases.anchor_rows[4], 3)
        self.assertEqual(phases.kind[7], TURNOVER)
        self.assertEqual(phases.team[7], AWAY)
        self.assertEqual(phases.kind[8], FLIGHT)
        self.assertEqual(phases.kind[11], RECEPTION)
        self.assertEqual(phases.team[11], AWAY)
        tags.dead_hint[8] = True
        stopped = infer_control_phase(trajectory, control, tags, strong, config)
        self.assertEqual(stopped.kind[8], STOPPAGE)
        self.assertEqual(stopped.kind[9], UNSEEN)
        self.assertEqual(stopped.kind[11], CONTACT)
        tags.restart_team[9] = HOME
        self.assertEqual(infer_control_phase(
            trajectory, control, tags, strong, config,
        ).kind[9], STOPPAGE)

    def test_early_opponent_veto_requires_observed_same_player_support(self):
        trajectory, control, tags, strong, config = self._fixture()
        config.update({
            "phase_contact_minimum_frames": 4,
            "phase_early_opponent_veto_enabled": True,
        })
        control.nearest_distances[5:8, AWAY, 0] = .5
        control.nearest_player_slot[5:8, AWAY] = 0
        strong[5:8, AWAY] = True
        phase = infer_control_phase(trajectory, control, tags, strong, config)
        self.assertTrue(phase.incipient_contact[7, AWAY])
        self.assertFalse(phase.incipient_contact[6, AWAY])
        trajectory.source[6] = 0
        interrupted = infer_control_phase(
            trajectory, control, tags, strong, config
        )
        self.assertFalse(interrupted.incipient_contact[7, AWAY])

    def test_missing_ball_and_identity_switch_invalidate_old_owner(self):
        trajectory, control, tags, strong, config = self._fixture()
        trajectory.source[4] = 2
        phases = infer_control_phase(trajectory, control, tags, strong, config)
        self.assertEqual(phases.kind[4], UNSEEN)
        self.assertEqual(phases.kind[5], UNSEEN)
        self.assertEqual(phases.kind[7], CONTACT)
        control.nearest_player_slot[3, HOME] = 1
        supported = sustained_player_contacts(
            trajectory.source, strong, control.nearest_player_slot, 3,
        )
        self.assertFalse(supported[3, HOME])
        with self.assertRaisesRegex(ValueError, "aligned player slots"):
            sustained_player_contacts(trajectory.source, strong, None, 2)

    def test_expiry_ambiguity_and_period_isolation(self):
        trajectory, control, tags, strong, config = self._fixture()
        config["phase_flight_max_frames"] = 2
        phase = infer_control_phase(trajectory, control, tags, strong, config)
        self.assertEqual(phase.kind[4], FLIGHT)
        self.assertEqual(phase.kind[5], FLIGHT)
        self.assertEqual(phase.kind[7], CONTACT)
        isolated = infer_control_phase(
            SimpleNamespace(source=trajectory.source[4:8], speed=trajectory.speed[4:8]),
            SimpleNamespace(nearest_distances=control.nearest_distances[4:8],
                            nearest_player_slot=control.nearest_player_slot[4:8]),
            empty_tags(4), strong[4:8], config,
        )
        self.assertEqual(isolated.kind[0], UNSEEN)
        control.nearest_distances[1:4, AWAY, 0] = .5
        control.nearest_player_slot[1:4, AWAY] = 0
        ambiguous = infer_control_phase(trajectory, control, tags, strong | (
            control.nearest_distances[:, :, 0] < 1
        ), config)
        self.assertEqual(ambiguous.kind[3], UNSEEN)
        self.assertEqual(ambiguous.kind[4], UNSEEN)

    def test_contested_ball_does_not_assert_flight_owner(self):
        trajectory, control, tags, strong, config = self._fixture()
        control.nearest_distances[4, AWAY, 0] = .4
        phase = infer_control_phase(trajectory, control, tags, strong, config)
        self.assertEqual(phase.kind[4], UNSEEN)
        self.assertEqual(phase.anchor_rows[4], -1)
        self.assertEqual(phase.kind[5], FLIGHT)

    def test_phase_config_and_decoder_validate_opt_in_evidence(self):
        config = default_config()
        config.update({
            "semi_markov_phase_enabled": True, "phase_contact_minimum_frames": 4,
            "phase_flight_max_frames": 20, "phase_minimum_ball_speed_mps": 2.,
            "phase_contact_distance_metres": 1., "phase_flight_weight": .25,
            "phase_turnover_transition_cost": 5.,
        })
        validate_control_phase_config(config)
        with self.assertRaisesRegex(ValueError, "aligned"):
            decode_semi_markov(Evidence(np.zeros((8, 2)), np.zeros(8)), config)
        phase = ControlPhase(np.full(8, UNSEEN, dtype=np.int8),
                             np.full(8, UNKNOWN, dtype=np.int8), {})
        phase.kind[3] = FLIGHT
        phase.team[3] = HOME
        phase.anchor_rows = np.full(8, -1, dtype=np.int32)
        phase.anchor_rows[3] = 2
        evidence = Evidence(np.zeros((8, 2)), np.zeros(8), control_phase=phase)
        plain = _semi_markov_emissions(evidence, {**config, "semi_markov_phase_enabled": False})
        actual = _semi_markov_emissions(evidence, config)
        self.assertAlmostEqual(actual[3, HOME] - plain[3, HOME], .25)
        np.testing.assert_array_equal(actual[:, DEAD], plain[:, DEAD])
        phase.team[3] = UNKNOWN
        with self.assertRaisesRegex(ValueError, "valid control phase"):
            decode_semi_markov(evidence, config)
        for key, invalid in (
            ("phase_contact_minimum_frames", 1),
            ("phase_flight_max_frames", 0),
            ("phase_flight_weight", float("nan")),
            ("phase_turnover_transition_cost", 31.),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_control_phase_config({**config, key: invalid})
        with self.assertRaisesRegex(ValueError, "semi-Markov"):
            validate_control_phase_config({**config, "decoder_mode": "hsmm"})
        with self.assertRaisesRegex(ValueError, "cannot combine"):
            validate_control_phase_config({**config, "semi_markov_owner_flight_enabled": True})
        scaled = config_at_fps(config, 25, 50)
        self.assertEqual(scaled["phase_contact_minimum_frames"], 8)
        self.assertEqual(scaled["phase_flight_max_frames"], 40)
        self.assertEqual(scaled["phase_flight_weight"], .25)
        self.assertEqual(scaled["phase_turnover_transition_cost"], 10.)
        bridging = {
            **config, "phase_decode_mode": "bounded",
            "phase_observed_reception_bridge_enabled": True,
            "phase_observed_reception_bridge_max_frames": 50,
        }
        validate_control_phase_config(bridging)
        self.assertEqual(config_at_fps(bridging, 25, 50)[
            "phase_observed_reception_bridge_max_frames"
        ], 100)
        for invalid in (
            {**bridging, "phase_observed_reception_bridge_max_frames": 20},
            {**bridging, "phase_decode_mode": "global"},
            {**bridging, "strong_control_relative_speed_metres_per_second": float("nan")},
        ):
            with self.assertRaises(ValueError):
                validate_control_phase_config(invalid)

    def test_confirmed_turnover_only_affects_team_switch_not_dead(self):
        config = default_config()
        config.update({
            "semi_markov_phase_enabled": True, "phase_contact_minimum_frames": 4,
            "phase_flight_max_frames": 20, "phase_minimum_ball_speed_mps": 2.,
            "phase_contact_distance_metres": 1., "phase_flight_weight": 0.,
            "phase_turnover_transition_cost": 0.,
            "semi_markov_minimum_live_frames": 5,
        })
        control = np.zeros((50, 2), dtype=np.float32)
        control[:40, HOME] = 1.
        control[40:, AWAY] = 1.
        phase = ControlPhase(
            np.full(50, UNSEEN, dtype=np.int8),
            np.full(50, UNKNOWN, dtype=np.int8), {},
            np.full(50, -1, dtype=np.int32),
        )
        phase.kind[40] = TURNOVER
        phase.team[40] = AWAY
        evidence = Evidence(control, np.zeros(50, dtype=np.float32),
                            control_phase=phase)
        no_phase = decode_semi_markov(
            evidence, {**config, "semi_markov_phase_enabled": False},
        )
        with_phase = decode_semi_markov(evidence, config)
        self.assertTrue(np.all(no_phase == HOME))
        self.assertTrue(np.all(with_phase[:40] == HOME))
        self.assertTrue(np.all(with_phase[40:] == AWAY))

    def test_decoded_phase_anchor_filters_disagreeing_flights_without_mutating_input(self):
        config = default_config()
        config.update({
            "semi_markov_phase_enabled": True, "phase_confirm_decoded_owner": True,
            "phase_contact_minimum_frames": 4, "phase_flight_max_frames": 20,
            "phase_minimum_ball_speed_mps": 2., "phase_contact_distance_metres": 1.,
            "phase_flight_weight": 2., "phase_turnover_transition_cost": 10.,
        })
        control = np.zeros((50, 2), dtype=np.float32)
        control[:, HOME] = 1.
        phase = ControlPhase(
            np.full(50, UNSEEN, dtype=np.int8),
            np.full(50, UNKNOWN, dtype=np.int8), {},
            np.full(50, -1, dtype=np.int32),
        )
        phase.kind[12:22] = FLIGHT
        phase.team[12:22] = AWAY
        phase.anchor_rows[12:22] = 11
        evidence = Evidence(control, np.zeros(50, dtype=np.float32), control_phase=phase)
        confirmed = decode(evidence, empty_tags(50), config)
        np.testing.assert_array_equal(confirmed, decode_semi_markov(
            evidence, {**config, "semi_markov_phase_enabled": False},
        ))
        np.testing.assert_array_equal(phase.kind[12:22], np.full(10, FLIGHT))

    def test_bounded_phases_keep_baseline_outside_completed_flight(self):
        config = default_config()
        config.update({
            "semi_markov_phase_enabled": True, "phase_decode_mode": "bounded",
            "phase_contact_minimum_frames": 4, "phase_flight_max_frames": 20,
            "phase_minimum_ball_speed_mps": 2., "phase_contact_distance_metres": 1.5,
            "phase_flight_weight": 4., "phase_turnover_transition_cost": 8.,
            "semi_markov_minimum_live_frames": 5,
            "semi_markov_live_transition_cost": 8.,
            "semi_markov_strong_transition_cost": 8.,
        })
        control = np.zeros((90, 2), dtype=np.float32)
        control[:, HOME] = 1.
        control[20:32, HOME] = 0.
        control[20:32, AWAY] = 1.
        control[60:75, HOME] = 0.
        control[60:75, AWAY] = 1.
        phase = ControlPhase(
            np.full(90, UNSEEN, dtype=np.int8),
            np.full(90, UNKNOWN, dtype=np.int8), {},
            np.full(90, -1, dtype=np.int32),
        )
        phase.kind[15] = CONTACT
        phase.team[15] = HOME
        phase.kind[16:34] = FLIGHT
        phase.team[16:34] = HOME
        phase.anchor_rows[16:34] = 15
        phase.kind[34] = RECEPTION
        phase.team[34] = HOME
        evidence = Evidence(control, np.zeros(90, dtype=np.float32), control_phase=phase)
        baseline = decode_semi_markov(evidence, {**config, "semi_markov_phase_enabled": False})
        bounded = decode(evidence, empty_tags(90), config)
        self.assertTrue(np.all(baseline[20:32] == AWAY))
        self.assertTrue(np.all(bounded[20:32] == HOME))
        np.testing.assert_array_equal(bounded[:16], baseline[:16])
        np.testing.assert_array_equal(bounded[34:], baseline[34:])
        self.assertTrue(np.all(bounded[60:75] == AWAY))
        self.assertEqual(evidence.control_phase.diagnostics["bounded_baseline_agreeing_brackets"], 1)
        self.assertEqual(evidence.control_phase.diagnostics["bounded_changed_live_frames"], 12)

    def test_bounded_phase_abstains_at_incipient_opponent_contact(self):
        config = default_config()
        config.update({
            "semi_markov_phase_enabled": True,
            "phase_decode_mode": "bounded",
            "phase_early_opponent_veto_enabled": True,
            "phase_contact_minimum_frames": 4,
            "phase_flight_max_frames": 20,
            "phase_minimum_ball_speed_mps": 2.,
            "phase_contact_distance_metres": 1.5,
            "phase_flight_weight": 4.,
            "phase_turnover_transition_cost": 8.,
            "semi_markov_minimum_live_frames": 5,
            "semi_markov_live_transition_cost": 8.,
            "semi_markov_strong_transition_cost": 8.,
        })
        control = np.zeros((90, 2), dtype=np.float32)
        control[:, HOME] = 1.
        control[20:32, HOME] = 0.
        control[20:32, AWAY] = 1.
        phase = ControlPhase(
            np.full(90, UNSEEN, dtype=np.int8),
            np.full(90, UNKNOWN, dtype=np.int8), {},
            np.full(90, -1, dtype=np.int32),
            np.zeros((90, 2), dtype=bool),
        )
        phase.kind[15] = CONTACT
        phase.team[15] = HOME
        phase.kind[16:34] = FLIGHT
        phase.team[16:34] = HOME
        phase.anchor_rows[16:34] = 15
        phase.kind[34] = RECEPTION
        phase.team[34] = HOME
        phase.incipient_contact[23:26, AWAY] = True
        evidence = Evidence(
            control, np.zeros(90, dtype=np.float32),
            control_phase=phase,
        )
        baseline = decode_semi_markov(
            evidence, {**config, "semi_markov_phase_enabled": False,
                       "phase_early_opponent_veto_enabled": False}
        )
        result = decode(evidence, empty_tags(90), config)
        np.testing.assert_array_equal(result[23:26], baseline[23:26])
        self.assertEqual(
            phase.diagnostics["incipient_opponent_veto_frames"], 3
        )
        self.assertFalse(np.any(result == DEAD))
        with self.assertRaisesRegex(ValueError, "boolean flag"):
            validate_control_phase_config({
                **config, "phase_early_opponent_veto_enabled": "yes"
            })
        with self.assertRaisesRegex(ValueError, "bounded"):
            validate_control_phase_config({
                **config, "phase_decode_mode": "global"
            })

    def test_bounded_phase_preserves_supported_baseline_handover(self):
        config = default_config()
        config.update({
            "semi_markov_phase_enabled": True,
            "phase_decode_mode": "bounded",
            "phase_preserve_supported_switch_enabled": True,
            "phase_contact_minimum_frames": 4,
            "phase_flight_max_frames": 20,
            "phase_minimum_ball_speed_mps": 2.,
            "phase_contact_distance_metres": 1.5,
            "phase_flight_weight": 4.,
            "phase_turnover_transition_cost": 8.,
            "semi_markov_minimum_live_frames": 5,
            "semi_markov_live_transition_cost": 8.,
            "semi_markov_strong_transition_cost": 8.,
            "semi_markov_strong_transition_window_frames": 20,
            "semi_markov_strong_transition_minimum_frames": 8,
        })
        control = np.zeros((90, 2), dtype=np.float32)
        control[:, HOME] = 1.
        control[20:32, HOME] = 0.
        control[20:32, AWAY] = 1.
        strong = np.zeros((90, 2), dtype=bool)
        strong[20:28, AWAY] = True
        phase = ControlPhase(
            np.full(90, UNSEEN, dtype=np.int8),
            np.full(90, UNKNOWN, dtype=np.int8), {},
            np.full(90, -1, dtype=np.int32),
        )
        phase.kind[15] = CONTACT
        phase.team[15] = HOME
        phase.kind[16:34] = FLIGHT
        phase.team[16:34] = HOME
        phase.anchor_rows[16:34] = 15
        phase.kind[34] = RECEPTION
        phase.team[34] = HOME
        evidence = Evidence(
            control, np.zeros(90, dtype=np.float32),
            strong_control=strong, control_phase=phase,
        )
        baseline = decode_semi_markov(
            evidence, {**config, "semi_markov_phase_enabled": False,
                       "phase_preserve_supported_switch_enabled": False}
        )
        self.assertTrue(np.all(baseline[20:32] == AWAY))
        protected = decode(evidence, empty_tags(90), config)
        np.testing.assert_array_equal(protected[20:25], baseline[20:25])
        self.assertEqual(
            phase.diagnostics["bounded_supported_switches_preserved"], 1
        )
        with self.assertRaisesRegex(ValueError, "boolean flag"):
            validate_control_phase_config({
                **config, "phase_preserve_supported_switch_enabled": "yes"
            })
        with self.assertRaisesRegex(ValueError, "bounded"):
            validate_control_phase_config({
                **config, "phase_decode_mode": "global"
            })

    def test_bounded_phases_abstain_without_agreeing_endpoint_or_flight(self):
        config = default_config()
        config.update({
            "semi_markov_phase_enabled": True, "phase_decode_mode": "bounded",
            "phase_contact_minimum_frames": 4, "phase_flight_max_frames": 20,
            "phase_minimum_ball_speed_mps": 2., "phase_contact_distance_metres": 1.5,
            "phase_flight_weight": 4., "phase_turnover_transition_cost": 0.,
        })
        control = np.zeros((65, 2), dtype=np.float32)
        control[:, HOME] = 1.
        phase = ControlPhase(
            np.full(65, UNSEEN, dtype=np.int8),
            np.full(65, UNKNOWN, dtype=np.int8), {},
            np.full(65, -1, dtype=np.int32),
        )
        phase.kind[15] = CONTACT
        phase.team[15] = HOME
        phase.kind[16:26] = FLIGHT
        phase.team[16:26] = HOME
        phase.anchor_rows[16:26] = 15
        phase.kind[26] = TURNOVER
        phase.team[26] = AWAY
        evidence = Evidence(control, np.zeros(65, dtype=np.float32), control_phase=phase)
        baseline = decode_semi_markov(evidence, {**config, "semi_markov_phase_enabled": False})
        np.testing.assert_array_equal(decode(evidence, empty_tags(65), config), baseline)
        self.assertEqual(phase.diagnostics["bounded_baseline_agreeing_brackets"], 0)
        phase.kind[26] = UNSEEN
        phase.team[26] = UNKNOWN
        np.testing.assert_array_equal(decode(evidence, empty_tags(65), config), baseline)
        self.assertEqual(phase.diagnostics["bounded_source_brackets"], 0)
        with self.assertRaisesRegex(ValueError, "must be global or bounded"):
            validate_control_phase_config({**config, "phase_decode_mode": "unknown"})
        with self.assertRaisesRegex(ValueError, "already require agreement"):
            validate_control_phase_config({**config, "phase_confirm_decoded_owner": True})
        with self.assertRaisesRegex(ValueError, "aligned H/A/D"):
            decode_semi_markov(evidence, config, fixed_labels=np.zeros(4, dtype=np.int8))

    def test_bounded_phases_preserve_dead_spells_and_minimum_durations(self):
        config = default_config()
        config.update({
            "semi_markov_phase_enabled": True, "phase_decode_mode": "bounded",
            "phase_contact_minimum_frames": 4, "phase_flight_max_frames": 20,
            "phase_minimum_ball_speed_mps": 2., "phase_contact_distance_metres": 1.5,
            "phase_flight_weight": 4., "phase_turnover_transition_cost": 8.,
            "semi_markov_minimum_live_frames": 5,
            "semi_markov_minimum_dead_frames": 5,
            "semi_markov_live_transition_cost": 8.,
            "semi_markov_strong_transition_cost": 8.,
        })
        control = np.zeros((90, 2), dtype=np.float32)
        control[:, HOME] = 1.
        dead = np.zeros(90, dtype=np.float32)
        dead[25:31] = 20.
        phase = ControlPhase(
            np.full(90, UNSEEN, dtype=np.int8),
            np.full(90, UNKNOWN, dtype=np.int8), {},
            np.full(90, -1, dtype=np.int32),
        )
        phase.kind[15] = CONTACT
        phase.team[15] = HOME
        phase.kind[16:36] = FLIGHT
        phase.team[16:36] = HOME
        phase.anchor_rows[16:36] = 15
        phase.kind[36] = RECEPTION
        phase.team[36] = HOME
        evidence = Evidence(control, dead, control_phase=phase)
        baseline = decode_semi_markov(evidence, {**config, "semi_markov_phase_enabled": False})
        bounded = decode(evidence, empty_tags(90), config)
        self.assertTrue(np.all(baseline[25:31] == DEAD))
        np.testing.assert_array_equal(baseline == DEAD, bounded == DEAD)
        starts = np.r_[0, np.flatnonzero(bounded[1:] != bounded[:-1]) + 1]
        ends = np.r_[starts[1:], len(bounded)]
        for start, end in zip(starts, ends):
            self.assertGreaterEqual(
                end - start,
                config["semi_markov_minimum_dead_frames"] if bounded[start] == DEAD
                else config["semi_markov_minimum_live_frames"],
            )
        impossible = np.full(90, HOME, dtype=np.int8)
        impossible[10:12] = AWAY
        with self.assertRaisesRegex(ValueError, "no duration-valid path"):
            decode_semi_markov(
                evidence, {**config, "semi_markov_phase_enabled": False},
                fixed_labels=impossible,
            )


class ObservedHandoverTests(unittest.TestCase):
    def _config(self):
        return {
            **default_config(),
            "semi_markov_observed_handover_enabled": True,
            "observed_handover_motion_window_frames": 4,
            "observed_handover_motion_minimum": .25,
        }

    def test_requires_observed_dominant_player_and_ball_turn(self):
        config = self._config()
        n = 52
        ball = np.column_stack((
            np.arange(n, dtype=np.float32) * .1,
            np.maximum(np.arange(n) - 19, 0) * .2,
        )).astype(np.float32)
        trajectory = repair_ball_trajectory(ball, 25)
        home = (ball + np.array([-4., 0.], dtype=np.float32))[:, None, :]
        away = (ball + np.array([4., 0.], dtype=np.float32))[:, None, :]
        away[20, 0] = ball[20] + np.array([.1, 0.], dtype=np.float32)
        control = control_features_from_arrays(home, away, trajectory.xy, 25, config)
        tags = empty_tags(n)
        evidence = build_evidence(control, trajectory, tags, (105., 68.), config, fps=25)
        self.assertTrue(evidence.observed_handover_support[20, AWAY])
        self.assertFalse(evidence.observed_handover_support[19, AWAY])
        without = _semi_markov_strong_transition_support(
            evidence, {**config, "semi_markov_observed_handover_enabled": False},
        )
        self.assertFalse(without[20, AWAY])
        self.assertTrue(_semi_markov_strong_transition_support(evidence, config)[20, AWAY])

        trajectory.source[20] = 0
        self.assertFalse(build_evidence(
            control, trajectory, tags, (105., 68.), config, fps=25,
        ).observed_handover_support[20, AWAY])
        with self.assertRaisesRegex(ValueError, "positive physical fps"):
            build_evidence(control, trajectory, tags, (105., 68.), config)

    def test_only_discounts_a_supported_live_team_switch(self):
        config = self._config()
        config.update({
            "semi_markov_minimum_live_frames": 5,
            "semi_markov_live_transition_cost": 30.,
            "semi_markov_strong_transition_cost": 10.,
        })
        n = 90
        control = np.zeros((n, 2), dtype=np.float32)
        control[:, HOME] = 1.
        control[20:44, HOME] = 0.
        control[20:44, AWAY] = 1.
        support = np.zeros((n, 2), dtype=bool)
        support[20, AWAY] = True
        evidence = Evidence(
            control, np.zeros(n, dtype=np.float32),
            strong_control=np.zeros((n, 2), dtype=bool),
            observed_handover_support=support,
        )
        disabled = {**config, "semi_markov_observed_handover_enabled": False}
        baseline = decode_semi_markov(evidence, disabled)
        self.assertTrue(np.all(baseline == HOME))
        candidate = decode_semi_markov(evidence, config)
        self.assertTrue(np.all(candidate[20:44] == AWAY))
        self.assertTrue(np.all(candidate[:20] == HOME))
        self.assertTrue(np.all(candidate[44:] == HOME))
        self.assertFalse(np.any(candidate == DEAD))
        evidence.observed_handover_support = None
        with self.assertRaisesRegex(ValueError, "aligned boolean"):
            decode_semi_markov(evidence, config)

    def test_config_is_explicit_and_scales_only_frame_counts(self):
        config = self._config()
        validate_observed_handover_config(config)
        self.assertEqual(config_at_fps(config, 25, 50)[
            "observed_handover_motion_window_frames"
        ], 8)
        for invalid in (
            {"semi_markov_observed_handover_enabled": "yes"},
            {"observed_handover_motion_window_frames": 0},
            {"observed_handover_motion_minimum": 0},
            {"observed_handover_motion_minimum": float("nan")},
            {"semi_markov_phase_enabled": True},
            {"semi_markov_local_handover_enabled": True},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_observed_handover_config({**config, **invalid})


class ContactCarryTests(unittest.TestCase):
    def _config(self):
        return {
            **default_config(),
            "semi_markov_contact_carry_enabled": True,
            "contact_carry_window_frames": 5,
            "contact_carry_maximum_frames": 50,
            "contact_carry_support_frames": 2,
            "contact_carry_distance_metres": .75,
            "contact_carry_turn_minimum": .3,
            "contact_carry_minimum_ball_speed_mps": 1.,
            "contact_carry_weight": .5,
        }

    def _fixture(self, config, turn=True, n=80):
        steps = np.zeros((n, 2), dtype=np.float32)
        steps[1:, 0] = .2
        if turn:
            steps[31:] = (0., .2)
        ball = np.cumsum(steps, axis=0) + np.array([30., 30.], dtype=np.float32)
        trajectory = repair_ball_trajectory(ball, 25)
        home = (ball + np.array([0., -4.], dtype=np.float32))[:, None, :]
        away = (ball + np.array([0., 4.], dtype=np.float32))[:, None, :]
        away[30, 0] = ball[30]
        control = control_features_from_arrays(home, away, trajectory.xy, 25, config)
        return trajectory, control

    def test_turning_touch_carries_until_a_later_touch_could_not_be_seen(self):
        config = self._config()
        trajectory, control = self._fixture(config)
        trajectory.source[60:] = 0
        carry, support, quality = contact_carry_evidence(control, trajectory, 25, config)
        self.assertEqual(quality["contact_carry_touches"], 1)
        start = int(np.flatnonzero(carry[:, AWAY])[0])
        self.assertTrue(26 <= start <= 31)
        self.assertTrue(support[start, AWAY])
        self.assertEqual(carry[54, AWAY], .5)
        self.assertFalse(carry[55:].any())
        self.assertFalse(carry[:, HOME].any())

        evidence = build_evidence(control, trajectory, empty_tags(80), (105., 68.), config, fps=25)
        disabled = {**config, "semi_markov_contact_carry_enabled": False}
        np.testing.assert_allclose(
            _semi_markov_emissions(evidence, config)[:, AWAY]
            - _semi_markov_emissions(evidence, disabled)[:, AWAY],
            carry[:, AWAY],
        )
        self.assertTrue(_semi_markov_strong_transition_support(evidence, config)[start, AWAY])

    def test_video_ball_reveals_touches_only_when_opted_in(self):
        config = self._config()
        trajectory, control = self._fixture(config)
        trajectory.source[:] = 4
        _, _, quality = contact_carry_evidence(control, trajectory, 25, config)
        self.assertEqual(quality["contact_carry_touches"], 0)
        _, _, quality = contact_carry_evidence(
            control, trajectory, 25, {**config, "contact_carry_include_video_ball": True},
        )
        self.assertEqual(quality["contact_carry_touches"], 1)
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            validate_contact_carry_config({**config, "contact_carry_include_video_ball": 1})

    def test_close_player_without_ball_turn_is_not_a_touch(self):
        config = self._config()
        trajectory, control = self._fixture(config, turn=False)
        carry, support, quality = contact_carry_evidence(control, trajectory, 25, config)
        self.assertEqual(quality["contact_carry_touches"], 0)
        self.assertFalse(carry.any() or support.any())

    def test_video_team_counts_as_a_touch_for_that_team(self):
        from possession.decoder import video_team_evidence
        config = {"contact_carry_weight": 1.0, "contact_carry_support_frames": 1}
        carry = np.zeros((6, 2), dtype=np.float32)
        carry[:, HOME] = 1.0
        support = np.zeros((6, 2), dtype=bool)
        new_carry, new_support = video_team_evidence(carry, support, ([3], [AWAY]), config)
        self.assertEqual(new_carry[3].tolist(), [0.0, 1.0])
        self.assertEqual(new_support[:, AWAY].tolist(), [False, False, True, True, True, False])
        self.assertTrue((carry[:, HOME] == 1.0).all())
        with self.assertRaisesRegex(ValueError, "in-range"):
            video_team_evidence(carry, support, ([9], [AWAY]), config)

    def test_video_team_pass_penalises_the_other_team_without_a_dead_column(self):
        from possession.decoder import video_team_pass_evidence
        config = {"contact_carry_weight": 1.0, "contact_carry_support_frames": 0}
        penalty = np.zeros((4, 3), dtype=np.float32)
        penalty[2, AWAY] = 0.7  # the ball-free model disfavours away here; the video overrides it
        new_penalty, new_support = video_team_pass_evidence(
            penalty, np.zeros((4, 2), dtype=bool), ([2], [AWAY]), 4, config)
        self.assertEqual(new_penalty[2].tolist(), [1.0, 0.0, 0.0])
        self.assertTrue((new_penalty[:, 2] == 0).all())
        self.assertEqual(new_support[:, AWAY].tolist(), [False, False, True, False])
        fresh, _ = video_team_pass_evidence(None, None, ([1], [HOME]), 4, config)
        self.assertEqual(fresh[1].tolist(), [0.0, 1.0, 0.0])

    def test_rejects_invalid_contact_carry_config(self):
        validate_contact_carry_config({"semi_markov_contact_carry_enabled": False})
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            validate_contact_carry_config({"semi_markov_contact_carry_enabled": 1})
        with self.assertRaisesRegex(ValueError, "at most 2"):
            validate_contact_carry_config({**self._config(), "contact_carry_turn_minimum": 2.5})
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            validate_contact_carry_config({**self._config(), "contact_carry_weight": 0})


class BroadcastAlignmentTests(unittest.TestCase):
    def _config(self):
        return {
            "broadcast_stoppage_enabled": True, "broadcast_minimum_players": 4,
            "broadcast_offset_samples": 40, "broadcast_search_stride_frames": 1,
            "broadcast_match_distance_metres": 1.0, "broadcast_offset_tolerance_frames": 1,
            "broadcast_minimum_offset_votes": 3, "broadcast_offset_smoothing_frames": 50,
            "broadcast_maximum_bracket_frames": 100, "broadcast_stoppage_minimum_frames": 10,
            "broadcast_stoppage_dead_weight": .9, "broadcast_stoppage_ball_speed_mps": 12.,
        }

    def _views(self):
        rng = np.random.default_rng(0)
        n = 400
        steps = rng.normal(0, .3, (n, 6, 2))
        tactical = np.cumsum(steps, axis=0) + rng.uniform(10, 60, (1, 6, 2))
        team = np.asarray([0, 0, 0, 1, 1, 1])
        # Broadcast: first block at offset +20, an edit, second block at offset +150.
        broadcast = np.concatenate([tactical[20:120], tactical[150:330]])
        valid = np.ones(len(broadcast), dtype=bool)
        valid[40:70] = False   # cutaway inside the first block
        valid[95:100] = False  # cutaway that runs into the edit: not bracketed
        broadcast[~valid] = np.nan
        return tactical, team, broadcast, valid

    def test_discovers_offsets_and_only_trusts_bracketed_cutaways(self):
        from possession.broadcast import NO_PITCH, UNCOVERED, align_broadcast
        tactical, team, broadcast, valid = self._views()
        mapped, state, diagnostics = align_broadcast(tactical, team, broadcast, team, valid, self._config())
        self.assertEqual(diagnostics["broadcast_offsets"], [20, 50])
        self.assertTrue(np.all(state[60:90] == NO_PITCH))
        self.assertTrue(np.all(state[115:150] == UNCOVERED))
        self.assertTrue(np.all(state[:20] == UNCOVERED))
        self.assertEqual(mapped[0], 20)
        self.assertEqual(mapped[150], 200)

    def test_detects_replays_of_earlier_play(self):
        from possession.broadcast import LIVE_VIEW, REPLAY, align_broadcast
        tactical, team, broadcast, valid = self._views()
        broadcast[75:90] = tactical[30:45]  # pitch shown, but an earlier moment
        valid[75:90] = True
        config = {**self._config(), "broadcast_replay_enabled": True, "broadcast_replay_stride_frames": 1,
                  "broadcast_replay_lookback_frames": 150, "broadcast_replay_minimum_lag_frames": 10}
        _, state, diagnostics = align_broadcast(tactical, team, broadcast, team, valid, config)
        self.assertTrue(np.all(state[95:110] == REPLAY))
        self.assertTrue(np.all(state[20:60] == LIVE_VIEW))
        self.assertEqual(diagnostics["broadcast_replay_frames"], 15)

    def test_cutaway_stoppage_ends_when_the_ball_moves_fast(self):
        from possession.broadcast import NO_PITCH, broadcast_stoppage_intervals
        n = 100
        ball = np.zeros((n, 2), dtype=np.float32)
        ball[60:, 0] = np.arange(40) * 1.0  # 25 m/s from frame 60
        trajectory = repair_ball_trajectory(ball, 25)
        view = np.zeros(n, dtype=np.int8)
        view[20:90] = NO_PITCH
        dead, quality = broadcast_stoppage_intervals(view, trajectory, empty_tags(n), 25, self._config())
        self.assertTrue(dead[20:60].all())
        self.assertFalse(dead[66:].any() or dead[:20].any())
        self.assertEqual(quality["broadcast_cutaways_truncated"], 1)

    def test_broadcast_ball_fills_only_plausible_live_view_gaps(self):
        from possession.broadcast import LIVE_VIEW, OTHER_VIEW, broadcast_ball_observations
        n = 40
        mapped = np.arange(n) + 5
        view = np.full(n + 5, LIVE_VIEW, dtype=np.int8)
        view[30:35] = OTHER_VIEW
        ball = np.column_stack((np.arange(n) * 0.2, np.full(n, 30.0)))
        players = np.full((n, 3, 2), np.nan)
        players[:, 0] = ball + (0.5, 0.0)
        players[10:15, 0] = ball[10:15] + (5.0, 0.0)   # nobody near the ball
        ball[20, 0] = 60.0                              # implausible jump
        source = np.zeros(n + 5, dtype=np.int8)
        source[:10] = 1                                 # tactical ball already seen
        config = {"broadcast_ball_player_distance_metres": 1.5, "broadcast_ball_maximum_speed_mps": 15.0,
                  "max_ball_speed_metres_per_second": 40.0}
        frames, xy, confidence = broadcast_ball_observations(mapped, view, ball, players, source, 25, config)
        # Broadcast frames kept: not seen tactically, live view, a player close, no jump nearby.
        kept = np.r_[5:10, 15:18, 19, 21, 23:25, 30:38]
        np.testing.assert_array_equal(frames, kept + 5)
        np.testing.assert_allclose(xy, ball[kept])
        self.assertTrue((confidence == 1).all())


class BallFreePossessionTests(unittest.TestCase):
    def _teams(self, n=200):
        rng = np.random.default_rng(1)
        home = rng.uniform(20, 60, (n, 11, 2))
        away = rng.uniform(45, 85, (n, 11, 2))
        home[:, 0] = (2., 34.)   # home goalkeeper near x = 0
        away[:, 0] = (103., 34.)
        return home, away

    def test_attack_direction_follows_goalkeepers(self):
        from possession.ballfree import attack_directions
        home, away = self._teams()
        self.assertTrue(np.all(attack_directions(home, away, [(0, 200)], (105., 68.)) == 1))
        self.assertTrue(np.all(attack_directions(away, home, [(0, 200)], (105., 68.)) == -1))

    def test_features_flip_sign_when_teams_swap(self):
        from possession.ballfree import attack_directions, ballfree_features
        home, away = self._teams()
        home[:, 1:, 0] += np.linspace(0, 5, 200)[:, None]  # home pushes forward
        forward = ballfree_features(home, away, attack_directions(home, away, [(0, 200)], (105., 68.)), 25)
        swapped = ballfree_features(away, home, attack_directions(away, home, [(0, 200)], (105., 68.)), 25)
        np.testing.assert_allclose(forward, -swapped, atol=1e-9)

    def test_penalty_only_inside_long_unseen_stretches(self):
        config = {**default_config(), "semi_markov_ballfree_enabled": True, "ballfree_weight": 1.,
                  "ballfree_support_lean": .5, "ballfree_minimum_unseen_frames": 50,
                  "ballfree_coefficients": [1., 0., 0., 0., 0.]}
        n = 200
        ball = np.full((n, 2), 30., dtype=np.float32)
        ball[20:40] = np.nan    # short gap
        ball[100:180] = np.nan  # long gap
        trajectory = repair_ball_trajectory(ball, 25, max_gap_frames=0)
        home = (ball * 0 + np.array([28., 30.], dtype=np.float32))[:, None, :]
        away = (ball * 0 + np.array([40., 30.], dtype=np.float32))[:, None, :]
        home[:], away[:] = (28., 30.), (40., 30.)
        control = control_features_from_arrays(home, away, trajectory.xy, 25, config)
        probability = np.full(n, .1)
        evidence = build_evidence(control, trajectory, empty_tags(n), (105., 68.), config, fps=25,
                                  ballfree_home_probability=probability)
        unseen = trajectory.source == 0
        self.assertFalse(evidence.ballfree_penalty[:100].any())
        self.assertTrue(unseen[100:180].sum() >= 50)
        self.assertTrue(np.all(evidence.ballfree_penalty[unseen & (np.arange(n) >= 100), HOME] > 0))
        self.assertFalse(evidence.ballfree_penalty[~unseen].any() or evidence.ballfree_penalty[:, AWAY].any())
        self.assertTrue(evidence.ballfree_support[unseen & (np.arange(n) >= 100), AWAY].all())

    def test_seen_weight_applies_only_where_no_player_is_near_the_ball(self):
        config = {**default_config(), "semi_markov_ballfree_enabled": True, "ballfree_weight": 1.,
                  "ballfree_support_lean": .5, "ballfree_minimum_unseen_frames": 50,
                  "ballfree_coefficients": [1., 0., 0., 0., 0.],
                  "ballfree_seen_weight": .5, "ballfree_seen_min_distance_metres": 3.}
        n = 100
        ball = np.full((n, 2), 30., dtype=np.float32)
        home = np.tile(np.array([[[31., 30.]]], dtype=np.float32), (n, 1, 1))  # 1 m from the ball
        away = np.tile(np.array([[[60., 30.]]], dtype=np.float32), (n, 1, 1))
        home[50:] = (40., 30.)                                                  # 10 m away from frame 50
        trajectory = repair_ball_trajectory(ball, 25, max_gap_frames=0)
        control = control_features_from_arrays(home, away, trajectory.xy, 25, config)
        evidence = build_evidence(control, trajectory, empty_tags(n), (105., 68.), config, fps=25,
                                  ballfree_home_probability=np.full(n, .1))
        self.assertFalse(evidence.ballfree_penalty[:50].any())
        np.testing.assert_allclose(evidence.ballfree_penalty[50:, HOME], .5 * .8, rtol=1e-6)
        self.assertFalse(evidence.ballfree_support[:, AWAY].any())
        moving = build_evidence(control, trajectory, empty_tags(n), (105., 68.),
                                {**config, "ballfree_seen_max_ball_speed_mps": -1.}, fps=25,
                                ballfree_home_probability=np.full(n, .1))
        self.assertFalse(moving.ballfree_penalty.any())

    def test_sequence_model_is_antisymmetric(self):
        from possession.ballfree_seq import sequence_home_probability
        home, away = self._teams(400)
        players = np.concatenate([home, away], axis=1)
        team = np.r_[np.zeros(11, dtype=int), np.ones(11, dtype=int)]
        path = str(ROOT / "models" / "ballfree_seq_cnn3.npz")
        p = sequence_home_probability(players, team, (105., 68.), 25, [(0, 400)], path)
        q = sequence_home_probability(players, 1 - team, (105., 68.), 25, [(0, 400)], path)
        self.assertEqual(p.shape, (400,))
        np.testing.assert_allclose(p + q, 1.0, atol=1e-6)

    def test_ball_aware_sequence_model_reads_ball_and_gaps_scope_keeps_players_only(self):
        from possession.ballfree import configured_home_probability
        from possession.ballfree_seq import sequence_home_probability
        home, away = self._teams(400)
        players = np.concatenate([home, away], axis=1)
        team = np.r_[np.zeros(11, dtype=int), np.ones(11, dtype=int)]
        path = str(ROOT / "models" / "ballfree_seq_ball_cnn3.npz")
        ball = home[:, 0].copy()
        ball[150:300] = np.nan
        p = sequence_home_probability(players, team, (105., 68.), 25, [(0, 400)], path, ball)
        q = sequence_home_probability(players, 1 - team, (105., 68.), 25, [(0, 400)], path, ball)
        np.testing.assert_allclose(p + q, 1.0, atol=1e-6)
        at_away = sequence_home_probability(players, team, (105., 68.), 25, [(0, 400)], path, away[:, 0])
        self.assertGreater(np.abs(p - at_away).max(), 1e-3)
        with self.assertRaisesRegex(ValueError, "need the ball trajectory"):
            sequence_home_probability(players, team, (105., 68.), 25, [(0, 400)], path)
        source = np.where(np.isfinite(ball).all(axis=1), 1, 0).astype(np.int8)
        trajectory = SimpleNamespace(xy=ball, source=source)
        config = {"ballfree_model": "sequence_ball", "ballfree_ball_weights_path": path,
                  "ballfree_ball_scope": "gaps", "ballfree_weights_path": "models/ballfree_seq_cnn3.npz",
                  "ballfree_minimum_unseen_frames": 125}
        gaps = configured_home_probability(players, team, (105., 68.), 25, [(0, 400)], config, trajectory=trajectory)
        players_only = sequence_home_probability(players, team, (105., 68.), 25, [(0, 400)],
                                                 str(ROOT / "models" / "ballfree_seq_cnn3.npz"))
        np.testing.assert_allclose(gaps[150:300], p[150:300])
        np.testing.assert_allclose(gaps[:150], players_only[:150])

    def test_rejects_invalid_ballfree_config(self):
        from possession.ballfree import validate_ballfree_config
        with self.assertRaisesRegex(ValueError, "one finite value per feature"):
            validate_ballfree_config({"semi_markov_ballfree_enabled": True, "ballfree_coefficients": [1.]})


class PlayerReliabilityTests(unittest.TestCase):
    def test_plausible_same_slot_short_gap_retains_age_without_filling_position(self):
        baseline = default_config()
        n = 20
        ball = np.tile([5., 5.], (n, 1))
        home = np.tile([[[5., 5.]]], (n, 1, 1))
        away = np.tile([[[30., 5.]]], (n, 1, 1))
        home[10:15] = np.nan
        no_resume = control_features_from_arrays(home, away, ball, 25, baseline)
        resume_config = {**baseline, "player_reliability_resume_gap_frames": 5}
        resumed = control_features_from_arrays(home, away, ball, 25, resume_config)
        self.assertEqual(no_resume.nearest_player_age_frames[15, HOME], 1)
        self.assertEqual(resumed.nearest_player_age_frames[15, HOME], 11)
        self.assertEqual(resumed.nearest_player_age_frames[13, HOME], 0)
        self.assertFalse(np.isfinite(resumed.nearest_player_xy[13, HOME]).any())
        home[15, 0, 0] = 50
        impossible = control_features_from_arrays(home, away, ball, 25, resume_config)
        self.assertEqual(impossible.nearest_player_age_frames[15, HOME], 1)
        home[15, 0, 0] = 5
        home[9] = np.nan
        too_long = control_features_from_arrays(home, away, ball, 25, resume_config)
        self.assertEqual(too_long.nearest_player_age_frames[15, HOME], 1)

    def test_atd_and_array_age_resume_match_with_same_slot_gaps(self):
        n = 12
        home = np.tile([[[5., 5.]]], (n, 1, 1))
        away = np.tile([[[30., 5.]]], (n, 1, 1))
        ball = np.tile([5., 5.], (n, 1))
        home[8:11] = np.nan
        config = {**default_config(), "player_reliability_resume_gap_frames": 3}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.txt"
            path.write_text("".join(
                "{}:{};30,5,0:5,5,0\n".format(
                    frame + 1, "NaN,NaN,NaN" if 8 <= frame < 11 else "5,5,0"
                ) for frame in range(n)
            ))
            raw = extract_control_features(
                path, TrackingMetadata(np.asarray([HOME, AWAY]), ("H", "A")),
                ball, (1, 1), fps=25, player_reliability_resume_gap_frames=3,
            )
        direct = control_features_from_arrays(home, away, ball, 25, config)
        np.testing.assert_array_equal(raw.nearest_player_age_frames, direct.nearest_player_age_frames)

    def test_new_near_ball_track_loses_only_opt_in_short_age_evidence(self):
        baseline = default_config()
        enabled = {**baseline, "player_reliability_enabled": True,
                   "player_reliability_minimum_age_frames": 8}
        n = 14
        ball = np.tile([5., 5.], (n, 1)).astype(np.float32)
        home = np.tile([[[5.2, 5.]]], (n, 1, 1)).astype(np.float32)
        home[:3] = np.nan
        away = np.tile([[[30., 5.]]], (n, 1, 1)).astype(np.float32)
        control = control_features_from_arrays(home, away, ball, 25, baseline)
        self.assertEqual(control.nearest_player_age_frames[3, HOME], 1)
        self.assertEqual(control.nearest_player_age_frames[10, HOME], 8)
        trajectory = BallTrajectory(
            xy=ball, velocity=np.zeros_like(ball), speed=np.zeros(n),
            confidence=np.ones(n), source=np.ones(n, dtype=np.int8),
        )
        standard = build_evidence(control, trajectory, empty_tags(n), (105, 68), baseline)
        cautious = build_evidence(control, trajectory, empty_tags(n), (105, 68), enabled)
        self.assertLess(cautious.control[3, HOME], standard.control[3, HOME])
        self.assertFalse(cautious.strong_control[3, HOME])
        self.assertTrue(cautious.strong_control[10, HOME])
        self.assertAlmostEqual(cautious.control[10, HOME], standard.control[10, HOME])
        self.assertGreater(cautious.player_reliability_diagnostics["young_near_ball_team_frames"], 0)
        trajectory.source[5] = 2
        unchanged_unobserved = build_evidence(control, trajectory, empty_tags(n), (105, 68), enabled)
        np.testing.assert_array_equal(unchanged_unobserved.control[5], standard.control[5])

    def test_player_reliability_rejects_missing_age_and_invalid_modes(self):
        config = {**default_config(), "player_reliability_enabled": True,
                  "player_reliability_minimum_age_frames": 8}
        for values, expected in (
            ({"player_reliability_enabled": "yes"}, "must be boolean"),
            ({"player_reliability_minimum_age_frames": 1}, "at least two"),
            ({"decoder_mode": "hsmm"}, "requires semi-Markov"),
            ({"player_reliability_resume_gap_frames": -1}, "nonnegative integer"),
        ):
            with self.assertRaisesRegex(ValueError, expected):
                validate_player_reliability_config({**config, **values})
        n = 8
        home = np.tile([[[5., 5.]]], (n, 1, 1))
        away = np.tile([[[9., 5.]]], (n, 1, 1))
        ball = np.tile([5., 5.], (n, 1))
        control = control_features_from_arrays(home, away, ball, 25, config)
        control.nearest_player_age_frames = None
        trajectory = BallTrajectory(
            ball, np.zeros_like(ball), np.zeros(n), np.ones(n), np.ones(n, dtype=np.int8),
        )
        with self.assertRaisesRegex(ValueError, "aligned nearest-player track ages"):
            build_evidence(control, trajectory, empty_tags(n), (105, 68), config)

    def test_reliability_selector_uses_original_frozen_soft_candidate(self):
        baseline = load_config(ROOT / "configs" / "compliant_tracking.json")
        soft = {"semi_markov_live_transition_cost": 20.}
        parity = {"baseline_sha256": "baseline", "protocol_sha256": "protocol"}
        with mock.patch("possession.external_temporal_parity.select_touch_control",
                        return_value=(soft, {}, {"diagnosis_sha256": "development"})), \
                mock.patch("possession.external_temporal_parity.load_frozen_diagnostic",
                           return_value={"candidate": "observed_touch_motion_softening"}):
            values, _, selection = select_player_reliability(
                ROOT, ROOT / "configs" / "metrica_player_reliability.json", baseline, parity, soft,
            )
        self.assertTrue(values["player_reliability_enabled"])
        self.assertEqual(values["player_reliability_minimum_age_frames"],
                         baseline["semi_markov_strong_transition_minimum_frames"])
        self.assertEqual(selection["contact_threshold_provenance"], "development")

    def test_age_resume_rejects_changed_external_stress_policy(self):
        baseline = load_config(ROOT / "configs" / "compliant_tracking.json")
        soft = {"semi_markov_live_transition_cost": 20.}
        parity = {"baseline_sha256": "baseline", "protocol_sha256": "protocol"}
        variant = json.loads(
            (ROOT / "configs" / "metrica_player_reliability_resume.json").read_text()
        )
        variant["source_player_stress_protocol_sha256"] = "stale"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stale_resume.json"
            path.write_text(json.dumps(variant))
            with mock.patch("possession.external_temporal_parity.select_touch_control",
                            return_value=(soft, {}, {"diagnosis_sha256": "development"})), \
                    mock.patch("possession.external_temporal_parity.load_frozen_diagnostic",
                               return_value={"candidate": "observed_touch_motion_softening"}):
                with self.assertRaisesRegex(ValueError, "frozen short-gap corruption protocol"):
                    select_player_reliability(
                        ROOT, ROOT / "configs" / "metrica_player_reliability.json",
                        baseline, parity, soft, path,
                    )


class OwnerFlightTests(unittest.TestCase):
    def test_owner_frozen_diagnostic_requires_matching_candidate_values(self):
        baseline = load_config(ROOT / "configs" / "compliant_tracking.json")
        soft = {**baseline, "semi_markov_owner_flight_enabled": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owner.json"
            path.write_text(json.dumps({
                "values": soft, "baseline_sha256": "baseline",
                "protocol_sha256": "protocol", "runtime_source_fps": 25,
                "results": [{"game": "Sample_Game_1", "candidate": {"safe": False}}],
            }))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            report = load_frozen_diagnostic(path, digest, soft, "baseline", "protocol")
            self.assertFalse(report["results"][0]["candidate"]["safe"])
            with self.assertRaisesRegex(ValueError, "candidate or deployment policy"):
                load_frozen_diagnostic(path, digest, {**soft, "owner_flight_weight": .35},
                                       "baseline", "protocol")

    def _config(self):
        return {
            **default_config(),
            "semi_markov_owner_flight_enabled": True, "owner_flight_horizon_frames": 4,
            "owner_flight_weight": .35, "owner_flight_minimum_ball_speed_mps": 2.,
            "owner_flight_touch_motion_minimum": .1, "touch_motion_window_frames": 2,
        }

    def test_owner_anchors_and_flight_stop_on_contact_or_restart(self):
        n = 11
        config = self._config()
        nearest = np.full((n, 2, 1), 4., dtype=np.float32)
        nearest[1, HOME, 0] = .5
        nearest[4, AWAY, 0] = .5
        nearest[7, HOME, 0] = .5
        control = SimpleNamespace(nearest_distances=nearest)
        trajectory = SimpleNamespace(
            xy=np.column_stack((np.arange(n, dtype=np.float32), np.zeros(n, dtype=np.float32))),
            source=np.ones(n, dtype=np.int8), speed=np.full(n, 5., dtype=np.float32),
        )
        tags = empty_tags(n)
        tags.dead_hint[8] = True
        strong = nearest[:, :, 0] <= 1
        disturbance = np.zeros((n, 2), dtype=np.float32)
        disturbance[1, HOME] = 1.
        disturbance[7, HOME] = 1.
        with mock.patch("possession.decoder.contact_motion_features",
                        return_value={"contact_motion_disturbance": disturbance}):
            prior, quality, anchors = owner_flight_evidence(trajectory, control, tags, strong, 25, config)
        self.assertAlmostEqual(prior[2, HOME], .35)
        self.assertAlmostEqual(prior[4, HOME], .35)
        self.assertEqual(prior[4, AWAY], 0)
        self.assertEqual(prior[6, HOME], 0)
        self.assertAlmostEqual(prior[9].sum(), 0)
        self.assertEqual(quality["trusted_contact_frames"], 2)
        self.assertEqual(quality["owner_flight_prior_frames"], 4)
        self.assertEqual(anchors[2], 1)
        self.assertEqual(anchors[8], -1)
        config["owner_flight_reset_on_gap"] = True
        trajectory.source[3] = 2
        with mock.patch("possession.decoder.contact_motion_features",
                        return_value={"contact_motion_disturbance": disturbance}):
            gap_prior, gap_quality, _ = owner_flight_evidence(trajectory, control, tags, strong, 25, config)
        self.assertAlmostEqual(gap_prior[2, HOME], .35)
        self.assertEqual(gap_prior[4, HOME], 0)
        self.assertEqual(gap_quality["unobserved_gap_resets"], 1)

    def test_unknown_or_absent_contact_does_not_invent_an_owner(self):
        config = self._config()
        n = 8
        nearest = np.ones((n, 2, 1), dtype=np.float32)
        control = SimpleNamespace(nearest_distances=nearest)
        trajectory = SimpleNamespace(
            xy=np.zeros((n, 2), dtype=np.float32),
            source=np.ones(n, dtype=np.int8), speed=np.full(n, 5., dtype=np.float32),
        )
        strong = np.ones((n, 2), dtype=bool)
        motion = np.full((n, 2), np.nan, dtype=np.float32)
        with mock.patch("possession.decoder.contact_motion_features",
                        return_value={"contact_motion_disturbance": motion}):
            prior, quality, _ = owner_flight_evidence(trajectory, control, empty_tags(n), strong, 25, config)
        self.assertFalse(prior.any())
        self.assertEqual(quality["trusted_contact_frames"], 0)
        with self.assertRaisesRegex(ValueError, "explicit runtime fps"):
            build_evidence(
                control_features_from_arrays(
                    np.zeros((n, 1, 2)), np.ones((n, 1, 2)), trajectory.xy,
                    25, config,
                ),
                BallTrajectory(
                    trajectory.xy, np.zeros((n, 2)), trajectory.speed, np.ones(n), trajectory.source,
                ),
                empty_tags(n), (105, 68), config,
            )

    def test_owner_feature_rejects_bad_config_and_scales_only_horizon(self):
        config = self._config()
        scaled = config_at_fps(config, 25, 50)
        self.assertEqual(scaled["owner_flight_horizon_frames"], 8)
        self.assertEqual(scaled["owner_flight_weight"], .35)
        for values, message in (
            ({"owner_flight_horizon_frames": 0}, "positive integer"),
            ({"owner_flight_weight": np.nan}, "finite and positive"),
            ({"owner_flight_touch_motion_minimum": 1.1}, "exceed one"),
            ({"owner_flight_reset_on_gap": "yes"}, "must be boolean"),
            ({"owner_flight_confirm_decoded_owner": "yes"}, "must be boolean"),
            ({"owner_flight_require_sustained_player": True}, "at least two consecutive"),
            ({"owner_flight_require_same_team_reception": True}, "requires gap reset"),
            ({"decoder_mode": "state_machine"}, "requires semi-Markov"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                validate_owner_flight_config({**config, **values})

    def test_owner_flight_requires_same_team_contact_after_pass(self):
        config = {**self._config(), "owner_flight_reset_on_gap": True,
                  "owner_flight_require_same_team_reception": True}
        n = 9
        nearest = np.full((n, 2, 1), 4., dtype=np.float32)
        nearest[1, HOME, 0] = .5
        nearest[5, AWAY, 0] = .5
        control = SimpleNamespace(nearest_distances=nearest)
        trajectory = SimpleNamespace(
            xy=np.column_stack((np.arange(n, dtype=np.float32), np.zeros(n, dtype=np.float32))),
            source=np.ones(n, dtype=np.int8), speed=np.full(n, 5., dtype=np.float32),
        )
        disturbance = np.ones((n, 2), dtype=np.float32)
        with mock.patch("possession.decoder.contact_motion_features",
                        return_value={"contact_motion_disturbance": disturbance}):
            prior, quality, _ = owner_flight_evidence(
                trajectory, control, empty_tags(n), nearest[:, :, 0] < 1, 25, config,
            )
        self.assertFalse(prior.any())
        self.assertEqual(quality["unconfirmed_flight_prior_frames"], 6)
        nearest[5, AWAY, 0] = 4
        nearest[5, HOME, 0] = .5
        with mock.patch("possession.decoder.contact_motion_features",
                        return_value={"contact_motion_disturbance": disturbance}):
            prior, quality, _ = owner_flight_evidence(
                trajectory, control, empty_tags(n), nearest[:, :, 0] < 1, 25, config,
            )
        self.assertAlmostEqual(prior[2:5, HOME].sum(), 3 * config["owner_flight_weight"])
        self.assertEqual(quality["unconfirmed_flight_prior_frames"], 3)
        trajectory.source[3] = 2
        with mock.patch("possession.decoder.contact_motion_features",
                        return_value={"contact_motion_disturbance": disturbance}):
            prior, quality, _ = owner_flight_evidence(
                trajectory, control, empty_tags(n), nearest[:, :, 0] < 1, 25, config,
            )
        self.assertFalse(prior.any())
        self.assertGreater(quality["unconfirmed_flight_prior_frames"], 0)

    def test_sustained_owner_uses_same_player_not_just_team_contact(self):
        config = {**self._config(), "owner_flight_require_sustained_player": True,
                  "owner_flight_contact_minimum_frames": 3}
        n = 9
        nearest = np.full((n, 2, 1), 4., dtype=np.float32)
        nearest[1:5, HOME, 0] = .5
        slots = np.full((n, 2), -1, dtype=np.int16)
        slots[1:3, HOME] = 0
        slots[3:5, HOME] = 1
        nearest[6, AWAY, 0] = .5
        slots[6, AWAY] = 0
        control = SimpleNamespace(nearest_distances=nearest, nearest_player_slot=slots)
        trajectory = SimpleNamespace(
            xy=np.column_stack((np.arange(n, dtype=np.float32), np.zeros(n, dtype=np.float32))),
            source=np.ones(n, dtype=np.int8), speed=np.full(n, 5., dtype=np.float32),
        )
        strong = nearest[:, :, 0] <= 1
        disturbance = np.ones((n, 2), dtype=np.float32)
        with mock.patch("possession.decoder.contact_motion_features",
                        return_value={"contact_motion_disturbance": disturbance}):
            prior, quality, _ = owner_flight_evidence(trajectory, control, empty_tags(n), strong, 25, config)
        self.assertEqual(quality["trusted_contact_frames"], 0)
        self.assertFalse(prior.any())
        slots[1:5, HOME] = 0
        with mock.patch("possession.decoder.contact_motion_features",
                        return_value={"contact_motion_disturbance": disturbance}):
            prior, quality, _ = owner_flight_evidence(trajectory, control, empty_tags(n), strong, 25, config)
        self.assertEqual(quality["trusted_contact_frames"], 2)
        self.assertAlmostEqual(prior[5, HOME], .35)

    def test_decoded_owner_confirmation_rejects_conflicting_priors_only(self):
        config = {
            **self._config(), "owner_flight_confirm_decoded_owner": True,
            "semi_markov_minimum_live_frames": 2, "semi_markov_minimum_dead_frames": 2,
            "semi_markov_control_margin_weight": 0.0, "semi_markov_live_transition_cost": 10.,
            "semi_markov_strong_transition_cost": 5., "semi_markov_dead_emission_threshold": 100.,
            "semi_markov_start_dead_bonus": 0.,
        }
        n = 8
        evidence = Evidence(
            control=np.tile([1., 0.], (n, 1)), dead=np.zeros(n),
            strong_control=np.zeros((n, 2), dtype=bool),
            owner_flight_prior=np.zeros((n, 2), dtype=np.float32),
            owner_flight_anchor_rows=np.full(n, -1, dtype=np.int64),
            owner_flight_diagnostics={"owner_flight_prior_frames": 2},
        )
        evidence.owner_flight_prior[2, AWAY] = config["owner_flight_weight"]
        evidence.owner_flight_prior[3, HOME] = config["owner_flight_weight"]
        evidence.owner_flight_anchor_rows[2:4] = 1
        labels = decode(evidence, empty_tags(n), config)
        np.testing.assert_array_equal(labels, np.full(n, HOME))
        self.assertEqual(evidence.owner_flight_diagnostics["first_pass_prior_frames_rejected"], 1)
        self.assertEqual(evidence.owner_flight_diagnostics["first_pass_prior_frames_retained"], 1)
        evidence.owner_flight_anchor_rows[2] = n
        with self.assertRaisesRegex(ValueError, "valid prior contact rows"):
            decode(evidence, empty_tags(n), config)

    def test_owner_selector_is_frozen_to_soft_contact_source(self):
        project = ROOT
        config = load_config(project / "configs" / "compliant_tracking.json")
        previous = {"baseline_sha256": "baseline", "protocol_sha256": "protocol"}
        core = {key: config[key] for key in CORE_CALIBRATION_KEYS}
        soft = {**core, "core_reliability_mode": "transitions",
                "semi_markov_touch_motion_gate_enabled": True, "touch_motion_mode": "soften"}
        with mock.patch("possession.external_temporal_parity.select_touch_control",
                        return_value=(soft, {}, {"threshold": .01, "diagnosis_sha256": "diagnosis"})), \
                mock.patch("possession.external_temporal_parity.load_frozen_diagnostic",
                           return_value={"candidate": "observed_touch_motion_softening"}) as frozen:
            values, source, selection = select_owner_flight(
                project, project / "configs" / "metrica_owner_flight.json",
                config, previous, core,
            )
        self.assertEqual(frozen.call_args.args[1], "e9a1eec68969ca325f7ba0ca0a23dddca20837e9bcdec7a50f4888b5f70ce8a1")
        self.assertEqual(source["candidate"], "observed_touch_motion_softening")
        self.assertEqual(values["owner_flight_weight"], config["restart_weight"])
        self.assertEqual(values["owner_flight_horizon_frames"],
                         config["semi_markov_strong_transition_window_frames"])
        self.assertEqual(values["owner_flight_touch_motion_minimum"],
                         selection["touch_motion_minimum"])


class TouchControlTests(unittest.TestCase):
    def test_official_range_diagnostics_leave_primary_guard_metrics_unchanged(self):
        values = {.3: .4, .4: .4, .5: .6, .6: .4, .7: .4, .75: .2}
        score = {
            "reference_conditioned_spell_scores": {str(iou): {"f1": value} for iou, value in values.items()},
            "known_live_macro_f1": .5,
        }
        prepared = {
            "data": {"fps": 25, "periods": np.ones(4)},
            "truth": np.zeros(4), "primary_references": {},
            "references": {"recovery_references": []},
        }
        case = {"is_live_stress": False}
        protocol = {"spell_ious": [.5, .75], "diagnostic_spell_ious": [.3, .4, .5, .6, .7],
                    "recovery_scoring": {}}
        with mock.patch("possession.external_guarded_calibration.score_possession_references",
                        return_value=score), \
                mock.patch("possession.external_guarded_calibration.score_recovery_transitions",
                           return_value={}):
            result = evaluate_case(np.zeros(4, dtype=np.int8), prepared, case, protocol)
        self.assertEqual(result["possession"]["reference_conditioned_spell_scores"],
                         {"0.5": {"f1": .6}, "0.75": {"f1": .2}})
        self.assertAlmostEqual(result["official_style_reference_spell"]["mean_f1"], .44)
        self.assertFalse(result["official_style_reference_spell"]["is_challenge_score"])

    def test_failed_development_diagnostic_requires_unchanged_frozen_candidate(self):
        frozen = {
            "values": {"value": 1}, "baseline_sha256": "baseline", "protocol_sha256": "protocol",
            "runtime_source_fps": 25,
            "results": [{"game": "Sample_Game_1", "candidate": {"safe": False}}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            path.write_text(json.dumps(frozen))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result = load_frozen_diagnostic(path, digest, {"value": 1}, "baseline", "protocol")
            self.assertFalse(result["results"][0]["candidate"]["safe"])
            with self.assertRaisesRegex(ValueError, "candidate or deployment policy"):
                load_frozen_diagnostic(path, digest, {"value": 2}, "baseline", "protocol")
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "unchanged frozen"):
                load_frozen_diagnostic(path, digest, {"value": 1}, "baseline", "protocol")

    def test_softening_changes_only_known_observed_transition_cost(self):
        config = {
            **self._config(), "touch_motion_mode": "soften",
            "touch_motion_weak_transition_cost": 9.,
            "semi_markov_live_transition_cost": 10.,
            "semi_markov_strong_transition_cost": 1.,
            "semi_markov_minimum_live_frames": 2, "semi_markov_minimum_dead_frames": 2,
            "semi_markov_strong_transition_window_frames": 4,
            "semi_markov_strong_transition_minimum_frames": 1,
            "semi_markov_control_margin_weight": 0., "minimum_control_score": 0.,
            "semi_markov_dead_emission_threshold": 100., "semi_markov_start_dead_bonus": 0.,
        }
        control = np.asarray([[2, 0]] * 4 + [[0, 2]] * 4, dtype=np.float32)
        expected = np.asarray([HOME] * 4 + [AWAY] * 4)
        for mode in ("soften", "confidence"):
            current = {**config, "touch_motion_mode": mode}
            evidence = Evidence(
                control=control, dead=np.zeros(8), strong_control=np.ones((8, 2), dtype=bool),
                reliable_observation=np.ones(8, dtype=bool), touch_motion_disturbance=np.zeros((8, 2)),
            )
            self.assertTrue(_semi_markov_strong_transition_support(evidence, current).all())
            self.assertEqual(len(np.unique(decode_semi_markov(evidence, current))), 1)
            evidence.reliable_observation[4] = False
            np.testing.assert_array_equal(decode_semi_markov(evidence, current), expected)
            evidence.reliable_observation[4] = True
            evidence.touch_motion_disturbance[4] = np.nan
            np.testing.assert_array_equal(decode_semi_markov(evidence, current), expected)
            evidence.touch_motion_disturbance[4] = 1
            np.testing.assert_array_equal(decode_semi_markov(evidence, current), expected)

    def _config(self):
        return {
            **default_config(), "semi_markov_touch_motion_gate_enabled": True,
            "touch_motion_horizon_frames": 20, "touch_motion_window_frames": 4,
            "touch_motion_minimum": .1,
        }

    def test_motion_gate_preserves_uncertain_and_unknown_frames(self):
        config = {**self._config(), "semi_markov_strong_transition_window_frames": 4,
                  "semi_markov_strong_transition_minimum_frames": 1}
        observed = np.asarray([True, False, True, True, True, True])
        motion = np.asarray([[0, 0], [0, 0], [np.nan, np.nan], [.5, .5], [.5, .5], [.5, .5]])
        evidence = Evidence(
            control=np.zeros((6, 2)), dead=np.zeros(6),
            strong_control=np.ones((6, 2), dtype=bool),
            reliable_observation=observed, touch_motion_disturbance=motion,
        )
        support = _semi_markov_strong_transition_support(evidence, config)
        self.assertFalse(support[0].any())
        self.assertTrue(support[1:].all())
        self.assertEqual(touch_motion_summary(evidence, config), {
            "observed_team_frames": 10, "available_team_frames": 8,
            "unknown_team_frames": 2, "low_motion_team_frames": 2,
        })
        evidence.touch_motion_disturbance = None
        with self.assertRaisesRegex(ValueError, "aligned observation and motion"):
            _semi_markov_strong_transition_support(evidence, config)

    def test_evidence_builder_requires_explicit_fps_and_is_opt_in(self):
        config = self._config()
        xy = np.column_stack((np.arange(20) * .5, np.zeros(20))).astype(np.float32)
        trajectory = BallTrajectory(
            xy=xy, velocity=np.tile([5., 0.], (20, 1)), speed=np.full(20, 5.),
            confidence=np.ones(20), source=np.ones(20, dtype=np.int8),
        )
        home = np.tile([[[4., 0.]]], (20, 1, 1))
        away = np.tile([[[40., 0.]]], (20, 1, 1))
        control = control_features_from_arrays(home, away, xy, 10, config)
        tags = empty_tags(20)
        with self.assertRaisesRegex(ValueError, "explicit runtime fps"):
            build_evidence(control, trajectory, tags, (105, 68), config)
        evidence = build_evidence(control, trajectory, tags, (105, 68), config, fps=10)
        self.assertAlmostEqual(evidence.touch_motion_disturbance[3, HOME], 0)
        disabled = {**config, "semi_markov_touch_motion_gate_enabled": False}
        original = build_evidence(control, trajectory, tags, (105, 68), disabled)
        explicit = build_evidence(control, trajectory, tags, (105, 68), disabled, fps=10)
        self.assertIsNone(original.touch_motion_disturbance)
        np.testing.assert_array_equal(original.control, explicit.control)
        np.testing.assert_array_equal(original.dead, explicit.dead)

    def test_touch_config_validation_and_frame_rate_scaling(self):
        config = self._config()
        scaled = config_at_fps(config, 25, 50)
        self.assertEqual(scaled["touch_motion_horizon_frames"], 40)
        self.assertEqual(scaled["touch_motion_window_frames"], 8)
        self.assertEqual(scaled["touch_motion_minimum"], config["touch_motion_minimum"])
        softened = config_at_fps({**config, "touch_motion_weak_transition_cost": 20.}, 25, 50)
        self.assertEqual(softened["touch_motion_weak_transition_cost"], 40.)
        for values, message in [
            ({"semi_markov_touch_motion_gate_enabled": "yes"}, "must be boolean"),
            ({"decoder_mode": "state_machine"}, "requires semi-Markov"),
            ({"touch_motion_minimum": np.nan}, "finite"),
            ({"touch_motion_window_frames": 0}, "positive integer"),
            ({"semi_markov_gap_handover_enabled": True}, "cannot be combined"),
            ({"touch_motion_mode": "unknown"}, "gate, soften or confidence"),
            ({"touch_motion_mode": "soften"}, "weak cost"),
            ({"touch_motion_mode": "confidence", "touch_motion_minimum": 0}, "must be positive"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                validate_touch_motion_config({**config, **values})
        with self.assertRaisesRegex(ValueError, "requires semi-Markov"):
            decode(Evidence(np.zeros((4, 2)), np.zeros(4)), empty_tags(4),
                   {**config, "decoder_mode": "state_machine"})

    def test_handover_runner_passes_game_fps_explicitly(self):
        game = {
            "fps": 25, "periods": np.ones(4), "prepared": {"clean": [(None, None, None)]},
            "truth": None, "references": {"recovery_references": []}, "targeted_recoveries": [],
        }
        possession = {
            "reference_conditioned_spell_scores": {"0.5": {"f1": 0.}},
            "known_live_macro_f1": 0., "known_frame_macro_f1": 0.,
        }
        recovery = {"same_direction_boundary_recall": {"0.5": 0.}, "accuracy_at_recovery": 0.}
        protocol = {
            "source_fps": 25, "recovery_scoring": {},
            "selection_weights": {"reference_spell_f1": 1., "boundary_recall": 0.,
                                  "recovery_accuracy": 0., "live_macro_f1": 0.},
        }
        with mock.patch("possession.external_handover_calibration.build_evidence") as build, \
                mock.patch("possession.external_handover_calibration.decode_semi_markov",
                           return_value=np.zeros(4, dtype=np.int8)), \
                mock.patch("possession.external_handover_calibration.score_possession_references",
                           return_value=possession), \
                mock.patch("possession.external_handover_calibration.score_recovery_transitions",
                           return_value=recovery):
            evaluate_handover(game, default_config(), protocol)
        self.assertEqual(build.call_args.kwargs["fps"], 25)




class ExternalPredictionReselectionTests(unittest.TestCase):
    def test_decoder_selection_preserves_original_guards_and_gain_threshold(self):
        baseline = {"name": "current_25fps", "safe": True, "evaluation": {"objective": 0.4}}
        unsafe = {"name": "responsive", "safe": False, "evaluation": {"objective": 0.5}}
        marginal = {"name": "marginal", "safe": True, "evaluation": {"objective": 0.401}}
        useful = {"name": "useful", "safe": True, "evaluation": {"objective": 0.405}}
        self.assertIs(choose_decoder_policy([baseline, unsafe, marginal], 0.002), baseline)
        self.assertIs(choose_decoder_policy([baseline, unsafe, marginal, useful], 0.002), useful)
        with self.assertRaisesRegex(ValueError, "guarded 25-fps"):
            choose_decoder_policy([{"name": "not-current", "safe": True,
                                    "evaluation": {"objective": 0.4}}], 0.002)

    def test_selects_external_utility_with_conservative_tie_breaks(self):
        cases = [
            {"history_frames": 10, "horizon_frames": 20, "objective": 0.1},
            {"history_frames": 7, "horizon_frames": 10, "objective": 0.1},
            {"history_frames": 5, "horizon_frames": 10, "objective": 0.1},
            {"history_frames": 5, "horizon_frames": 5, "objective": 0.09},
        ]
        self.assertEqual(choose_prediction_policy(cases)["history_frames"], 5)
        self.assertEqual(choose_prediction_policy(cases)["horizon_frames"], 10)
        cases[-1]["objective"] = 0.11
        self.assertEqual(choose_prediction_policy(cases)["horizon_frames"], 5)
        with self.assertRaisesRegex(ValueError, "finite, positive"):
            choose_prediction_policy([])
        with self.assertRaisesRegex(ValueError, "finite, positive"):
            choose_prediction_policy([{"history_frames": 5, "horizon_frames": 10,
                                       "objective": float("nan")}])


class ExternalDFLValidationTests(unittest.TestCase):
    def test_spell_structure_audit_separates_missed_stoppages_and_team_switches(self):
        truth = np.asarray([
            HOME, HOME, DEAD, HOME, HOME, AWAY, AWAY, DEAD, AWAY, AWAY,
            AWAY, DEAD, AWAY, HOME, HOME, DEAD, HOME, HOME,
        ], dtype=np.int8)
        predicted = np.asarray([
            HOME, HOME, HOME, HOME, HOME, HOME, HOME, AWAY, AWAY, AWAY,
            AWAY, AWAY, AWAY, HOME, HOME, HOME, HOME, HOME,
        ], dtype=np.int8)
        ball = np.full((len(truth), 2), 10., dtype=np.float32)
        ball[7, 0] = 106.
        audit = audit_dfl_spell_structure(predicted, truth, [10, 8], ball, (105., 68.))
        self.assertEqual(audit["reference_live_spells"], 8)
        self.assertEqual(audit["predicted_live_spells"], 4)
        self.assertEqual(audit["reference_dead_spells_completely_missed"], 4)
        self.assertEqual(audit["reference_same_team_restart_pairs_merged"], 3)
        self.assertEqual(audit["reference_direct_live_switches"], 2)
        self.assertEqual(audit["predicted_direct_live_switches"], 2)
        self.assertEqual(audit["reference_direct_switches_correct_at_boundary"], 1)
        self.assertEqual(audit["reference_dead_frames_outside_pitch"], 1)
        with self.assertRaisesRegex(ValueError, "aligned two-half"):
            audit_dfl_spell_structure(predicted[:-1], truth, [10, 8], ball, (105., 68.))

    def test_reads_period_aligned_dfl_ball_and_player_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = root / "metadata.xml"
            positions = root / "positions.xml"
            info.write_text(
                '<PutDataRequest><MatchInformation>'
                '<General HomeTeamId="h" GuestTeamId="a"/>'
                '<Environment PitchX="105" PitchY="68"/>'
                '</MatchInformation></PutDataRequest>'
            )
            sets = []
            for name, first in (("firstHalf", 100), ("secondHalf", 200)):
                for team, person in (("referee", "official"), ("h", "home"), ("a", "away"), ("BALL", "ball")):
                    frames = []
                    for index in range(6):
                        status = 0 if index == 3 else 1
                        possession = 1 if index < 2 else 2
                        frames.append(
                            '<Frame N="{}" T="2023-01-01T00:00:00.{:03d}+00:00" '
                            'X="{}" Y="0" BallStatus="{}" BallPossession="{}"/>'.format(
                                first + index, index * 40,
                                1 if team == "h" else -1, status, possession,
                            )
                        )
                    sets.append('<FrameSet GameSection="{}" TeamId="{}" PersonId="{}">{}</FrameSet>'.format(
                        name, team, person, "".join(frames),
                    ))
            positions.write_text('<PutDataRequest><Positions>{}</Positions></PutDataRequest>'.format(
                "".join(sets),
            ))
            pitch, periods = load_dfl_match(info, positions)
            self.assertEqual(pitch, (105., 68.))
            self.assertEqual([len(item["ball"]) for item in periods], [6, 6])
            self.assertEqual(periods[0]["home"].shape, (6, 1, 2))
            np.testing.assert_array_equal(periods[0]["truth"], [HOME, HOME, AWAY, DEAD, AWAY, AWAY])
            truth = np.concatenate([item["truth"] for item in periods])
            score = score_dfl_labels(truth.copy(), truth, [6, 6])
            self.assertEqual(score["live_spell_f1"]["0.75"]["f1"], 1.)
            self.assertEqual(score["dead"]["dead_recall_on_known_frames"], 1.)
            with self.assertRaisesRegex(ValueError, "aligned model labels"):
                score_dfl_labels(truth[:-1], truth, [6, 6])
            reordered = sets[1:4] + sets[:1] + sets[5:8] + sets[4:5]
            positions.write_text(
                '<PutDataRequest><Positions>{}</Positions></PutDataRequest>'
                .format(''.join(reordered))
            )
            _, reordered_periods = load_dfl_match(info, positions)
            for original, updated in zip(periods, reordered_periods):
                np.testing.assert_array_equal(original["truth"], updated["truth"])
                np.testing.assert_array_equal(original["home"], updated["home"])
                np.testing.assert_array_equal(original["away"], updated["away"])
            without_referee = [item for index, item in enumerate(sets)
                               if index not in (0, 4)]
            positions.write_text(
                '<PutDataRequest><Positions>{}</Positions></PutDataRequest>'
                .format(''.join(without_referee))
            )
            _, without_referee_periods = load_dfl_match(info, positions)
            for original, updated in zip(periods, without_referee_periods):
                np.testing.assert_array_equal(original["truth"], updated["truth"])
                np.testing.assert_array_equal(original["home"], updated["home"])
                np.testing.assert_array_equal(original["away"], updated["away"])

    def test_rejects_undocumented_dfl_ball_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = root / "info.xml"
            positions = root / "positions.xml"
            info.write_text(
                '<PutDataRequest><MatchInformation><General HomeTeamId="h" GuestTeamId="a"/>'
                '<Environment PitchX="105" PitchY="68"/></MatchInformation></PutDataRequest>'
            )
            sets = []
            for name, first in (("firstHalf", 1), ("secondHalf", 10)):
                for team in ("referee", "h", "a", "BALL"):
                    frames = ''.join(
                        '<Frame N="{}" T="2023-01-01T00:00:00.{:03d}+00:00" X="0" Y="0" '
                        'BallStatus="{}" BallPossession="1"/>'.format(
                            first + index, index * 40,
                            3 if team == "BALL" else 1,
                        )
                        for index in range(2)
                    )
                    sets.append('<FrameSet GameSection="{}" TeamId="{}" PersonId="{}">{}</FrameSet>'.format(
                        name, team, team, frames,
                    ))
            positions.write_text('<PutDataRequest><Positions>{}</Positions></PutDataRequest>'.format(
                ''.join(sets),
            ))
            with self.assertRaisesRegex(ValueError, "documented codes"):
                load_dfl_match(info, positions)




class TrajectoryTests(unittest.TestCase):
    def test_fuses_video_observations_only_over_allowed_sources(self):
        raw = np.array([
            [0.0, 0.0],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [3.0, 0.0],
        ], dtype=np.float32)
        trajectory = repair_ball_trajectory(
            raw, fps=1, max_gap_frames=0, prediction_horizon_frames=0
        )
        fused = fuse_video_ball_observations(
            trajectory,
            [1, 3],
            [[1.0, 0.0], [30.0, 0.0]],
            [0.4, 0.4],
            fps=1,
            max_speed=45.0,
            replace_source_codes=(0,),
        )
        np.testing.assert_allclose(fused.xy[1], [1.0, 0.0])
        np.testing.assert_allclose(fused.xy[3], [3.0, 0.0])
        self.assertEqual(fused.source.tolist(), [1, 4, 0, 1])
        self.assertAlmostEqual(float(fused.confidence[1]), 0.4)

    def test_fused_joins_drop_contradicted_predictions_and_trim_at_observations(self):
        from possession.trajectory import BallTrajectory
        xy = np.array([[0, 0], [1, 0], [2, 0], [np.nan, np.nan], [np.nan, np.nan],
                       [np.nan, np.nan], [np.nan, np.nan], [60, 0]], dtype=np.float32)
        source = np.array([1, 3, 3, 0, 0, 0, 0, 1], dtype=np.int8)
        trajectory = BallTrajectory(xy=xy, velocity=np.zeros_like(xy), speed=np.zeros(8, np.float32),
                                    confidence=np.ones(8, np.float32), source=source)
        # Fused run 3-6 sits 38 m from the predictions and its far end jumps 30 m to frame 7.
        fused = fuse_video_ball_observations(
            trajectory, [3, 4, 5, 6], [[40, 0], [41, 0], [42, 0], [30, 0]], [1, 1, 1, 1],
            fps=1, max_speed=20.0,
        )
        self.assertEqual(fused.source.tolist(), [1, 0, 0, 4, 4, 4, 0, 1])
        self.assertTrue(np.isnan(fused.xy[[1, 2, 6]]).all())
        np.testing.assert_allclose(fused.xy[[0, 7]], [[0, 0], [60, 0]])

    def test_interpolates_short_physically_plausible_gap(self):
        raw = np.array(
            [[0.0, 0.0], [np.nan, np.nan], [2.0, 0.0]], dtype=np.float32
        )
        result = repair_ball_trajectory(
            raw, fps=10, max_gap_frames=2, max_speed=20.0
        )
        np.testing.assert_allclose(result.xy[1], [1.0, 0.0])
        self.assertEqual(result.source.tolist(), [1, 2, 1])
        self.assertAlmostEqual(float(result.confidence[1]), 0.55)

    def test_does_not_interpolate_implausible_gap(self):
        raw = np.array(
            [[0.0, 0.0], [np.nan, np.nan], [20.0, 0.0]], dtype=np.float32
        )
        result = repair_ball_trajectory(
            raw, fps=10, max_gap_frames=2, max_speed=20.0
        )
        self.assertTrue(np.isnan(result.xy[1]).all())
        self.assertEqual(result.source.tolist(), [1, 0, 1])

    def test_does_not_interpolate_long_gap(self):
        raw = np.full((7, 2), np.nan, dtype=np.float32)
        raw[0] = [0.0, 0.0]
        raw[-1] = [1.0, 0.0]
        result = repair_ball_trajectory(
            raw, fps=10, max_gap_frames=3, max_speed=20.0
        )
        self.assertTrue(np.isnan(result.xy[1:-1]).all())


class VideoTemporalPathTests(unittest.TestCase):
    def test_edge_anchored_diagnostic_decodes_only_ends_of_long_gap(self):
        class Capture:
            def __init__(self):
                self.position = 0
                self.read_count = 0

            def isOpened(self):
                return True

            def set(self, _property, frame):
                self.position = int(frame)

            def read(self):
                frame = self.position
                self.position += 1
                self.read_count += 1
                return True, frame

            def release(self):
                pass

        capture = Capture()
        cv2 = SimpleNamespace(
            VideoCapture=lambda _path: capture,
            CAP_PROP_POS_FRAMES=1,
        )
        ultralytics = SimpleNamespace(
            YOLO=lambda _path: SimpleNamespace(names={32: "sports ball"}),
            settings=SimpleNamespace(update=lambda _values: None),
        )
        source = np.zeros(202, dtype=np.int8)
        source[[0, 201]] = 1
        ball = np.full((202, 2), np.nan)
        ball[0] = [0.0, 34.0]
        ball[201] = [105.0, 34.0]

        def candidate(frame, *_args, **_kwargs):
            return [{
                "normalized_pitch": [frame / 201.0, 0.5],
                "confidence": 0.8,
            }]

        with mock.patch.dict(
            "sys.modules", {"cv2": cv2, "ultralytics": ultralytics}
        ), mock.patch(
            "possession.video.detect_tiled_ball_candidates",
            side_effect=candidate,
        ):
            reports = detect_interval_paths(
                "test.mp4", "test.pt",
                [SimpleNamespace(boundary_frame=50)],
                source, ball, [np.eye(3)] * 202,
                fps=25, pitch=(105.0, 68.0),
                class_id=32, confidence=0.03, image_size=1280,
                device="cpu", overlap_fraction=0.1,
                merge_radius_pixels=20, pitch_margin=0.02,
                maximum_gap_frames=5, maximum_speed=45,
                maximum_acceleration=120, confidence_reward=2,
                gap_penalty=1, speed_weight=0.25,
                acceleration_weight=0.5,
                path_mode="edge_anchored",
                kalman_maximum_missing_frames=38,
            )
        self.assertEqual(len(reports), 1)
        self.assertEqual(capture.read_count, 50)
        self.assertEqual(len(reports[0]["edge_tracklets"]), 2)
        self.assertFalse(any(
            26 <= point["frame"] <= 175
            for point in reports[0]["points"]
        ))

    def test_edge_anchored_paths_reject_unconnected_middle(self):
        for length in (200, 938):
            with self.subTest(gap_frames=length):
                candidates = []
                for frame in range(length):
                    if frame < 25 or frame >= length - 25:
                        candidates.append([{
                            "normalized_pitch": [
                                (frame + 1) / (length + 1), 0.5
                            ],
                            "confidence": 0.8,
                        }])
                    else:
                        candidates.append([])
                path = select_edge_anchored_ball_paths(
                    candidates, np.asarray([0.0, 34.0]),
                    np.asarray([105.0, 34.0]),
                    fps=25, pitch=(105.0, 68.0),
                    beam_width=24, maximum_missing_frames=5,
                    process_acceleration_std=15.0,
                    measurement_noise_base=0.5,
                    measurement_noise_minimum=0.1,
                    measurement_noise_maximum=2.0,
                    initial_velocity_std=30.0,
                    innovation_gate=4.0,
                    missing_penalty=3.0,
                    confidence_reward=2.0,
                )
                self.assertEqual(len(path.tracklets), 2)
                self.assertGreaterEqual(
                    np.count_nonzero(path.candidate_index >= 0), 40
                )
                self.assertLessEqual(
                    np.count_nonzero(path.candidate_index >= 0), 50
                )
                self.assertTrue(np.isnan(path.xy[25:length - 25]).all())
                self.assertTrue(
                    np.isfinite(path.xy[[0, length - 1]]).all()
                )
                self.assertEqual(
                    path.tracklets[-1]["anchor_side"], "right"
                )

    def test_tiled_pixel_detection_reuses_operational_pitch_filter(self):
        class FakeTensor:
            def __init__(self, values):
                self.values = np.asarray(values)

            def __getitem__(self, index):
                return FakeTensor(self.values[index])

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        model = mock.Mock()
        model.predict.return_value = [
            SimpleNamespace(boxes=SimpleNamespace(
                xywh=FakeTensor([[50.0, 48.0, 8.0, 8.0]]),
                conf=FakeTensor([0.4]),
            )),
            SimpleNamespace(boxes=SimpleNamespace(
                xywh=FakeTensor([[5.0, 48.0, 8.0, 8.0]]),
                conf=FakeTensor([0.9]),
            )),
            SimpleNamespace(boxes=None),
            SimpleNamespace(boxes=None),
        ]
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        options = {
            "class_id": 32, "confidence": 0.03, "image_size": 1280,
            "device": "cpu", "overlap_fraction": 0.1,
            "merge_radius_pixels": 20,
        }
        pixels = detect_tiled_ball_pixels(image, model, **options)
        projected = detect_tiled_ball_candidates(
            image, np.diag([100.0, 100.0, 1.0]), model,
            pitch_margin=0.02, **options
        )
        self.assertEqual(len(pixels), 1)
        self.assertEqual(len(projected), 1)
        np.testing.assert_array_equal(pixels[0]["pixel"], [50.0, 48.0])
        np.testing.assert_allclose(
            projected[0]["normalized_pitch"], [0.5, 0.48]
        )
        self.assertEqual(pixels[0]["confidence"], 0.9)
        self.assertEqual(projected[0]["confidence"], 0.9)

    def test_player_relative_score_can_override_detector_confidence(self):
        candidates = [[{
            "normalized_pitch": [0.1, 0.1],
            "confidence": 0.9,
        }, {
            "normalized_pitch": [0.5, 0.5],
            "confidence": 0.5,
        }]]
        players = {
            4: (
                np.array([[50.0, 34.0]], dtype=np.float32),
                np.array([[90.0, 60.0]], dtype=np.float32),
            )
        }
        ranked = score_player_relative_candidates(
            candidates,
            [4],
            players,
            pitch=(100.0, 68.0),
            distance_scale=2.0,
            distance_weight=2.0,
        )
        self.assertEqual(ranked[0][0]["candidate_index"], 1)
        self.assertEqual(ranked[0][0]["nearest_player_team"], HOME)
        self.assertAlmostEqual(
            ranked[0][0]["nearest_player_distance"], 0.0
        )

    def test_loads_players_only_for_requested_frames(self):
        metadata = type("Metadata", (), {
            "team_of_track": np.array([HOME, AWAY], dtype=np.int8)
        })()
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as handle:
            handle.write("1:0.1,0.2,1;0.8,0.7,2:N\n")
            handle.write("2:0.2,0.3,1;N:N\n")
            path = handle.name
        try:
            positions = load_player_positions_at_frames(
                path, metadata, [1], (100.0, 50.0)
            )
        finally:
            Path(path).unlink()
        self.assertEqual(set(positions), {1})
        np.testing.assert_allclose(positions[1][0], [[20.0, 15.0]])
        self.assertEqual(positions[1][1].shape, (0, 2))

    def test_right_anchored_tracklet_runs_backward(self):
        candidates = [[{
            "normalized_pitch": [(frame + 1) / 11.0, 0.5],
            "confidence": 0.8,
        }] for frame in range(10)]
        result = select_anchored_kalman_tracklet(
            candidates,
            np.array([105.0, 34.0]),
            anchor_side="right",
            fps=1,
            pitch=(105.0, 68.0),
            endpoint_confidence=0.5,
            minimum_tracklet_frames=5,
            maximum_tracklet_frames=10,
            beam_width=12,
            maximum_missing_frames=2,
            process_acceleration_std=5.0,
            measurement_noise_base=0.5,
            measurement_noise_minimum=0.1,
            measurement_noise_maximum=2.0,
            initial_velocity_std=20.0,
            innovation_gate=5.0,
            missing_penalty=3.0,
            confidence_reward=2.0,
        )
        self.assertEqual(len(result.tracklets), 1)
        np.testing.assert_allclose(
            result.xy[:, 0], np.arange(1, 11) * 105.0 / 11.0, atol=1.0
        )
        np.testing.assert_allclose(result.xy[:, 1], 34.0, atol=1.0)

    def test_kalman_tracklets_restart_between_local_segments(self):
        candidates = []
        for frame in range(14):
            x = (frame + 1) / 15.0
            candidates.append([{
                "normalized_pitch": [x, 0.5],
                "confidence": 0.8,
            }, {
                "normalized_pitch": [0.9, 0.1],
                "confidence": 0.2,
            }])
        candidates[6] = []
        result = select_kalman_ball_tracklets(
            candidates,
            fps=1,
            pitch=(105.0, 68.0),
            anchor_confidence=0.5,
            minimum_tracklet_frames=4,
            maximum_tracklet_frames=7,
            beam_width=12,
            maximum_missing_frames=2,
            process_acceleration_std=5.0,
            measurement_noise_base=0.5,
            measurement_noise_minimum=0.1,
            measurement_noise_maximum=2.0,
            initial_velocity_std=20.0,
            innovation_gate=5.0,
            missing_penalty=3.0,
            confidence_reward=2.0,
        )
        selected = np.isfinite(result.xy).all(axis=1)
        self.assertGreaterEqual(len(result.tracklets), 2)
        self.assertGreaterEqual(int(np.count_nonzero(selected)), 12)
        np.testing.assert_allclose(
            result.xy[selected, 1], 34.0, atol=1.0
        )

    def test_kalman_tracklets_require_automatic_anchors(self):
        result = select_kalman_ball_tracklets(
            [[{
                "normalized_pitch": [0.5, 0.5],
                "confidence": 0.1,
            }] for _ in range(8)],
            fps=25,
            pitch=(105.0, 68.0),
            anchor_confidence=0.5,
            minimum_tracklet_frames=4,
            maximum_tracklet_frames=8,
            beam_width=4,
            maximum_missing_frames=2,
            process_acceleration_std=15.0,
            measurement_noise_base=0.5,
            measurement_noise_minimum=0.1,
            measurement_noise_maximum=2.0,
            initial_velocity_std=20.0,
            innovation_gate=4.0,
            missing_penalty=3.0,
            confidence_reward=2.0,
        )
        self.assertEqual(result.tracklets, [])
        self.assertTrue(np.isnan(result.xy).all())

    def test_kalman_beam_selects_linear_track_and_smooths_gap(self):
        candidates = []
        for frame in range(6):
            frame_candidates = [{
                "normalized_pitch": [(frame + 1) / 7.0, 0.5],
                "confidence": 0.8,
            }, {
                "normalized_pitch": [0.9, 0.1 + frame * 0.1],
                "confidence": 0.95,
            }]
            if frame == 3:
                frame_candidates = frame_candidates[1:]
            candidates.append(frame_candidates)
        result = select_kalman_ball_path(
            candidates,
            (-1, np.array([0.0, 34.0])),
            (6, np.array([105.0, 34.0])),
            fps=1,
            pitch=(105.0, 68.0),
            beam_width=12,
            maximum_missing_frames=2,
            process_acceleration_std=5.0,
            measurement_noise_base=0.5,
            measurement_noise_minimum=0.1,
            measurement_noise_maximum=2.0,
            initial_velocity_std=20.0,
            innovation_gate=5.0,
            missing_penalty=3.0,
            confidence_reward=2.0,
        )
        expected_x = np.arange(1, 7) * 15.0
        np.testing.assert_allclose(result.xy[:, 0], expected_x, atol=1.0)
        np.testing.assert_allclose(result.xy[:, 1], 34.0, atol=1.0)
        self.assertEqual(result.candidate_index[3], -1)
        self.assertTrue(np.isfinite(result.total_cost))

    def test_kalman_beam_rejects_unsupported_long_gap(self):
        result = select_kalman_ball_path(
            [[], [], [], []],
            (-1, np.array([0.0, 0.0])),
            (4, np.array([1.0, 0.0])),
            fps=25,
            pitch=(105.0, 68.0),
            beam_width=4,
            maximum_missing_frames=2,
            process_acceleration_std=20.0,
            measurement_noise_base=0.5,
            measurement_noise_minimum=0.1,
            measurement_noise_maximum=2.0,
            initial_velocity_std=10.0,
            innovation_gate=4.0,
            missing_penalty=2.0,
            confidence_reward=2.0,
        )
        self.assertTrue(np.isnan(result.xy).all())
        self.assertTrue(np.isinf(result.total_cost))

    def test_selects_smooth_path_over_high_confidence_distractors(self):
        candidates = []
        for frame in range(5):
            candidates.append([
                {
                    "normalized_pitch": np.array([(frame + 1) / 6.0, 0.5]),
                    "confidence": 0.7,
                },
                {
                    "normalized_pitch": np.array([0.9, 0.1 + 0.15 * frame]),
                    "confidence": 0.95,
                },
            ])
        result = select_temporal_ball_path(
            candidates,
            (-1, np.array([0.0, 34.0])),
            (5, np.array([105.0, 34.0])),
            fps=1,
            pitch=(105.0, 68.0),
            max_gap_frames=1,
            max_speed=25.0,
            max_acceleration=30.0,
        )
        self.assertEqual(result.candidate_index.tolist(), [0, 0, 0, 0, 0])
        self.assertTrue(np.isfinite(result.total_cost))

    def test_bridges_short_detection_gap(self):
        candidates = [
            [{"normalized_pitch": [0.25, 0.5], "confidence": 0.8}],
            [],
            [{"normalized_pitch": [0.75, 0.5], "confidence": 0.8}],
        ]
        result = select_temporal_ball_path(
            candidates,
            (-1, np.array([0.0, 34.0])),
            (3, np.array([105.0, 34.0])),
            fps=2,
            pitch=(105.0, 68.0),
            max_gap_frames=1,
            max_speed=60.0,
            max_acceleration=60.0,
        )
        self.assertEqual(result.candidate_index.tolist(), [0, -1, 0])

    def test_rejects_path_that_cannot_reach_anchor(self):
        candidates = [[{
            "normalized_pitch": [0.9, 0.9],
            "confidence": 1.0,
        }]]
        result = select_temporal_ball_path(
            candidates,
            (-1, np.array([0.0, 0.0])),
            (1, np.array([1.0, 1.0])),
            fps=25,
            pitch=(105.0, 68.0),
            max_gap_frames=0,
            max_speed=45.0,
            max_acceleration=80.0,
        )
        self.assertTrue(np.isnan(result.xy).all())
        self.assertTrue(np.isinf(result.total_cost))

    def test_replaces_isolated_jump_with_interpolation(self):
        raw = np.array(
            [[0.0, 0.0], [20.0, 0.0], [0.2, 0.0]], dtype=np.float32
        )
        result = repair_ball_trajectory(
            raw, fps=10, max_gap_frames=2, max_speed=10.0
        )
        np.testing.assert_allclose(result.xy[1], [0.1, 0.0], atol=1e-6)
        self.assertEqual(result.source.tolist(), [1, 2, 1])

    def test_acceleration_aware_prediction_fills_only_gap_edge(self):
        time = np.arange(12, dtype=np.float32)
        raw = np.column_stack((0.05 * time * time, np.zeros_like(time)))
        raw[7:] = np.nan
        result = repair_ball_trajectory(
            raw,
            fps=10,
            max_gap_frames=0,
            max_speed=30.0,
            prediction_history_frames=7,
            prediction_horizon_frames=3,
            max_acceleration=20.0,
            pitch=(100.0, 50.0),
        )
        np.testing.assert_allclose(
            result.xy[7:10, 0],
            0.05 * np.arange(7, 10, dtype=np.float32) ** 2,
            atol=1e-4,
        )
        self.assertEqual(result.source.tolist(), [1] * 7 + [3] * 3 + [0] * 2)
        self.assertGreater(result.confidence[7], result.confidence[9])

    def test_prediction_requires_observed_history(self):
        raw = np.full((12, 2), np.nan, dtype=np.float32)
        raw[:6, 0] = np.arange(6, dtype=np.float32)
        raw[:6, 1] = 0.0
        result = repair_ball_trajectory(
            raw,
            fps=10,
            max_gap_frames=0,
            prediction_history_frames=7,
            prediction_horizon_frames=3,
        )
        self.assertTrue(np.isnan(result.xy[6:]).all())
        self.assertTrue(np.all(result.source[6:] == 0))

    def test_prunes_incompatible_prediction_hypotheses(self):
        xy = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
            dtype=np.float32,
        )
        source = np.asarray([1, 3, 3, 1], dtype=np.int8)
        confidence = np.asarray([0.0, 0.4, 0.4, 0.0], dtype=np.float32)
        pruned_xy, pruned_source, pruned_confidence = _prune_prediction_jumps(
            xy, source, confidence, fps=1, max_speed=2.0
        )
        self.assertTrue(np.isnan(pruned_xy[1:3]).all())
        self.assertEqual(pruned_source.tolist(), [1, 0, 0, 1])
        np.testing.assert_allclose(pruned_confidence, 0.0)


class HalfLocalDecodeTests(unittest.TestCase):
    def test_decodes_authoritative_halves_independently_and_covers_ungraded_frames(self):
        config = default_config()
        config["semi_markov_minimum_live_frames"] = 5
        config["semi_markov_minimum_dead_frames"] = 5
        control = np.zeros((100, 2), dtype=np.float32)
        control[:10, AWAY] = 1.
        control[10:40, HOME] = 1.
        control[40:65, AWAY] = 1.
        control[65:85, HOME] = 1.
        control[85:, AWAY] = 1.
        evidence = Evidence(
            control=control, dead=np.zeros(100, dtype=np.float32),
            strong_control=np.zeros((100, 2), dtype=bool),
        )
        blocks = [(1, 10, 39), (2, 65, 84)]
        actual = decode_half_local(evidence, empty_tags(100), config, blocks)
        sections = ((0, 10), (10, 40), (40, 65), (65, 85), (85, 100))
        expected = np.concatenate([
            decode(
                Evidence(control[start:stop], evidence.dead[start:stop],
                         evidence.strong_control[start:stop]),
                empty_tags(stop - start), config,
            ) for start, stop in sections
        ])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(actual), 100)
        with self.assertRaisesRegex(ValueError, "ordered, disjoint"):
            decode_half_local(evidence, empty_tags(100), config, [(1, 10, 40), (2, 39, 84)])
        with self.assertRaisesRegex(ValueError, "not aligned"):
            decode_half_local(
                Evidence(control, evidence.dead, strong_control=np.zeros((99, 2))),
                empty_tags(100), config, blocks,
            )

    def test_half_local_config_is_opt_in_and_rejects_cross_half_owner_modes(self):
        config = default_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for overrides, error in (
                ({"decode_halves_separately_enabled": "yes"}, "boolean"),
                ({"decode_halves_separately_enabled": True, "decoder_mode": "hsmm"}, "semi-Markov"),
                ({"decode_halves_separately_enabled": True, "semi_markov_phase_enabled": True},
                 "phase configuration is missing"),
            ):
                path.write_text(json.dumps({**config, **overrides}))
                with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, error):
                    load_config(path)
        with self.assertRaisesRegex(ValueError, "without cross-half owner phases"):
            decode_half_local(
                Evidence(np.zeros((70, 2)), np.zeros(70)), empty_tags(70),
                {**config, "semi_markov_owner_flight_enabled": True}, [(1, 0, 29), (2, 40, 69)],
            )


class TrackingAndTaggingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.meta = self.root / "atd.xml"
        self.raw = self.root / "atd.txt"
        self.tags = self.root / "tags.xml"
        self.meta.write_text(
            '<root><Team id="h"><Name>Barcelona</Name></Team>'
            '<Team id="a"><Name>Athletic Club</Name></Team>'
            '<Player teamId="h"/><Player teamId="a"/></root>'
        )
        lines = []
        for frame in range(1, 11):
            ball_x = 0.2 if frame <= 5 else 0.8
            lines.append(
                "{}:0.2,0.5,0;0.8,0.5,0:{},0.5,1\n".format(frame, ball_x)
            )
        self.raw.write_text("".join(lines))
        self.tags.write_text(
            "<root>"
            "<instance><start>0</start><end>0.2</end><code>SET PIECES</code>"
            "<label><group>Team</group><text>Barcelona</text></label></instance>"
            "<instance><start>0.2</start><end>0.8</end><code>BUILD UP</code>"
            "<label><group>Team</group><text>Atletic Club</text></label></instance>"
            "</root>"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_parses_metric_team_control_features(self):
        metadata = load_tracking_metadata(str(self.meta))
        ball_xy, _ = load_raw_ball(str(self.raw), 10, (100.0, 50.0))
        features = extract_control_features(
            str(self.raw), metadata, ball_xy, (100.0, 50.0)
        )
        self.assertEqual(metadata.team_names, ("Barcelona", "Athletic Club"))
        np.testing.assert_allclose(features.nearest_distances[:5, HOME, 0], 0.0)
        np.testing.assert_allclose(features.nearest_distances[5:, AWAY, 0], 0.0)
        self.assertTrue(np.all(features.player_counts == 1))
        np.testing.assert_allclose(features.median_player_speed, 0.0)
        np.testing.assert_allclose(features.stationary_player_fraction, 1.0)
        np.testing.assert_allclose(
            features.nearest_player_velocity[1:], 0.0, equal_nan=False
        )

    def test_goalkeeper_track_at_the_other_goal_counts_for_that_team(self):
        from possession.tracking import goalkeeper_frame_teams
        self.meta.write_text(
            '<root><Team id="h"><Name>Barcelona</Name></Team>'
            '<Team id="a"><Name>Athletic Club</Name></Team>'
            '<Player teamId="h"><ProviderParameter><Name>position_type</Name><Value>Unknown</Value>'
            '</ProviderParameter></Player>'
            '<Player teamId="a"><ProviderParameter><Name>position_type</Name><Value>Goalkeeper</Value>'
            '</ProviderParameter></Player></root>'
        )
        metadata = load_tracking_metadata(str(self.meta))
        self.assertEqual(metadata.goalkeeper_slots, (1,))
        players = np.full((4, 2, 2), np.nan)
        players[:, 1] = [[95.0, 34.0], [10.0, 34.0], [10.0, 34.0], [10.0, 34.0]]
        # Home attacks toward x = length in frames 0-1, toward x = 0 in frame 2; frame 3 unknown.
        teams = goalkeeper_frame_teams(players, metadata, np.array([1, 1, -1, 0]), (105.0, 68.0), 30.0)
        self.assertEqual(teams[:, 1].tolist(), [AWAY, HOME, AWAY, AWAY])
        self.assertTrue((teams[:, 0] == HOME).all())
        ball_xy, _ = load_raw_ball(str(self.raw), 10, (100.0, 50.0))
        frame_teams = np.tile(metadata.team_of_track, (10, 1))
        frame_teams[5:, 1] = HOME
        features = extract_control_features(
            str(self.raw), metadata, ball_xy, (100.0, 50.0), frame_teams=frame_teams)
        np.testing.assert_allclose(features.nearest_distances[5:, HOME, 0], 0.0)
        self.assertTrue(np.all(features.player_counts[5:, AWAY] == 0))
        self.assertTrue(np.all(features.nearest_player_slot[5:, HOME] == 3))

    def test_rejects_noncontiguous_tracking_frames(self):
        self.raw.write_text(
            "1:0.2,0.5,0;0.8,0.5,0:0.2,0.5,1\n"
            "3:0.2,0.5,0;0.8,0.5,0:0.2,0.5,1\n"
        )
        with self.assertRaisesRegex(ValueError, "expected frame 2"):
            load_raw_ball(str(self.raw), 2, (100.0, 50.0))

    def test_aligns_set_piece_and_fuzzy_team_name(self):
        result = load_tag_features(
            str(self.tags),
            ("Barcelona", "Athletic Club"),
            n_frames=10,
            fps=10,
            restart_offset_frames=1,
            restart_prior_frames=2,
            dead_lookback_frames=3,
        )
        self.assertTrue(result.dead_hint[0])
        self.assertTrue(np.all(~result.dead_hint[1:]))
        self.assertTrue(result.pre_restart_window[0])
        self.assertTrue(np.all(~result.pre_restart_window[1:]))
        self.assertTrue(np.all(result.team_prior[2:8] < 0))
        self.assertEqual(result.restart_team[1], HOME)

    def test_defending_tag_team_can_be_excluded_without_losing_restarts(self):
        self.tags.write_text(
            "<root>"
            "<instance><start>0</start><end>0.5</end><code>ATTACKING TRANSITION</code>"
            "<label><group>Team</group><text>Barcelona</text></label></instance>"
            "<instance><start>0</start><end>0.5</end><code>DEFENDING IN MIDDLE THIRD</code>"
            "<label><group>Team</group><text>Athletic Club</text></label></instance>"
            "<instance><start>0.5</start><end>0.8</end><code>SET PIECES</code>"
            "<label><group>Team</group><text>Athletic Club</text></label></instance>"
            "</root>"
        )
        kwargs = dict(
            smart_tagging_path=str(self.tags), team_names=("Barcelona", "Athletic Club"),
            n_frames=10, fps=10, restart_offset_frames=1,
        )
        original = load_tag_features(**kwargs)
        code_aware = load_tag_features(**kwargs, team_prior_mode="exclude_defending")
        np.testing.assert_array_equal(original.team_prior[:5], np.zeros(5))
        np.testing.assert_array_equal(code_aware.team_prior[:5], np.ones(5))
        np.testing.assert_array_equal(original.dead_hint, code_aware.dead_hint)
        np.testing.assert_array_equal(original.restart_team, code_aware.restart_team)
        np.testing.assert_array_equal(original.team_prior[5:], code_aware.team_prior[5:])
        # Signed mode credits the opponent of a defending team.
        signed = load_tag_features(**kwargs, team_prior_mode="signed")
        np.testing.assert_array_equal(signed.team_prior[:5], np.ones(5))
        np.testing.assert_array_equal(original.restart_team, signed.restart_team)
        with self.assertRaisesRegex(ValueError, "exclude_defending or signed"):
            load_tag_features(**kwargs, team_prior_mode="invented")

    def test_localizes_restart_from_sustained_ball_motion(self):
        tags = TagFeatures(
            team_prior=np.zeros(30, dtype=np.float32),
            dead_hint=np.r_[np.ones(8, dtype=bool), np.zeros(22, dtype=bool)],
            pre_restart_window=np.zeros(30, dtype=bool),
            restart_team=np.full(30, -1, dtype=np.int8),
            restart_events=((0, 25, 8, HOME),),
            restart_frames=np.asarray([8], dtype=np.int32),
            restart_localized=np.asarray([False]),
        )

        class Trajectory:
            xy = np.zeros((30, 2), dtype=np.float32)
            xy[10:, 0] = np.arange(20, dtype=np.float32)
            speed = np.r_[np.zeros(10), np.full(20, 3.0)]
            source = np.ones(30, dtype=np.int8)

        config = default_config()
        config.update({
            "restart_localization_search_radius_frames": 5,
            "restart_localization_motion_window_frames": 4,
            "restart_localization_min_pre_observed_frames": 3,
            "restart_localization_min_post_reliable_frames": 4,
        })
        result = localize_restarts(tags, Trajectory(), 10.0, config)
        self.assertEqual(result.restart_frames.tolist(), [11])
        self.assertEqual(result.restart_localized.tolist(), [True])
        self.assertTrue(np.all(result.dead_hint[:11]))
        self.assertTrue(np.all(~result.dead_hint[11:]))
        self.assertTrue(np.all(result.restart_team[11:30] == HOME))

    def test_infers_unknown_restart_owner_from_sustained_control(self):
        tags = TagFeatures(
            team_prior=np.zeros(40, dtype=np.float32),
            dead_hint=np.zeros(40, dtype=bool),
            pre_restart_window=np.zeros(40, dtype=bool),
            restart_team=np.full(40, -1, dtype=np.int8),
            restart_events=((0, 30, 10, -1),),
            restart_frames=np.asarray([10], dtype=np.int32),
            restart_localized=np.asarray([True]),
            restart_types=("throw-in",),
            restart_teams=np.asarray([-1], dtype=np.int8),
            restart_team_inferred=np.asarray([False]),
        )

        class Trajectory:
            source = np.ones(40, dtype=np.int8)

        class Control:
            nearest_distances = np.full(
                (40, 2, 3), np.inf, dtype=np.float32
            )
            nearby_counts = np.zeros((40, 2, 3), dtype=np.int16)

        Control.nearest_distances[10:20, HOME, 0] = 1.0
        Control.nearest_distances[10:20, AWAY, 0] = 3.0
        Control.nearby_counts[10:20, HOME, 1] = 1
        config = default_config()
        result = infer_restart_ownership(
            tags, Trajectory(), Control(), config
        )
        self.assertEqual(result.restart_teams.tolist(), [HOME])
        self.assertEqual(result.restart_team_inferred.tolist(), [True])
        self.assertTrue(np.all(result.restart_team[10:35] == HOME))


class DecoderTests(unittest.TestCase):
    def config(self):
        config = default_config()
        config.update(
            decoder_mode="state_machine",
            live_confirm_frames=3,
            dead_confirm_frames=3,
            transition_confirm_frames=20,
        )
        return config

    def test_ignores_brief_opponent_touch(self):
        control = np.zeros((70, 2), dtype=np.float32)
        control[:20, HOME] = 1.0
        control[20:35, AWAY] = 1.0
        control[35:, HOME] = 1.0
        labels = decode(
            Evidence(control=control, dead=np.zeros(70, dtype=np.float32)),
            empty_tags(70),
            self.config(),
        )
        np.testing.assert_array_equal(labels, np.full(70, HOME))

    def test_confirms_sustained_possession_change(self):
        control = np.zeros((50, 2), dtype=np.float32)
        control[:20, HOME] = 1.0
        control[20:, AWAY] = 1.0
        labels = decode(
            Evidence(control=control, dead=np.zeros(50, dtype=np.float32)),
            empty_tags(50),
            self.config(),
        )
        np.testing.assert_array_equal(labels[:20], np.full(20, HOME))
        np.testing.assert_array_equal(labels[20:], np.full(30, AWAY))

    def test_confirms_short_change_only_with_strong_control(self):
        control = np.zeros((40, 2), dtype=np.float32)
        control[:20, HOME] = 1.0
        control[20:, AWAY] = 1.0
        strong = np.zeros((40, 2), dtype=bool)
        strong[20:, AWAY] = True
        config = self.config()
        config["strong_transition_confirm_frames"] = 5
        labels = decode(
            Evidence(
                control=control,
                dead=np.zeros(40, dtype=np.float32),
                strong_control=strong,
            ),
            empty_tags(40),
            config,
        )
        np.testing.assert_array_equal(labels[:20], np.full(20, HOME))
        np.testing.assert_array_equal(labels[20:], np.full(20, AWAY))

    def test_inserts_dead_ball_and_restarts(self):
        control = np.zeros((40, 2), dtype=np.float32)
        control[:, HOME] = 1.0
        dead = np.zeros(40, dtype=np.float32)
        dead[15:25] = 1.0
        labels = decode(
            Evidence(control=control, dead=dead),
            empty_tags(40),
            self.config(),
        )
        np.testing.assert_array_equal(labels[:15], np.full(15, HOME))
        np.testing.assert_array_equal(labels[15:25], np.full(10, DEAD))
        np.testing.assert_array_equal(labels[25:], np.full(15, HOME))

    def test_preserves_live_state_without_new_evidence(self):
        control = np.zeros((40, 2), dtype=np.float32)
        control[:10, HOME] = 1.0
        labels = decode(
            Evidence(control=control, dead=np.zeros(40, dtype=np.float32)),
            empty_tags(40),
            self.config(),
        )
        np.testing.assert_array_equal(labels, np.full(40, HOME))

    def test_extends_one_contiguous_stoppage_into_restart(self):
        class Features:
            median_player_speed = np.asarray(
                [2.0, 2.0, 2.0, 0.8, 0.8, 2.0, 2.0, 0.8, 0.8, 0.8]
            )
            stationary_player_fraction = np.asarray(
                [0.0, 0.0, 0.0, 0.7, 0.7, 0.0, 0.0, 0.7, 0.7, 0.7]
            )

        class Trajectory:
            source = np.asarray([1, 1, 1, 0, 0, 0, 0, 1, 1, 1])

        tags = TagFeatures(
            team_prior=np.zeros(10, dtype=np.float32),
            dead_hint=np.asarray(
                [False, False, False, False, False, False, False, True, True, True]
            ),
            pre_restart_window=np.ones(10, dtype=bool),
            restart_team=np.full(10, -1, dtype=np.int8),
        )
        config = default_config()
        config["stoppage_gap_tolerance_frames"] = 2
        hint = _restart_stoppage_hint(Features(), Trajectory(), tags, config)
        np.testing.assert_array_equal(
            hint,
            [False, True, True, True, True, True, True, True, True, True],
        )

    def test_semi_markov_decodes_complete_duration_constrained_spells(self):
        control = np.zeros((60, 2), dtype=np.float32)
        control[:30, HOME] = 1.0
        control[30:, AWAY] = 1.0
        config = default_config()
        config.update({
            "decoder_mode": "semi_markov",
            "semi_markov_minimum_live_frames": 5,
            "semi_markov_minimum_dead_frames": 3,
            "semi_markov_live_transition_cost": 1.0,
            "semi_markov_strong_transition_cost": 1.0,
            "semi_markov_dead_transition_cost": 1.0,
        })
        labels = decode_semi_markov(
            Evidence(
                control=control,
                dead=np.zeros(60, dtype=np.float32),
            ),
            config,
        )
        np.testing.assert_array_equal(labels[:30], np.full(30, HOME))
        np.testing.assert_array_equal(labels[30:], np.full(30, AWAY))

    def test_semi_markov_rejects_short_opponent_pulse(self):
        control = np.zeros((50, 2), dtype=np.float32)
        control[:, HOME] = 1.0
        control[20:23, HOME] = 0.0
        control[20:23, AWAY] = 1.0
        config = default_config()
        config.update({
            "decoder_mode": "semi_markov",
            "semi_markov_minimum_live_frames": 5,
            "semi_markov_minimum_dead_frames": 3,
            "semi_markov_live_transition_cost": 2.0,
            "semi_markov_strong_transition_cost": 2.0,
        })
        labels = decode_semi_markov(
            Evidence(
                control=control,
                dead=np.zeros(50, dtype=np.float32),
            ),
            config,
        )
        np.testing.assert_array_equal(labels, np.full(50, HOME))

    def test_semi_markov_discounts_sustained_strong_handover(self):
        control = np.zeros((70, 2), dtype=np.float32)
        control[:60, HOME] = 1.0
        control[60:, AWAY] = 1.0
        strong = np.zeros((70, 2), dtype=bool)
        strong[60:, AWAY] = True
        config = default_config()
        config.update({
            "decoder_mode": "semi_markov",
            "semi_markov_minimum_live_frames": 5,
            "semi_markov_minimum_dead_frames": 3,
            "semi_markov_live_transition_cost": 30.0,
            "semi_markov_strong_transition_cost": 10.0,
            "semi_markov_strong_transition_window_frames": 10,
            "semi_markov_strong_transition_minimum_frames": 8,
            "semi_markov_dead_transition_cost": 30.0,
        })
        evidence = Evidence(
            control=control,
            dead=np.zeros(70, dtype=np.float32),
            strong_control=strong,
        )
        labels = decode_semi_markov(evidence, config)
        np.testing.assert_array_equal(labels[:60], np.full(60, HOME))
        np.testing.assert_array_equal(labels[60:], np.full(10, AWAY))

        evidence.strong_control[:] = False
        labels_without_support = decode_semi_markov(evidence, config)
        self.assertTrue(np.all(labels_without_support == HOME))

    def test_hsmm_duration_score_prefers_configured_mode(self):
        config = default_config()
        durations = np.asarray([25, 100, 400], dtype=np.int32)
        scores = _hsmm_duration_scores(durations, DEAD, config)
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], scores[2])

    def test_hsmm_decodes_complete_segments_near_proposed_boundary(self):
        control = np.zeros((80, 2), dtype=np.float32)
        control[:40, HOME] = 1.0
        control[40:, AWAY] = 1.0
        strong = np.zeros((80, 2), dtype=bool)
        strong[:40, HOME] = True
        strong[40:, AWAY] = True
        config = default_config()
        config.update({
            "decoder_mode": "hsmm",
            "semi_markov_minimum_live_frames": 5,
            "semi_markov_minimum_dead_frames": 3,
            "semi_markov_live_transition_cost": 5.0,
            "semi_markov_strong_transition_cost": 1.0,
            "semi_markov_dead_transition_cost": 5.0,
            "hsmm_minimum_live_frames": 5,
            "hsmm_minimum_dead_frames": 3,
            "hsmm_maximum_segment_frames": 100,
            "hsmm_boundary_search_radius_frames": 5,
            "hsmm_live_duration_mode_frames": 40,
            "hsmm_dead_duration_mode_frames": 10,
        })
        labels = decode_hsmm(
            Evidence(
                control=control,
                dead=np.zeros(80, dtype=np.float32),
                strong_control=strong,
            ),
            config,
        )
        np.testing.assert_array_equal(labels[:40], np.full(40, HOME))
        np.testing.assert_array_equal(labels[40:], np.full(40, AWAY))


class SubmissionAndConfigTests(unittest.TestCase):
    def test_missing_video_evidence_points_to_regeneration_tools(self):
        from possession.cli import load_video_team
        config = {"video_team_evidence_path": "evidence/does_not_exist.npz"}
        with self.assertRaisesRegex(FileNotFoundError, "regenerate it from the tactical video with tools/"):
            load_video_team(config, np.zeros(10, dtype=np.int8))

    def test_video_path_override_requires_diagnostic_report(self):
        with self.assertRaisesRegex(ValueError, "require --video-path-report"):
            run_pipeline(
                None, None, DEFAULT_CONFIG,
                video_path_mode="edge_anchored",
            )

    def test_working_contact_carry_preserves_restart_handover_and_exit_rollbacks(self):
        baseline = load_config(ROOT / "configs" / "compliant_tracking.json")
        frozen = load_config(ROOT / "configs" / "exit_bracket_release_candidate.json")
        exit_only = load_config(ROOT / "configs" / "submission_tracking.json")
        handover = load_config(ROOT / "configs" / "observed_handover_candidate.json")
        kickoff = load_config(DEFAULT_CONFIG)
        kickoff_trial = load_config(ROOT / "configs" / "pre_kickoff_trial.json")
        aware = load_config(ROOT / "configs" / "submission_ball_aware.json")
        aware_trial = load_config(ROOT / "configs" / "ballaware_trial.json")
        tracked = load_config(ROOT / "configs" / "submission_tracked_ballfree.json")
        tracked_trial = load_config(ROOT / "configs" / "tracked_ballfree_trial.json")
        video = load_config(ROOT / "configs" / "submission_video_evidence.json")
        video_trial = load_config(ROOT / "configs" / "video_evidence_trial.json")
        keeper = load_config(ROOT / "configs" / "submission_goalkeeper_fix.json")
        keeper_trial = load_config(ROOT / "configs" / "goalkeeper_fix_trial.json")
        radius = load_config(ROOT / "configs" / "submission_touch_radius.json")
        radius_trial = load_config(ROOT / "configs" / "touch_radius_trial.json")
        fused = load_config(ROOT / "configs" / "submission_broadcast_ball.json")
        fused_trial = load_config(ROOT / "configs" / "broadcast_ball_trial.json")
        signed = load_config(ROOT / "configs" / "submission_signed_tags.json")
        signed_trial = load_config(ROOT / "configs" / "signed_tags_trial.json")
        switch = load_config(ROOT / "configs" / "submission_supported_switch.json")
        switch_trial = load_config(ROOT / "configs" / "supported_switch_trial.json")
        sequence = load_config(ROOT / "configs" / "submission_ballfree_seq.json")
        sequence_trial = load_config(ROOT / "configs" / "ballfree_seq_trial.json")
        strong = load_config(ROOT / "configs" / "submission_contact_strong.json")
        strong_trial = load_config(ROOT / "configs" / "contact_carry_strong_trial.json")
        ballfree = load_config(ROOT / "configs" / "submission_ballfree.json")
        ballfree_trial = load_config(ROOT / "configs" / "ballfree_trial.json")
        broadcast = load_config(ROOT / "configs" / "submission_broadcast_stoppage.json")
        broadcast_trial = load_config(ROOT / "configs" / "broadcast_stoppage_trial.json")
        contact = load_config(ROOT / "configs" / "submission_contact_carry.json")
        contact_trial = load_config(ROOT / "configs" / "contact_carry_trial.json")
        working = load_config(ROOT / "configs" / "submission_restart_dead.json")
        trial = load_config(ROOT / "configs" / "restart_dead_trial.json")
        self.assertEqual(Path(DEFAULT_CONFIG), ROOT / "configs" / "submission_pre_kickoff.json")
        self.assertEqual(
            {key: value for key, value in kickoff.items() if key != "working_submission_selection"},
            kickoff_trial,
        )
        self.assertEqual(
            {key for key in set(kickoff) | set(aware)
             if key != "working_submission_selection" and kickoff.get(key) != aware.get(key)},
            set(kickoff["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertEqual(
            {key: value for key, value in aware.items() if key != "working_submission_selection"},
            aware_trial,
        )
        self.assertEqual(
            {key for key in set(aware) | set(tracked)
             if key != "working_submission_selection" and aware.get(key) != tracked.get(key)},
            set(aware["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertTrue((ROOT / aware["ballfree_ball_weights_path"]).is_file())
        self.assertEqual(
            {key: value for key, value in tracked.items() if key != "working_submission_selection"},
            tracked_trial,
        )
        self.assertEqual(
            {key for key in set(tracked) | set(video)
             if key != "working_submission_selection" and tracked.get(key) != video.get(key)},
            set(tracked["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertEqual(
            {key: value for key, value in video.items() if key != "working_submission_selection"},
            video_trial,
        )
        self.assertEqual(
            {key for key in set(video) | set(keeper)
             if key != "working_submission_selection" and video.get(key) != keeper.get(key)},
            set(video["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        for key in ("track_team_override_path", "video_team_evidence_path", "ballfree_weights_path"):
            self.assertTrue((ROOT / video[key]).is_file())
        self.assertEqual(
            {key: value for key, value in keeper.items() if key != "working_submission_selection"},
            keeper_trial,
        )
        self.assertEqual(
            {key for key in set(keeper) | set(radius)
             if key != "working_submission_selection" and keeper.get(key) != radius.get(key)},
            set(keeper["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertEqual(
            {key: value for key, value in radius.items() if key != "working_submission_selection"},
            radius_trial,
        )
        self.assertEqual(
            {key for key in set(radius) | set(fused)
             if key != "working_submission_selection" and radius.get(key) != fused.get(key)},
            set(radius["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertEqual(
            {key: value for key, value in fused.items() if key != "working_submission_selection"},
            fused_trial,
        )
        self.assertEqual(
            {key for key in set(fused) | set(signed)
             if key != "working_submission_selection" and fused.get(key) != signed.get(key)},
            set(fused["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertEqual(
            {key: value for key, value in signed.items() if key != "working_submission_selection"},
            signed_trial,
        )
        self.assertEqual(
            {key for key in set(signed) | set(switch)
             if key != "working_submission_selection" and signed.get(key) != switch.get(key)},
            set(signed["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertEqual(
            {key: value for key, value in switch.items() if key != "working_submission_selection"},
            switch_trial,
        )
        self.assertEqual(
            set(switch["working_submission_selection"]["parameter_changes_from_previous"]),
            {"semi_markov_strong_transition_cost"},
        )
        self.assertEqual(
            {key: value for key, value in sequence.items() if key != "working_submission_selection"},
            sequence_trial,
        )
        self.assertEqual(sequence["ballfree_model"], "sequence")
        self.assertTrue((ROOT / sequence["ballfree_weights_path"]).is_file())
        self.assertEqual(
            {key for key in set(sequence) | set(strong)
             if key != "working_submission_selection" and sequence.get(key) != strong.get(key)},
            set(sequence["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertEqual(
            {key: value for key, value in strong.items() if key != "working_submission_selection"},
            strong_trial,
        )
        self.assertEqual(
            set(strong["working_submission_selection"]["parameter_changes_from_previous"]),
            {"contact_carry_weight"},
        )
        self.assertEqual(strong["contact_carry_weight"], 1.0)
        self.assertEqual(
            {key: value for key, value in ballfree.items() if key != "working_submission_selection"},
            ballfree_trial,
        )
        self.assertEqual(
            {key for key in set(ballfree) | set(broadcast)
             if key != "working_submission_selection" and ballfree.get(key) != broadcast.get(key)},
            set(ballfree["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertTrue(ballfree["semi_markov_ballfree_enabled"])
        self.assertFalse(broadcast.get("semi_markov_ballfree_enabled", False))
        self.assertEqual(
            {key: value for key, value in broadcast.items() if key != "working_submission_selection"},
            broadcast_trial,
        )
        self.assertEqual(
            {key for key in set(broadcast) | set(contact)
             if key != "working_submission_selection" and broadcast.get(key) != contact.get(key)},
            set(broadcast["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertTrue(broadcast["broadcast_stoppage_enabled"])
        self.assertFalse(contact.get("broadcast_stoppage_enabled", False))
        self.assertEqual(
            {key: value for key, value in contact.items()
             if key != "working_submission_selection"},
            {key: value for key, value in contact_trial.items()
             if key != "contact_carry_trial_provenance"},
        )
        self.assertEqual(
            {key for key in set(contact) | set(working)
             if key != "working_submission_selection"
             and contact.get(key) != working.get(key)},
            set(contact["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertTrue(contact["semi_markov_contact_carry_enabled"])
        self.assertFalse(working.get("semi_markov_contact_carry_enabled", False))
        self.assertEqual(
            {key: value for key, value in working.items()
             if key != "working_submission_selection"},
            {key: value for key, value in trial.items()
             if key != "restart_dead_trial_provenance"},
        )
        selected = load_config(ROOT / "configs" / "submission_handover.json")
        self.assertEqual(
            {key for key in set(working) | set(selected)
             if key != "working_submission_selection"
             and working.get(key) != selected.get(key)},
            set(working["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertTrue(working["tracking_kickoff_dead_enabled"])
        self.assertTrue(working["tracking_restart_extension_enabled"])
        self.assertEqual(Path(DEFAULT_OUTPUT), ROOT / "submissions" / "possession_compliant.csv")
        self.assertEqual(
            {key: value for key, value in exit_only.items()
             if key != "working_submission_selection"},
            {key: value for key, value in frozen.items()
             if key != "exit_bracket_release_provenance"},
        )
        self.assertEqual(
            {key for key in set(exit_only) | set(baseline)
             if key != "working_submission_selection"
             and exit_only.get(key) != baseline.get(key)},
            set(exit_only["working_submission_selection"]["parameter_changes_from_original"]),
        )
        self.assertEqual(
            {key: value for key, value in selected.items()
             if key != "working_submission_selection"},
            {key: value for key, value in handover.items()
             if key != "handover_candidate_provenance"},
        )
        self.assertEqual(
            {key for key in set(selected) | set(exit_only)
             if key != "working_submission_selection"
             and selected.get(key) != exit_only.get(key)},
            set(selected["working_submission_selection"]["parameter_changes_from_previous"]),
        )
        self.assertTrue(selected["semi_markov_observed_handover_enabled"])
        self.assertFalse(exit_only.get("semi_markov_observed_handover_enabled", False))
        self.assertIsNone(selected["video_ball_observations_path"])
        self.assertEqual(
            selected["working_submission_selection"]["status"],
            "selected_working_under_published_rules",
        )

    def test_writes_and_validates_exact_submission(self):
        labels = np.asarray([HOME, HOME, DEAD, AWAY], dtype=np.int8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "submission.csv"
            write_submission(str(path), labels)
            validate_submission_file(str(path), 4)
            self.assertEqual(
                path.read_text(),
                "frame,label\n0,H\n1,H\n2,D\n3,A\n",
            )

    def test_summarizes_possession_spells(self):
        labels = np.asarray(
            [HOME, HOME, DEAD, HOME, AWAY, AWAY], dtype=np.int8
        )
        report = summarize(labels, fps=2)
        self.assertEqual(report["possessions"], 3)
        self.assertEqual(report["shorter_than_one_second"], 1)

    def test_rejects_incomplete_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "missing config keys"):
                load_config(str(path))


class VideoRecoveryTests(unittest.TestCase):
    def test_selects_paired_long_gaps_evenly_across_halves_and_lengths(self):
        samples = []
        for half in (1, 2):
            for gap in (269, 938):
                start = half * 1000 + gap
                samples.extend([
                    {"half": half, "kind": "endpoint", "side": "left",
                     "frame": start - 1, "gap_frames": gap},
                    {"half": half, "kind": "endpoint", "side": "right",
                     "frame": start + gap, "gap_frames": gap},
                ])
        selected = select_balanced_long_gap_runs(
            samples, runs_per_cohort=1
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {(item["half"], item["band"]) for item in selected},
            {(1, "269_to_937"), (1, "938_plus"),
             (2, "269_to_937"), (2, "938_plus")},
        )
        self.assertTrue(all(
            item["start_frame"] == item["half"] * 1000 + item["gap_frames"]
            for item in selected
        ))
        with self.assertRaisesRegex(ValueError, "only 1 eligible gaps"):
            select_balanced_long_gap_runs(samples, runs_per_cohort=2)

    def test_input_only_video_audit_reports_agreement_without_labels(self):
        class Capture:
            position = 0

            def isOpened(self):
                return True

            def set(self, _property, frame):
                self.position = frame

            def read(self):
                self.position += 1
                return True, self.position - 1

            def get(self, prop):
                return {
                    1: 4, 2: 25.0, 3: 100, 4: 100, 5: self.position,
                }[prop]

            def release(self):
                pass

        cv2 = SimpleNamespace(
            VideoCapture=lambda _path: Capture(),
            CAP_PROP_FRAME_COUNT=1, CAP_PROP_FPS=2,
            CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
            CAP_PROP_POS_FRAMES=5,
        )
        ultralytics = SimpleNamespace(
            YOLO=lambda _path: SimpleNamespace(
                names={32: "sports ball"},
                ckpt={"train_args": {"data": "coco.yaml"}},
            ),
            settings=SimpleNamespace(update=lambda _values: None),
        )
        samples = [
            {"frame": 1, "kind": "endpoint", "supported": True,
             "side": "left", "gap_frames": 938, "half": 1},
            {"frame": 3, "kind": "stable", "supported": True,
             "side": None, "gap_frames": 0, "half": 1},
        ]
        raw = np.array([
            [0.0, 25.0], [10.0, 25.0],
            [0.0, 25.0], [20.0, 25.0],
        ])

        def detector(image, *_args, **_kwargs):
            return [{
                "pixel": np.asarray(
                    [10.0, 50.0] if image == 1 else [90.0, 50.0]
                ),
                "confidence": 0.8,
            }]

        with mock.patch.dict(
            "sys.modules", {"cv2": cv2, "ultralytics": ultralytics}
        ), mock.patch(
            "possession.video.detect_tiled_ball_pixels",
            side_effect=detector,
        ):
            result = audit_input_ball_anchor_video(
                "synthetic.mp4", "coco.pt", samples, raw,
                np.repeat(np.diag([100.0, 100.0, 1.0])[None, :, :], 4, axis=0),
                (100.0, 50.0), fps=25, device="cpu",
            )
        self.assertEqual(
            result["groups"]["all_endpoints"]["nearest_match_40px"], 1
        )
        self.assertEqual(
            result["groups"]["stable_reference"]["nearest_match_40px"], 0
        )
        self.assertEqual(
            result["groups"]["all_endpoints"]["on_screen_frames"], 1
        )
        self.assertEqual(
            result["groups"]["half_1_endpoints"]["nearest_match_40px"], 1
        )
        self.assertEqual(
            result["groups"]["very_long_938_plus_endpoints"]["nearest_match_40px"], 1
        )

    def test_selects_observed_long_gap_ends_and_separate_stable_controls(self):
        raw = np.column_stack((
            np.arange(50, dtype=np.float32) * 0.1,
            np.full(50, 34.0, dtype=np.float32),
        ))
        raw[10:20] = np.nan
        samples, summary = select_input_ball_anchor_samples(
            raw, np.full(50, 0.25), np.ones(50, dtype=bool),
            [(1, 0, 49)],
            fps=25, minimum_gap_frames=10,
            maximum_homography_error=2.0,
            maximum_ball_speed=65.0, stable_sample_count=4,
        )
        ends = [sample for sample in samples if sample["kind"] == "endpoint"]
        controls = [sample for sample in samples if sample["kind"] == "stable"]
        self.assertEqual([(item["frame"], item["side"]) for item in ends], [
            (9, "left"), (20, "right"),
        ])
        self.assertTrue(all(item["supported"] for item in ends))
        self.assertEqual([item["support_frames"] for item in ends], [5, 5])
        self.assertEqual(len(controls), 4)
        self.assertEqual(summary["long_runs_inside_halves"], 1)
        masked, hidden, eligible = make_long_gap_edge_dropout(
            raw, samples, [(1, 0, 49)],
            minimum_gap_frames=10, edge_width_frames=3,
        )
        self.assertEqual(len(eligible), 2)
        self.assertEqual(
            np.flatnonzero(hidden).tolist(), [7, 8, 9, 20, 21, 22]
        )
        self.assertTrue(np.isnan(masked[hidden]).all())
        self.assertTrue(np.isfinite(raw[hidden]).all())
        with self.assertRaisesRegex(ValueError, "no eligible"):
            make_long_gap_edge_dropout(
                raw, samples, [(1, 0, 49)],
                minimum_gap_frames=100, edge_width_frames=3,
            )

        crossed, summary = select_input_ball_anchor_samples(
            raw, np.full(50, 0.25), np.ones(50, dtype=bool),
            [(1, 0, 15), (2, 16, 49)],
            fps=25, minimum_gap_frames=10,
            maximum_homography_error=2.0,
            maximum_ball_speed=65.0, stable_sample_count=0,
        )
        self.assertEqual(crossed, [])
        self.assertEqual(summary["long_runs_inside_halves"], 0)

    def test_tactical_tiles_cover_frame_with_overlap(self):
        self.assertEqual(
            _tile_regions(1920, 1080, 0.1),
            [
                (0, 0, 1056, 594),
                (864, 0, 1920, 594),
                (0, 486, 1056, 1080),
                (864, 486, 1920, 1080),
            ],
        )

    def test_project_round_trip(self):
        matrix = np.asarray([
            [2.0, 0.0, 10.0],
            [0.0, 3.0, 20.0],
            [0.0, 0.0, 1.0],
        ])
        pixel = _project(matrix, 0.25, 0.75)
        restored = _project(np.linalg.inv(matrix), pixel[0], pixel[1])
        np.testing.assert_allclose(restored, [0.25, 0.75])

    def test_loads_aligned_homography_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "homography.csv"
            path.write_text(
                "frame,reprojection_error,valid\n"
                "0,0.2,1\n"
                "1,1.5,0\n"
                "2,0.4,1\n"
            )
            error, valid = load_homography_quality(str(path), 3)
        np.testing.assert_allclose(error, [0.2, 1.5, 0.4])
        np.testing.assert_array_equal(valid, [True, False, True])

    def test_ranks_missing_run_crossing_team_boundary(self):
        source = np.asarray([1, 1, 0, 0, 0, 0, 1, 1], dtype=np.int8)
        labels = np.asarray(
            [HOME, HOME, HOME, HOME, AWAY, AWAY, AWAY, AWAY],
            dtype=np.int8,
        )
        error = np.full(8, 0.25, dtype=np.float32)
        valid = np.ones(8, dtype=bool)
        intervals = rank_recovery_intervals(
            source,
            labels,
            error,
            valid,
            boundary_radius=3,
            minimum_missing_frames=2,
            minimum_valid_fraction=0.9,
            maximum_median_error=1.0,
            context_frames=1,
            maximum_intervals=5,
        )
        self.assertEqual(len(intervals), 1)
        interval = intervals[0]
        self.assertEqual((interval.start_frame, interval.end_frame), (2, 6))
        self.assertEqual(interval.boundary_frame, 4)
        self.assertEqual((interval.from_team, interval.to_team), (HOME, AWAY))
        self.assertTrue(interval.left_anchor)
        self.assertTrue(interval.right_anchor)
        json.dumps(interval.to_dict())

    def test_rejects_interval_with_poor_homography(self):
        intervals = rank_recovery_intervals(
            np.asarray([1, 0, 0, 1], dtype=np.int8),
            np.asarray([HOME, HOME, AWAY, AWAY], dtype=np.int8),
            np.full(4, 3.0, dtype=np.float32),
            np.ones(4, dtype=bool),
            boundary_radius=2,
            minimum_missing_frames=2,
            minimum_valid_fraction=0.9,
            maximum_median_error=1.0,
            context_frames=1,
            maximum_intervals=5,
        )
        self.assertEqual(intervals, [])


def _match_constants_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("derive_match_constants", ROOT / "tools" / "derive_match_constants.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MatchConstantDerivationTests(unittest.TestCase):
    """tools/derive_match_constants.py: video-evidence inputs derived from data, not hand-set frames."""

    @classmethod
    def setUpClass(cls):
        cls.C = _match_constants_module()

    def test_home_left_follows_goalkeeper_side_of_each_half(self):
        C = self.C
        (first, stop1), (second, stop2) = C.BLOCKS
        pos = np.full((C.N_FRAMES, 4, 2), np.nan)
        team = np.array([0, 0, 1, 1])
        for start, stop, home_x, away_x in ((first, stop1, 5.0, 100.0), (second, stop2, 100.0, 5.0)):
            pos[start:stop, :2, 0] = (home_x, 50.0)  # deepest player + one at the halfway line
            pos[start:stop, 2:, 0] = (away_x, 50.0)
        home_left = C.home_left_frames(pos, team)
        self.assertTrue(home_left[first:stop1].all() and home_left[:first].all())
        self.assertTrue(home_left[stop1:second].all())  # halftime keeps the latest half
        self.assertFalse(home_left[second:].any())

    def test_masked_windows_centre_gap_lengths_in_observed_live_runs(self):
        C = self.C
        (first, _), (second, _) = C.BLOCKS
        ball = np.zeros((C.N_FRAMES, 2))
        labels = np.full(C.N_FRAMES, "D", dtype=object)
        ball[first + 1000:first + 1200] = np.nan  # the only live gap (200 frames)
        labels[first + 900:first + 1300] = "H"
        labels[first + 5000:first + 6000] = "A"  # long observed run: full gap length
        labels[second + 100:second + 250] = "H"  # 150-frame run: capped to 130
        labels[second + 500:second + 640] = "H"  # shorter than 150: no window
        self.assertEqual(C.masked_windows(ball, labels),
                         [[first + 5400, first + 5600, "d"], [second + 110, second + 240, "t"]])

    def test_static_cells_need_recurrence_across_blocks(self):
        C = self.C
        frames = [block * 15000 + i for block in range(5) for i in range(10)]
        cands = {}
        for k, f in enumerate(frames):
            row = {name: [] for name in C.PILOT_DETECTORS}
            row["S3"].append([0.5, 1.0, 1.0, 5.0, 5.0, [10.2, 20.7]])  # recurring clutter
            row["L3"].append([0.04, 1.0, 1.0, 5.0, 5.0, [40.2, 30.1]])  # below 0.05: ignored
            row["S2"].append([0.9, 1.0, 1.0, 5.0, 5.0, [60.0 + k, 10.0]])  # moving ball
            if f < 30000:
                row["X2"].append([0.9, 1.0, 1.0, 5.0, 5.0, [80.5, 5.5]])  # only two 10-minute blocks
            cands[f] = row
        self.assertEqual(C.static_cells(cands, frames), [[10.5, 20.5, 50]])


if __name__ == "__main__":
    unittest.main()
