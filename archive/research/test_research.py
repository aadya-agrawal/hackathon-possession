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
from archive.research.external_soccertrack import (
    _half_actions,
    build_soccertrack_alignment,
    build_touch_review_workspace,
    make_touch_review_handler,
    select_blinded_touch_windows,
    validate_touch_review_task,
    validate_touch_review_workspace,
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
from archive.research.external_player_stress import (
    choose_player_stress_windows, corrupt_player_tracks, local_stress_effect,
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
from archive.research.external_spell_localization import (
    annotate_spurious, localization_summary, matched_spell_pairs, spurious_lineage,
    control_cue_arrays, cue_group_summary, cue_rank_contrasts, control_start_diagnosis,
    weak_transition_restore_values,
    recovery_regressions, high_iou_regressions,
    player_quality_features,
    diagnose_rebased_regressions,
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
from archive.research.external_contact_audit import (
    GROUP_QUOTAS, _provider_context, select_blinded_cases,
    validate_contact_audit_task, validate_contact_audit_workspace,
)
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


ROOT = Path(__file__).resolve().parents[2]


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


class ExternalSoccerTrackTests(unittest.TestCase):
    def test_prepares_blinded_external_touch_review_without_prefilling_truth(self):
        labels = (
            ["PASS"] * 6 + ["BALL PLAYER BLOCK"] * 6
            + ["THROW IN"] * 4 + ["DRIVE"] * 2
        )
        actions = [
            {
                "gameTime": "1 - 00:00", "frame": 101 + i * 300,
                "position": str((100 + i * 300) * 40),
                "label": label, "team": "left", "player_id": "123",
            }
            for i, label in enumerate(labels)
        ]
        selected = select_blinded_touch_windows(actions, 20000)
        self.assertEqual(len(selected), 24)
        self.assertGreaterEqual(
            min(right["center_frame"] - left["center_frame"]
                for left, right in zip(selected, selected[1:])), 100
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bas" / "117092").mkdir(parents=True)
            video_dir = root / "videos" / "117092"
            video_dir.mkdir(parents=True)
            (video_dir / "117092_panorama_1st_half.mp4").write_bytes(b"fixture")
            bas_path = root / "bas" / "117092" / "117092_12_class_events.json"
            bas_path.write_text(json.dumps({
                "match_id": "117092", "fps": 25, "actions": actions,
            }))
            alignment = root / "alignment.json"
            alignment.write_text(json.dumps({
                "dataset": "SoccerTrack v2", "match": "117092",
                "half": 1, "physical_contact_verified": False,
                "bas_sha256": hashlib.sha256(bas_path.read_bytes()).hexdigest(),
                "video": {
                    "frames": 20000,
                    "sha256": hashlib.sha256(b"fixture").hexdigest(),
                },
            }))
            workspace = root / "review"
            summary = build_touch_review_workspace(
                root, "117092", 1, alignment, workspace,
            )
            self.assertEqual(summary["total_cases"], 24)
            self.assertFalse(summary["physical_touch_review_complete"])
            self.assertEqual(
                validate_touch_review_workspace(workspace)["review_status"],
                {"unreviewed": 24},
            )
            task = json.loads(
                (workspace / "tasks" / "case-01.json").read_text()
            )
            self.assertNotIn("event", task)
            self.assertNotIn("action_index", task)
            self.assertIsNone(task["review"]["contact_team"])
            ambiguous = {
                **task,
                "review": {
                    "status": "verified", "reviewer": "tester",
                    "permission_confirmed": True,
                    "contact_team": "ambiguous",
                    "contact_frame": task["center_frame"],
                    "actor_id": None, "confidence": 0.8,
                    "notes": "Touch visible; kit unclear.",
                },
            }
            self.assertEqual(validate_touch_review_task(ambiguous), "verified")
            ambiguous["review"]["contact_frame"] = None
            with self.assertRaisesRegex(ValueError, "marked contact frame"):
                validate_touch_review_task(ambiguous)
            ambiguous["review"]["contact_frame"] = task["center_frame"]
            ambiguous["review"]["confidence"] = 0
            self.assertEqual(validate_touch_review_task(ambiguous), "verified")
            ambiguous["review"]["confidence"] = None
            self.assertEqual(validate_touch_review_task(ambiguous), "verified")
            ambiguous["review"]["confidence"] = 2
            with self.assertRaisesRegex(ValueError, "optional confidence"):
                validate_touch_review_task(ambiguous)
            with self.assertRaisesRegex(ValueError, "still needs"):
                validate_touch_review_workspace(
                    workspace, require_reviewed=True,
                )
            with self.assertRaises(FileExistsError):
                build_touch_review_workspace(
                    root, "117092", 1, alignment, workspace,
                )

            from http.client import HTTPConnection
            from http.server import ThreadingHTTPServer
            from threading import Thread

            handler = make_touch_review_handler(
                workspace, video_dir / "117092_panorama_1st_half.mp4",
                "local-test-token",
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = Thread(target=server.serve_forever)
            thread.start()
            try:
                def request(method, path, body=None, headers=None):
                    connection = HTTPConnection(
                        "127.0.0.1", server.server_port, timeout=5
                    )
                    try:
                        connection.request(
                            method, path, body=body, headers=headers or {}
                        )
                        response = connection.getresponse()
                        return (
                            response.status, response.read(),
                            response.getheader("Content-Range"),
                        )
                    finally:
                        connection.close()

                status, body, _ = request("GET", "/api/index")
                self.assertEqual(status, 200)
                self.assertEqual(len(json.loads(body)["cases"]), 24)
                status, body, _ = request("GET", "/")
                self.assertEqual(status, 200)
                self.assertIn(b"Fullscreen video", body)
                self.assertIn(b"Team unknown", body)
                self.assertIn(b"Touch at this frame", body)
                self.assertIn(b"Can't tell", body)
                self.assertIn(b"local-test-token", body)
                status, body, _ = request(
                    "GET", "/api/task?case=case-01"
                )
                self.assertEqual(status, 200)
                self.assertNotIn("event", json.loads(body))
                status, _, _ = request(
                    "GET", "/selection_private.json"
                )
                self.assertEqual(status, 404)
                status, body, content_range = request(
                    "GET", "/media", headers={"Range": "bytes=0-2"}
                )
                self.assertEqual(status, 206)
                self.assertEqual(body, b"fix")
                self.assertEqual(content_range, "bytes 0-2/7")
                with mock.patch(
                    "archive.research.external_soccertrack.MAX_MEDIA_CHUNK", 3
                ):
                    status, body, content_range = request(
                        "GET", "/media", headers={"Range": "bytes=0-"}
                    )
                self.assertEqual(status, 206)
                self.assertEqual(body, b"fix")
                self.assertEqual(content_range, "bytes 0-2/7")
                review = {
                    "status": "verified", "reviewer": "tester",
                    "permission_confirmed": True,
                    "contact_team": "none", "contact_frame": None,
                    "actor_id": None, "confidence": 0.8,
                    "notes": "No visible touch.",
                }
                payload = json.dumps({"review": review})
                status, _, _ = request(
                    "POST", "/api/task?case=case-01",
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(status, 403)
                invalid_review = {
                    **review, "contact_team": "left",
                    "contact_frame": None,
                }
                status, body, _ = request(
                    "POST", "/api/task?case=case-01",
                    body=json.dumps({"review": invalid_review}),
                    headers={
                        "Content-Type": "application/json",
                        "X-Review-Token": "local-test-token",
                    },
                )
                self.assertEqual(status, 400)
                self.assertIn(
                    "marked contact frame",
                    json.loads(body)["error"],
                )
                status, _, _ = request(
                    "POST", "/api/task?case=case-01",
                    body=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Review-Token": "local-test-token",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(
                    validate_touch_review_workspace(workspace)[
                        "review_status"
                    ],
                    {"verified": 1, "unreviewed": 23},
                )
                unsure = {
                    **review, "status": "unverifiable",
                    "contact_team": None, "confidence": None,
                    "notes": None,
                }
                status, _, _ = request(
                    "POST", "/api/task?case=case-01",
                    body=json.dumps({"review": unsure}),
                    headers={
                        "Content-Type": "application/json",
                        "X-Review-Token": "local-test-token",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(
                    validate_touch_review_workspace(workspace)[
                        "review_status"
                    ],
                    {"unverifiable": 1, "unreviewed": 23},
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_bas_event_frame_is_one_based_and_shipped_class_is_uppercase(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actions.json"
            action = {
                "gameTime": "1 - 00:00", "frame": 1,
                "position": "0", "label": "PASS",
                "team": "left", "player_id": "123",
            }
            data = {"match_id": "117092", "fps": 25, "actions": [action]}
            path.write_text(json.dumps(data))
            self.assertEqual(_half_actions(path, "117092", 1, 2, 25), [action])
            data["actions"].append({
                "gameTime": "1 - 00:00", "frame": 2,
                "position": "40", "label": "GOAL", "team": "right",
            })
            path.write_text(json.dumps(data))
            self.assertEqual(len(_half_actions(path, "117092", 1, 2, 25)), 2)
            data["actions"][0]["frame"] = 0
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "1-based GSR"):
                _half_actions(path, "117092", 1, 2, 25)
            data["actions"][0]["frame"] = 1
            data["actions"][0]["label"] = "Pass"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "unknown ball action"):
                _half_actions(path, "117092", 1, 2, 25)

    def test_aligns_numeric_bas_actor_to_integer_gsr_actor(self):
        try:
            import ijson
        except ImportError:
            self.skipTest("optional SoccerTrack streaming dependency is absent")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bas" / "117092").mkdir(parents=True)
            (root / "gsr" / "117092").mkdir(parents=True)
            video_dir = root / "videos" / "117092"
            video_dir.mkdir(parents=True)
            (video_dir / "117092_panorama_1st_half.mp4").write_bytes(b"fixture")
            (root / "bas" / "117092" / "117092_12_class_events.json").write_text(
                json.dumps({
                    "match_id": "117092", "fps": 25,
                    "actions": [
                        {
                            "gameTime": "1 - 00:00", "frame": 1,
                            "position": "0", "label": "PASS",
                            "team": "left", "player_id": "123",
                        },
                        {
                            "gameTime": "1 - 00:00", "frame": 2,
                            "position": "40", "label": "GOAL",
                            "team": "right",
                        },
                    ],
                })
            )
            (root / "gsr" / "117092" / "117092_1st.json").write_text(
                json.dumps({
                    "info": {"frame_rate": 25, "seq_length": 2},
                    "images": [{
                        "image_id": "3901000001", "width": 100, "height": 100,
                    }, {
                        "image_id": "3901000002", "width": 100, "height": 100,
                    }],
                    "annotations": [{
                        "image_id": "3901000001", "supercategory": "object",
                        "attributes": {
                            "role": "player", "team": "left",
                            "player_id": 123,
                        },
                    }],
                })
            )
            with mock.patch(
                "archive.research.external_soccertrack.inspect_video",
                return_value=SimpleNamespace(
                    frames=2, fps=25, width=100, height=100,
                ),
            ):
                report = build_soccertrack_alignment(root, "117092", 1)
            self.assertEqual(report["actor_present_at_exact_frame"], 1)
            self.assertEqual(report["actor_team_agrees_at_exact_frame"], 1)
            self.assertFalse(report["physical_contact_verified"])


class PlayerIdentityTests(unittest.TestCase):
    def _arrays(self):
        home = np.tile([[[0., .1], [1., .1]]], (3, 1, 1)).astype(np.float32)
        away = np.tile([[[.5, .9]]], (3, 1, 1)).astype(np.float32)
        ball = np.asarray([[.4, .1], [.6, .1], [.6, .1]], dtype=np.float32)
        return home, away, ball

    def test_nearest_velocity_uses_current_track_and_survives_ball_gaps(self):
        home, away, ball = self._arrays()
        control = control_features_from_arrays(home, away, ball, 10, default_config())
        np.testing.assert_array_equal(control.nearest_player_velocity[1, HOME], [0, 0])
        ball[1] = np.nan
        control = control_features_from_arrays(home, away, ball, 10, default_config())
        self.assertTrue(np.isnan(control.nearest_player_velocity[1]).all())
        np.testing.assert_array_equal(control.nearest_player_velocity[2, HOME], [0, 0])
        summary, features = player_quality_features(home, ball, np.ones(3), 10, 12, 1)
        self.assertEqual(summary["legacy_proxy_missing_when_track_valid"], 1)
        self.assertEqual(features["implausible_nearest_step"][2], 0)

    def test_atd_and_array_adapters_share_identity_aware_velocity(self):
        home, away, ball = self._arrays()
        ball[1] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            meta = Path(directory) / "tracking.xml"
            raw = Path(directory) / "tracking.txt"
            meta.write_text(
                '<Root><Team id="h"><Name>Home</Name></Team><Team id="a"><Name>Away</Name></Team>'
                '<Player teamId="h"/><Player teamId="h"/><Player teamId="a"/></Root>'
            )
            raw.write_text(
                '1:0,0.1,0;1,0.1,0;0.5,0.9,0:0.4,0.1,0\n'
                '2:0,0.1,0;1,0.1,0;0.5,0.9,0:NaN,NaN,NaN\n'
                '3:0,0.1,0;1,0.1,0;0.5,0.9,0:0.6,0.1,0\n'
            )
            tracking = extract_control_features(raw, load_tracking_metadata(meta), ball, (1, 1), fps=10)
            arrays = control_features_from_arrays(home, away, ball, 10, default_config())
        np.testing.assert_allclose(tracking.nearest_player_velocity, arrays.nearest_player_velocity, equal_nan=True)
        np.testing.assert_array_equal(tracking.nearest_player_velocity[2, HOME], [0, 0])

    def test_metadata_rejects_unknown_or_ambiguous_team_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.xml"
            path.write_text(
                '<Root><Team id="h"><Name>Home</Name></Team><Team id="a"><Name>Away</Name></Team>'
                '<Player teamId="other"/></Root>'
            )
            with self.assertRaisesRegex(ValueError, "unknown team ID"):
                load_tracking_metadata(path)
            path.write_text(
                '<Root><Team id="h"><Name>Home</Name></Team><Team id="h"><Name>Away</Name></Team>'
                '<Player teamId="h"/></Root>'
            )
            with self.assertRaisesRegex(ValueError, "present and distinct"):
                load_tracking_metadata(path)

    def test_player_velocity_rejects_birth_gaps_and_implausible_steps(self):
        previous = np.asarray([[np.nan, 0], [0, 0], [0, 0]])
        current = np.asarray([[1., 0], [1., 0], [3., 0]])
        velocity = player_velocity_steps(previous, current, 5, 12)
        self.assertTrue(np.isnan(velocity[0]).all())
        np.testing.assert_array_equal(velocity[1], [5, 0])
        self.assertTrue(np.isnan(velocity[2]).all())


class PlayerStressTests(unittest.TestCase):
    def test_local_stress_region_stays_within_period_and_keeps_unknown_out_of_accuracy(self):
        periods = np.r_[np.ones(8), np.full(8, 2)]
        truth = np.asarray([HOME] * 8 + [UNKNOWN] * 8)
        original = np.full(16, HOME)
        stressed = original.copy()
        stressed[7:10] = AWAY
        windows = [{"start_row": 6, "end_row": 8, "period": 1}]
        result = local_stress_effect(original, stressed, truth, periods, windows, 3)
        self.assertEqual(result["direct_frames"], 2)
        self.assertEqual(result["nearby_frames"], 5)
        self.assertEqual(result["nearby_label_changes"], 1)
        self.assertEqual(result["known_live_nearby_frames"], 5)
        self.assertAlmostEqual(result["known_live_accuracy_after"], .8)
        with self.assertRaisesRegex(ValueError, "crosses periods"):
            local_stress_effect(original, stressed, truth, periods,
                                [{"start_row": 7, "end_row": 9, "period": 1}], 3)

    def _fixture(self):
        n = 120
        ball = np.tile([5., 5.], (n, 1))
        home = np.tile([[[5.2, 5.], [8., 5.]]], (n, 1, 1))
        away = np.tile([[[9., 5.], [np.nan, np.nan]]], (n, 1, 1))
        data = {
            "ball": ball, "home": home, "away": away,
            "periods": np.r_[np.ones(60), np.full(60, 2)],
            "fps": 25.,
        }
        config = {
            "window_seconds": .2, "windows_per_game": 4,
            "near_ball_metres": 2., "minimum_players_per_team": 1,
            "jitter_std_metres": .25, "seed": 2027,
            "modes": ["nearest_dropout", "near_player_jitter", "team_swap"],
        }
        return data, config

    def test_player_corruption_windows_are_deterministic_nonoverlapping_and_period_local(self):
        data, config = self._fixture()
        windows = choose_player_stress_windows(data, config)
        self.assertEqual(windows, choose_player_stress_windows(data, config))
        self.assertEqual(len(windows), 4)
        for item in windows:
            self.assertLessEqual(item["end_row"] - item["start_row"], 5)
            self.assertTrue(np.all(
                data["periods"][item["start_row"]:item["end_row"]] == item["period"]
            ))
        self.assertTrue(all(b["start_row"] >= a["end_row"] + 2
                            for a, b in zip(windows[:-1], windows[1:])))
        data["ball"][:] = np.nan
        with self.assertRaisesRegex(ValueError, "insufficient valid observed"):
            choose_player_stress_windows(data, config)

    def test_dropout_jitter_and_team_swap_change_only_selected_player_windows(self):
        data, config = self._fixture()
        windows = choose_player_stress_windows(data, config)
        dropped, dropped_windows = corrupt_player_tracks(data, windows, config, "nearest_dropout")
        self.assertEqual(len(dropped_windows), 4)
        self.assertTrue(np.isnan(dropped["home"][windows[0]["start_row"]:windows[0]["end_row"], 0]).all())
        np.testing.assert_array_equal(dropped["away"], data["away"])
        jittered, _ = corrupt_player_tracks(data, windows, config, "near_player_jitter")
        self.assertFalse(np.array_equal(jittered["home"], data["home"]))
        np.testing.assert_allclose(jittered["home"][:, 1], data["home"][:, 1])
        repeated, _ = corrupt_player_tracks(data, windows, config, "near_player_jitter")
        np.testing.assert_array_equal(jittered["home"], repeated["home"])
        swapped, used = corrupt_player_tracks(data, windows, config, "team_swap")
        self.assertEqual(len(used), 4)
        start, stop = used[0]["start_row"], used[0]["end_row"]
        np.testing.assert_array_equal(swapped["away"][start:stop, 1], data["home"][start:stop, 0])
        self.assertTrue(np.isnan(swapped["home"][start:stop, 0]).all())
        np.testing.assert_array_equal(data["home"][:, 0], np.tile([5.2, 5.], (120, 1)))

    def test_stress_rejects_bad_modes_or_empty_receiving_slots(self):
        data, config = self._fixture()
        windows = choose_player_stress_windows(data, config)
        with self.assertRaisesRegex(ValueError, "not frozen"):
            corrupt_player_tracks(data, windows, config, "arbitrary")
        data["away"][:, 1] = [9., 6.]
        with self.assertRaisesRegex(ValueError, "could not corrupt"):
            corrupt_player_tracks(data, windows, config, "team_swap")


class SpellLocalizationTests(unittest.TestCase):
    def test_contact_motion_distinguishes_straight_transit_and_direction_change(self):
        config = {**default_config(), "strong_control_distance_metres": 1.,
                  "semi_markov_strong_transition_window_frames": 10,
                  "semi_markov_strong_transition_minimum_frames": 2}
        home = np.tile([[[4., 0.]]], (20, 1, 1)).astype(np.float32)
        away = np.tile([[[40., 0.]]], (20, 1, 1)).astype(np.float32)
        outcomes = []
        for turn in (False, True):
            x = np.arange(20, dtype=np.float32) * .5
            if turn:
                x[9:] = 4 - np.arange(1, 12) * .5
            xy = np.column_stack((x, np.zeros(20))).astype(np.float32)
            trajectory = SimpleNamespace(xy=xy, source=np.ones(20, dtype=np.int8))
            control = control_features_from_arrays(home, away, xy, 10, config)
            evidence = Evidence(
                control=np.tile([1., 0.], (20, 1)), dead=np.zeros(20),
                strong_control=control.nearest_distances[:, :, 0] <= 1,
            )
            cues = control_cue_arrays(home, away, trajectory, control, evidence, config, 10, 10)
            outcomes.append(cues["contact_motion_disturbance"][3, HOME])
            self.assertEqual(cues["future_team_advantage_fraction"][3, HOME], 1)
            self.assertEqual(cues["future_team_advantage_fraction"][3, AWAY], 0)
            self.assertAlmostEqual(cues["contact_offset_seconds"][3, HOME], .5)
        self.assertAlmostEqual(outcomes[0], 0)
        self.assertAlmostEqual(outcomes[1], 1)

    def test_contact_motion_remains_unknown_when_contact_window_crosses_missing_ball(self):
        home, away, trajectory, control, evidence, config = self._control_fixture()
        cues = control_cue_arrays(home, away, trajectory, control, evidence, config, 10, 4)
        self.assertTrue(np.isnan(cues["contact_motion_disturbance"][1, HOME]))

    def test_recovery_diagnosis_detects_support_cost_inversion_and_supported_delay(self):
        baseline = np.asarray([HOME] * 5 + [AWAY] * 15)
        candidate = np.asarray([HOME] * 8 + [AWAY] * 12)
        source = np.ones(20, dtype=np.int8)
        source[4:8] = 2
        references = [{
            "row": 5, "period": 1, "incoming": AWAY, "outgoing": HOME,
            "boundary_start_row": 5, "boundary_end_row": 5,
            "support_end_row": 15, "reference_support_seconds": 1.,
        }]
        cues = {name: {
            "strong_future": np.zeros((20, 2), dtype=bool),
            "long_strong_count": np.full((20, 2), 4),
            "control_margin": np.zeros((20, 2)),
            "nearest_distance_metres": np.ones((20, 2)),
        } for name in ("baseline", "candidate")}
        cues["baseline"]["strong_future"][5, AWAY] = True
        configs = {"baseline": default_config(), "candidate": {
            **default_config(), "semi_markov_live_transition_cost": 20.,
            "semi_markov_strong_transition_cost": 5.,
            "semi_markov_strong_transition_minimum_frames": 4,
        }}
        result = recovery_regressions(
            {"baseline": baseline, "candidate": candidate}, cues, source,
            np.ones(20), references, configs, 10,
        )
        summary = result["groups"]["regression"]
        self.assertEqual(summary["events"], 1)
        self.assertEqual(summary["support_cost_inversion"], 1)
        self.assertEqual(summary["long_window_would_enable_support"], 1)
        self.assertEqual(summary["long_window_at_baseline_fraction"], 0)
        self.assertEqual(summary["candidate_delayed_handover"], 1)
        self.assertAlmostEqual(result["details"][0]["models"]["candidate"]["boundary_offset_seconds"], .3)
        references[0]["support_end_row"] = 7
        result = recovery_regressions(
            {"baseline": baseline, "candidate": candidate}, cues, source,
            np.ones(20), references, configs, 10,
        )
        self.assertEqual(result["groups"]["regression"]["candidate_delayed_handover"], 0)
        self.assertEqual(result["groups"]["regression"]["scorable_at_0.5s"], 0)

    def test_high_iou_diagnosis_distinguishes_boundary_drift_from_lost_spells(self):
        result = high_iou_regressions(
            [(0, 20, HOME), (20, 40, AWAY)],
            [(0, 20, HOME), (20, 40, AWAY)],
            [(0, 30, HOME), (30, 40, AWAY)], np.ones(40), 10,
        )
        self.assertEqual(result["lost"], 2)
        self.assertEqual(result["lost_but_still_matched_at_0.5"], 2)
        self.assertEqual(result["late_starts"], 1)
        self.assertEqual(result["late_ends"], 1)
        self.assertEqual(result["gained"], 0)
        with self.assertRaisesRegex(ValueError, "previously inspected"):
            diagnose_rebased_regressions("/missing", "/missing", "/missing",
                                         game="Sample_Game_3")

    def test_fixed_weak_transition_restore_changes_only_regular_switch_cost(self):
        baseline = default_config()
        source = {key: baseline[key] for key in CORE_CALIBRATION_KEYS}
        source.update(semi_markov_live_transition_cost=20., semi_markov_strong_transition_cost=5.)
        restored = weak_transition_restore_values(baseline, source)
        self.assertEqual([key for key in source if restored[key] != source[key]],
                         ["semi_markov_live_transition_cost"])
        self.assertEqual(restored["semi_markov_strong_transition_cost"], 5.)
        self.assertEqual(source["semi_markov_live_transition_cost"], 20.)
        with self.assertRaisesRegex(ValueError, "reduced source switch cost"):
            weak_transition_restore_values(baseline, restored)

    def _control_fixture(self):
        config = {**default_config(), "strong_control_distance_metres": 0.45,
                  "semi_markov_strong_transition_window_frames": 4,
                  "semi_markov_strong_transition_minimum_frames": 2}
        home = np.tile([[[0., 0.], [1., 0.]]], (6, 1, 1)).astype(np.float32)
        away = np.tile([[[4., 0.]]], (6, 1, 1)).astype(np.float32)
        xy = np.column_stack(([0.4, 0.6, 0.6, 0.6, 2., 2.], np.zeros(6))).astype(np.float32)
        trajectory = SimpleNamespace(xy=xy, source=np.asarray([1, 1, 1, 2, 1, 1]))
        control = control_features_from_arrays(home, away, xy, 10, config)
        evidence = Evidence(np.zeros((6, 2)), np.zeros(6), np.zeros((6, 2), dtype=bool))
        return home, away, trajectory, control, evidence, config

    def test_control_motion_follows_current_track_across_nearest_identity_change(self):
        home, away, trajectory, control, evidence, config = self._control_fixture()
        cues = control_cue_arrays(home, away, trajectory, control, evidence, config, 10, 4)
        self.assertAlmostEqual(cues["relative_speed_mps"][1, HOME], 2., places=5)
        self.assertAlmostEqual(cues["velocity_proxy_error_mps"][1, HOME], 10., places=5)
        self.assertEqual(cues["nearest_identity_switch"][1, HOME], 1)
        self.assertTrue(np.isnan(cues["ball_speed_mps"][0, HOME]))

    def test_motion_does_not_bridge_missing_observations_or_count_unknown_as_no_contact(self):
        home, away, trajectory, control, evidence, config = self._control_fixture()
        cues = control_cue_arrays(home, away, trajectory, control, evidence, config, 10, 4)
        self.assertTrue(np.isnan(cues["relative_speed_mps"][3, HOME]))
        self.assertTrue(np.isnan(cues["relative_speed_mps"][4, HOME]))
        self.assertAlmostEqual(cues["same_player_contact_fraction"][1, HOME], 2 / 3, places=5)
        self.assertAlmostEqual(cues["same_player_contact_fraction"][0, HOME], 1 / 3, places=5)
        self.assertEqual(cues["team_contact_fraction"][0, HOME], 1)
        self.assertTrue(np.isnan(cues["same_player_contact_fraction"][3, HOME]))
        short = control_cue_arrays(home, away, trajectory, control, evidence, config, 10, 20)
        self.assertEqual(short["team_contact_fraction"].shape, (6, 2))

    def test_rank_contrasts_handle_ties_and_preserve_missing_denominators(self):
        items = [{"start_row": row, "state": HOME} for row in range(3)]
        arrays = {"speed": np.asarray([[1, 0], [1, 0], [np.nan, 0]])}
        result = cue_rank_contrasts(items[:1], items[1:], arrays)["speed"]
        self.assertEqual(result["probability_error_greater"], 0.5)
        self.assertEqual(result["matched_available"], 1)
        result = cue_group_summary(items, arrays)
        self.assertEqual(result["events"], 3)
        self.assertEqual(result["metrics"]["speed"]["available"], 2)
        self.assertIsNone(cue_rank_contrasts([], items, arrays)["speed"]["probability_error_greater"])
        boolean_arrays = {"support": np.asarray([[True, False], [False, False], [True, False]])}
        self.assertAlmostEqual(cue_group_summary(items, boolean_arrays)["metrics"]["support"]["mean"], 2 / 3)
        self.assertEqual(
            cue_rank_contrasts(items[:1], items[1:], boolean_arrays)["support"]["probability_error_greater"],
            .75,
        )

    def test_diagnosis_separates_dead_references_and_observed_live_handovers(self):
        labels = np.asarray([HOME] * 5 + [AWAY] * 5 + [HOME] * 5)
        periods = np.asarray([1] * 10 + [2] * 5)
        truth = np.asarray([HOME] * 5 + [DEAD] * 5 + [AWAY] * 5)
        details = [
            {"prediction_index": 1, "start_row": 5, "stop_row": 10, "state": AWAY, "lineage": "new_prediction"},
            {"prediction_index": 2, "start_row": 10, "stop_row": 15, "state": HOME, "lineage": "new_prediction"},
        ]
        diagnosis = control_start_diagnosis(
            [(0, 5, HOME)], [(0, 5, HOME), (5, 10, AWAY), (10, 15, HOME)],
            labels, details, {"speed": np.ones((15, 2))}, np.ones(15), truth, periods, 10, 0.5,
        )
        self.assertEqual(diagnosis["groups"]["introduced_during_reference_dead"]["events"], 1)
        self.assertEqual(diagnosis["groups"]["introduced_wrong_live_team"]["events"], 1)
        self.assertEqual(diagnosis["observed_handover_groups"]["introduced_wrong_live_team"]["events"], 0)
        self.assertEqual(details[1]["previous_predicted_state"], UNKNOWN)

    def test_lineage_separates_existing_errors_and_new_degradation(self):
        reference = [(0, 10, HOME)]
        baseline = [(0, 6, HOME), (20, 30, AWAY)]
        candidate = [(0, 4, HOME), (4, 10, HOME), (20, 30, AWAY)]
        result = spurious_lineage(reference, baseline, candidate, 0.5, 0.5)
        self.assertEqual(result["baseline_spurious"], 1)
        self.assertEqual(result["candidate_spurious"], 2)
        self.assertEqual(result["shared_spurious"], 1)
        self.assertEqual(result["introduced_or_degraded"], 1)
        self.assertEqual(result["net_increase"], 1)
        self.assertIn("previously_matched_now_spurious",
                      [item["lineage"] for item in result["details"]])

    def test_matching_does_not_reuse_a_reference_or_prediction(self):
        pairs = matched_spell_pairs(
            [(0, 10, HOME)], [(0, 8, HOME), (2, 10, HOME)], 0.5
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1], 0)

    def test_cue_locations_use_period_local_time_and_source_state(self):
        source = np.ones(60, dtype=np.int8)
        source[10:20] = 2
        periods = np.r_[np.ones(40), np.full(20, 2)]
        lineage = {"details": [
            {"start_row": 21, "stop_row": 25, "state": HOME,
             "lineage": "new_prediction"},
            {"start_row": 12, "stop_row": 16, "state": AWAY,
             "lineage": "new_prediction"},
        ]}
        details = annotate_spurious(
            lineage, source, periods, np.zeros(60, dtype=np.int8),
            np.asarray([20, 40]), np.asarray([20]), np.asarray([23]), 10,
        )
        self.assertAlmostEqual(details[0]["seconds_after_last_reappearance"], 0.1)
        self.assertIsNone(details[1]["seconds_after_last_reappearance"])
        self.assertEqual(details[1]["start_source"], 2)
        summary = localization_summary(details, {
            "reappearance_bins_seconds": [0.5, 1.0, 3.0],
            "tag_bins_seconds": [1.0, 3.0],
        })
        self.assertEqual(summary["introduced_or_degraded"]["post_reappearance_0.5s"], 1)
        self.assertEqual(summary["introduced_or_degraded"]["starts_in_interpolation"], 1)
        self.assertEqual(summary["introduced_or_degraded"]["correct_reference_team_at_start"], 1)
        self.assertEqual(summary["introduced_or_degraded"]["wrong_reference_team_at_start"], 1)


class ExternalContactAuditTests(unittest.TestCase):
    def _task(self):
        return {
            "schema_version": 1, "dataset": "Metrica sample-data",
            "case_id": "case-01", "period": 1, "center_frame": 100,
            "start_frame": 50, "end_frame": 150, "fps": 25.,
            "video_available": False,
            "review_instructions": "Independent video only; no provider events before review.",
            "review": {
                "status": "unreviewed", "reviewer": None, "video_source": None,
                "permission_confirmed": None, "contact_team": None,
                "contact_frame": None, "possession_boundary_frame": None,
                "confidence": None, "notes": None,
            },
        }

    def test_blinded_source_only_selection_is_deterministic_and_balanced(self):
        candidates = [
            {"group": group, "row": index * 10000 + slot * 200,
             "period": 1 if slot % 2 else 2}
            for index, group in enumerate(GROUP_QUOTAS)
            for slot in range(24)
        ]
        selected = select_blinded_cases(candidates)
        self.assertEqual(selected, select_blinded_cases(candidates))
        self.assertEqual(
            {group: sum(item["group"] == group for item in selected)
             for group in GROUP_QUOTAS}, GROUP_QUOTAS,
        )
        for index, item in enumerate(selected):
            self.assertTrue(all(
                item["period"] != prior["period"]
                or abs(item["row"] - prior["row"]) >= 100
                for prior in selected[:index]
            ))
        with self.assertRaisesRegex(ValueError, "only 0 source-only candidates"):
            select_blinded_cases([])
        with self.assertRaisesRegex(ValueError, "nonoverlapping review windows"):
            select_blinded_cases(
                [{"group": group, "row": slot, "period": 1}
                 for group in GROUP_QUOTAS for slot in range(24)],
            )

    def test_review_cards_hide_model_and_event_context_until_video_verified(self):
        task = self._task()
        self.assertEqual(validate_contact_audit_task(task), "unreviewed")
        with self.assertRaisesRegex(ValueError, "needs independent review"):
            validate_contact_audit_task(task, require_reviewed=True)
        with self.assertRaisesRegex(ValueError, "unblinded"):
            validate_contact_audit_task({**task, "provider_events": []})
        task["review"]["contact_team"] = "H"
        with self.assertRaisesRegex(ValueError, "prefilled labels"):
            validate_contact_audit_task(task)
        task = self._task()
        task["review"].update({
            "status": "verified", "reviewer": "reviewer", "video_source": "external synced video",
            "contact_team": "H", "contact_frame": 100,
            "possession_boundary_frame": 102, "confidence": .8,
        })
        with self.assertRaisesRegex(ValueError, "independent permitted video"):
            validate_contact_audit_task(task)
        task["review"]["permission_confirmed"] = True
        self.assertEqual(validate_contact_audit_task(task, require_reviewed=True), "verified")
        task["review"]["contact_frame"] = None
        with self.assertRaisesRegex(ValueError, "must match a definite"):
            validate_contact_audit_task(task)
        task["review"]["contact_frame"] = 100
        task["review"]["possession_boundary_frame"] = 151
        with self.assertRaisesRegex(ValueError, "outside the review window"):
            validate_contact_audit_task(task)

    def test_provider_event_proximity_is_stored_separately_and_is_not_touch_truth(self):
        data = {"frames": np.arange(100, 201), "periods": np.ones(101, dtype=np.int8)}
        selected = [{"group": "turnover_delayed", "case_id": "case-01",
                     "row": 20, "period": 1, "receiver_team": HOME}]
        events = [
            {"type": "RECOVERY", "frame": 121, "end_frame": 121, "period": 1, "team": "home"},
            {"type": "PASS", "frame": 115, "end_frame": 128, "period": 1, "team": "away"},
            {"type": "RECOVERY", "frame": 121, "end_frame": 121, "period": 2, "team": "away"},
        ]
        private, summary = _provider_context(events, data, selected)
        self.assertEqual(len(private["case-01"]), 2)
        self.assertEqual(summary["turnover_delayed"]["nearby_provider_recovery_same_team"], 1)
        self.assertEqual(summary["turnover_delayed"]["nearby_provider_pass_end_same_team"], 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks").mkdir()
            (root / "index.json").write_text(json.dumps({
                "schema_version": 1, "dataset": "Metrica sample-data",
                "selection_policy": {
                    "provider_events_used_for_selection": False,
                    "independent_video_available": False,
                },
                "tasks": ["case-01"],
            }))
            (root / "tasks" / "case-01.json").write_text(json.dumps(self._task()))
            self.assertEqual(validate_contact_audit_workspace(root)["review_status"], {"unreviewed": 1})
            with self.assertRaisesRegex(ValueError, "needs independent review"):
                validate_contact_audit_workspace(root, require_reviewed=True)


if __name__ == "__main__":
    unittest.main()
