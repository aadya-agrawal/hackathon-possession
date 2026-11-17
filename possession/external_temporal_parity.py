import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from games import GAME, GAMES
from possession import AWAY, HOME
from possession.cli import load_config
from possession.decoder import CORE_CALIBRATION_KEYS
from possession.external_guarded_calibration import (
    _evaluate, _prepare, _save, evaluate_case, guarded_checks, summarize_evaluation,
)
from possession.external_metrica_validation import (
    _candidate_labels, _file_sha256, _load_events, _scoring_events, config_at_fps,
    load_validation_protocol,
)
from possession.external_reliability_calibration import reliability_overrides


def gap_time_correction(baseline, selection, runtime_fps):
    key = selection["config_key"]
    if key != "max_interpolation_gap_frames":
        raise ValueError("time correction is limited to the declared interpolation-gap parameter")
    if (baseline[key] != selection["selection_frames"]
            or baseline["compliance_provenance"]["target_fps"] != selection["selection_source_fps"]):
        raise ValueError("fixed gap correction differs from its calibration lineage")
    corrected = config_at_fps({key: baseline[key]}, selection["selection_source_fps"], runtime_fps)[key]
    if corrected == baseline[key]:
        raise ValueError("fixed gap correction must change the current frame count")
    return {**baseline, key: corrected}


def temporal_settings(config, fps):
    keys = (
        "max_interpolation_gap_frames", "prediction_history_frames", "prediction_horizon_frames",
        "semi_markov_minimum_live_frames", "semi_markov_minimum_dead_frames",
        "semi_markov_strong_transition_window_frames",
        "semi_markov_strong_transition_minimum_frames", "restart_offset_frames",
        "restart_localization_motion_window_frames", "restart_ownership_minimum_votes",
    )
    return {
        "fps": fps,
        "windows": {
            key: {"frames": config[key], "seconds": config[key] / fps} for key in keys
        },
        "costs": {
            key: config[key] for key in (
                "semi_markov_live_transition_cost", "semi_markov_strong_transition_cost",
                "semi_markov_dead_transition_cost",
            )
        },
    }


def support_horizon_repair_values(baseline, source, preserve_fraction=False,
                                  restore_observed_strong_cost=False):
    window = baseline["semi_markov_strong_transition_window_frames"]
    if window <= source["semi_markov_strong_transition_window_frames"]:
        raise ValueError("support-horizon repair requires a shorter candidate support window")
    values = {
        **source,
        **reliability_overrides({**baseline, **source}, "transitions"),
        "core_uncertain_strong_transition_window_frames": window,
    }
    if preserve_fraction:
        minimum = source["semi_markov_strong_transition_minimum_frames"]
        short_window = source["semi_markov_strong_transition_window_frames"]
        values["core_uncertain_strong_transition_minimum_frames"] = (
            minimum * window + short_window - 1
        ) // short_window
    if restore_observed_strong_cost:
        if baseline["semi_markov_strong_transition_cost"] <= source["semi_markov_strong_transition_cost"]:
            raise ValueError("observed strong-cost restore requires a discounted candidate cost")
        values["semi_markov_strong_transition_cost"] = baseline["semi_markov_strong_transition_cost"]
    return values


def select_touch_control(project, protocol_path, baseline, parity, source_values,
                         soften_touch=False, continuous_touch=False):
    descriptor = json.loads(Path(protocol_path).read_text())
    source_path = project / descriptor["source_repair_report"]
    diagnosis_path = project / descriptor["diagnosis_report"]
    if _file_sha256(source_path) != descriptor["source_repair_sha256"]:
        raise ValueError("touch-control source repair changed")
    if _file_sha256(diagnosis_path) != descriptor["diagnosis_sha256"]:
        raise ValueError("touch-control diagnosis changed")
    source = json.loads(source_path.read_text())
    diagnosis = json.loads(diagnosis_path.read_text())
    if (
        source["baseline_sha256"] != parity["baseline_sha256"]
        or source["values"] != support_horizon_repair_values(baseline, source_values)
        or diagnosis["rebase_report_sha256"] != descriptor["source_repair_sha256"]
        or diagnosis["runtime_fps"] != GAMES[GAME]["fps"]
        or not diagnosis["control_cues_measured"]
        or not diagnosis["original_model_outputs_reproduced"]
        or diagnosis["development_game"] != "Sample_Game_1"
        or descriptor["selection_case"] != "clean:none"
        or descriptor["matched_transition_quantile"] != .1
    ):
        raise ValueError("touch-control pilot requires the fixed Game1 matched-transition diagnosis")
    case = next(item for item in diagnosis["cases"] if item["case"] == descriptor["selection_case"])
    matched = [
        item for item in case["control_diagnosis"]["matched_starts"]
        if item["start_source"] == 1 and item["previous_predicted_state"] in (HOME, AWAY)
    ]
    values = np.asarray([
        item["cues"]["contact_motion_disturbance"] for item in matched
        if item["cues"]["contact_motion_disturbance"] is not None
    ])
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("touch-control threshold requires finite matched-transition observations")
    threshold = float(np.quantile(values, descriptor["matched_transition_quantile"]))
    if not 0 < threshold <= 1:
        raise ValueError("touch-control threshold must be positive and at most one")
    candidate_values = {
        **source["values"], "semi_markov_touch_motion_gate_enabled": True,
        "touch_motion_horizon_frames": baseline["semi_markov_strong_transition_window_frames"],
        "touch_motion_window_frames": source_values["semi_markov_strong_transition_minimum_frames"],
        "touch_motion_minimum": threshold,
    }
    if soften_touch or continuous_touch:
        candidate_values.update({
            "touch_motion_mode": "confidence" if continuous_touch else "soften",
            "touch_motion_weak_transition_cost": baseline["semi_markov_strong_transition_cost"],
        })
    return candidate_values, source, {
        "protocol_sha256": _file_sha256(protocol_path),
        "source_repair_sha256": descriptor["source_repair_sha256"],
        "diagnosis_sha256": descriptor["diagnosis_sha256"],
        "selection_case": descriptor["selection_case"],
        "quantile": descriptor["matched_transition_quantile"],
        "matched_observed_handovers": len(matched),
        "available_motion_samples": len(values),
        "samples_below_threshold": int(np.count_nonzero(values < threshold)),
        "threshold": threshold, "unknown_behavior": "retain_original_support",
        "action": "confidence" if continuous_touch else "soften" if soften_touch else "gate",
    }


def select_owner_flight(project, protocol_path, baseline, parity, source_values,
                        variant_protocol=None):
    extension = json.loads(Path(protocol_path).read_text())
    if (extension["owner_flight_horizon_source"] != "semi_markov_strong_transition_window_frames"
            or extension["owner_flight_weight_source"] != "restart_weight"
            or extension["owner_flight_speed_source"] != "strong_control_relative_speed_metres_per_second"
            or extension["owner_touch_motion_source"] != "metrica_touch_control clean matched q10"):
        raise ValueError("owner-flight design must use its declared fixed source settings")
    touch_path = project / extension["touch_control_protocol"]
    values, _, selection = select_touch_control(
        project, touch_path, baseline, parity, source_values, soften_touch=True,
    )
    report_path = project / extension["source_soft_contact_report"]
    report = load_frozen_diagnostic(
        report_path, extension["source_soft_contact_sha256"], values,
        parity["baseline_sha256"], parity["protocol_sha256"],
    )
    owner_values = {
        **values,
        "semi_markov_owner_flight_enabled": True,
        "owner_flight_horizon_frames": baseline["semi_markov_strong_transition_window_frames"],
        "owner_flight_weight": baseline["restart_weight"],
        "owner_flight_minimum_ball_speed_mps": baseline["strong_control_relative_speed_metres_per_second"],
        "owner_flight_touch_motion_minimum": selection["threshold"],
    }
    if variant_protocol is not None:
        variant = json.loads(Path(variant_protocol).read_text())
        decoded_confirmation = variant.get("owner_flight_confirm_decoded_owner", False)
        sustained_player = variant.get("owner_flight_require_sustained_player", False)
        weaker_weight = variant.get("owner_flight_weight_source") == "tag_weight"
        same_team_reception = variant.get("owner_flight_require_same_team_reception", False)
        stronger_reception = same_team_reception and variant.get("owner_flight_weight_source") == "restart_weight"
        if (
            variant["base_owner_protocol"] != str(Path(protocol_path).resolve().relative_to(project))
            or variant["base_owner_protocol_sha256"] != _file_sha256(protocol_path)
            or variant["owner_flight_reset_on_gap"] is not True
            or not isinstance(decoded_confirmation, bool)
            or not isinstance(sustained_player, bool)
            or (decoded_confirmation and sustained_player)
            or (weaker_weight and not sustained_player)
            or (same_team_reception and not (weaker_weight or stronger_reception))
            or not isinstance(same_team_reception, bool)
            or (variant.get("owner_flight_weight_source") not in (None, "tag_weight", "restart_weight"))
            or (sustained_player and variant.get("owner_flight_contact_minimum_source")
                != "semi_markov_strong_transition_minimum_frames")
        ):
            raise ValueError("owner-flight gap-safe revision must retain the frozen initial protocol")
        previous_values = {
            **owner_values,
            **({"owner_flight_reset_on_gap": True} if decoded_confirmation or sustained_player else {}),
            **({
                "owner_flight_require_sustained_player": True,
                "owner_flight_contact_minimum_frames":
                    source_values["semi_markov_strong_transition_minimum_frames"],
            } if weaker_weight or stronger_reception else {}),
            **({"owner_flight_weight": baseline["tag_weight"],
                "owner_flight_require_same_team_reception": True}
               if stronger_reception else {}),
        }
        previous = load_frozen_diagnostic(
            project / variant["source_owner_report"], variant["source_owner_report_sha256"],
            previous_values, parity["baseline_sha256"], parity["protocol_sha256"],
        )
        expected_candidate = (
            "same_team_reception_owner_flight" if stronger_reception else
            "weaker_sustained_owner_flight_continuity" if same_team_reception else
            "sustained_owner_flight_continuity" if weaker_weight else
            "gap_safe_owner_flight_continuity" if decoded_confirmation or sustained_player
            else "observed_owner_flight_continuity"
        )
        if previous["candidate"] != expected_candidate:
            raise ValueError("gap-safe revision requires the original owner-flight comparator")
        report = previous
        owner_values["owner_flight_reset_on_gap"] = True
        if decoded_confirmation:
            owner_values["owner_flight_confirm_decoded_owner"] = True
        if sustained_player:
            owner_values["owner_flight_require_sustained_player"] = True
            owner_values["owner_flight_contact_minimum_frames"] = source_values[
                "semi_markov_strong_transition_minimum_frames"
            ]
        if weaker_weight:
            owner_values["owner_flight_weight"] = baseline["tag_weight"]
        if same_team_reception:
            owner_values["owner_flight_require_same_team_reception"] = True
    return owner_values, report, {
        "protocol_sha256": _file_sha256(protocol_path),
        **({"variant_protocol_sha256": _file_sha256(variant_protocol)}
           if variant_protocol is not None else {}),
        "source_soft_contact_sha256": extension["source_soft_contact_sha256"],
        "threshold_source_sha256": selection["diagnosis_sha256"],
        "horizon_frames": owner_values["owner_flight_horizon_frames"],
        "weight": owner_values["owner_flight_weight"],
        "minimum_speed_mps": owner_values["owner_flight_minimum_ball_speed_mps"],
        "touch_motion_minimum": selection["threshold"],
    }


def select_player_reliability(project, protocol_path, baseline, parity, source_values,
                              variant_protocol=None):
    extension = json.loads(Path(protocol_path).read_text())
    if extension["minimum_age_source"] != "semi_markov_strong_transition_minimum_frames":
        raise ValueError("player reliability must use the existing strong-control support minimum")
    soft_path = project / extension["touch_control_protocol"]
    values, _, selection = select_touch_control(
        project, soft_path, baseline, parity, source_values, soften_touch=True,
    )
    report = load_frozen_diagnostic(
        project / extension["source_soft_contact_report"],
        extension["source_soft_contact_sha256"], values,
        parity["baseline_sha256"], parity["protocol_sha256"],
    )
    if report["candidate"] != "observed_touch_motion_softening":
        raise ValueError("player reliability requires the original frozen soft-contact comparison")
    minimum = baseline["semi_markov_strong_transition_minimum_frames"]
    values = {**values, "player_reliability_enabled": True,
              "player_reliability_minimum_age_frames": minimum}
    selection = {
        "protocol_sha256": _file_sha256(protocol_path),
        "source_soft_contact_sha256": extension["source_soft_contact_sha256"],
        "minimum_age_frames": minimum,
        "contact_threshold_provenance": selection["diagnosis_sha256"],
    }
    if variant_protocol is not None:
        variant = json.loads(Path(variant_protocol).read_text())
        stress_path = project / variant["source_player_stress_protocol"]
        stress_protocol = json.loads(stress_path.read_text())
        if (variant["base_player_reliability_protocol"]
                != str(Path(protocol_path).resolve().relative_to(project))
                or variant["base_player_reliability_protocol_sha256"] != _file_sha256(protocol_path)
                or variant["source_player_stress_protocol_sha256"] != _file_sha256(stress_path)
                or variant["source_age_report_sha256"] != _file_sha256(
                    project / variant["source_age_report"]
                )
                or variant["maximum_player_resume_gap_seconds"] != stress_protocol["window_seconds"]
                or not stress_protocol["policy"]["player_perturbations_synthetic"]):
            raise ValueError("player-age resume must use the frozen short-gap corruption protocol")
        previous = load_frozen_diagnostic(
            project / variant["source_age_report"], variant["source_age_report_sha256"],
            values, parity["baseline_sha256"], parity["protocol_sha256"],
        )
        if previous["candidate"] != "age_confident_near_ball_control":
            raise ValueError("player-age resume requires the prior unchanged age-confidence study")
        report = previous
        gap = int(round(variant["maximum_player_resume_gap_seconds"] * GAMES[GAME]["fps"]))
        if gap < 1:
            raise ValueError("player-age resume must allow at least one frame")
        values["player_reliability_resume_gap_frames"] = gap
        selection.update({
            "variant_protocol_sha256": _file_sha256(variant_protocol),
            "frozen_age_report_sha256": variant["source_age_report_sha256"],
            "maximum_resume_gap_frames": gap,
        })
    return values, report, selection


def load_frozen_diagnostic(path, expected_sha256, values, baseline_sha256, protocol_sha256):
    if not expected_sha256 or _file_sha256(path) != expected_sha256:
        raise ValueError("diagnostic transfer requires the unchanged frozen candidate report")
    frozen = json.loads(Path(path).read_text())
    if (
        frozen["values"] != values or frozen["baseline_sha256"] != baseline_sha256
        or frozen["protocol_sha256"] != protocol_sha256
        or frozen["runtime_source_fps"] != GAMES[GAME]["fps"]
        or not any(item["game"] == "Sample_Game_1" for item in frozen["results"])
    ):
        raise ValueError("diagnostic transfer candidate or deployment policy differs from its frozen source")
    return frozen


def _scenario_identity(prepared):
    return [
        {key: case[key] for key in (
            "name", "scenario", "tag_mode", "mask_start_rows", "mask_length_frames", "is_live_stress",
        )} for case in prepared["cases"]
    ]


def check_temporal_parity(root, protocol_path, output, output_config=None):
    project = Path(__file__).resolve().parents[1]
    extension = json.loads(Path(protocol_path).read_text())
    validation = load_validation_protocol(project / extension["validation_protocol"])
    scenario_path = project / extension["scenario_protocol"]
    if _file_sha256(scenario_path) != extension["scenario_protocol_sha256"]:
        raise ValueError("temporal-parity scenario protocol changed")
    scenario = json.loads(scenario_path.read_text())
    if extension["deployment_source_fps"] != GAMES[GAME]["fps"]:
        raise ValueError("temporal-parity baseline must use the deployed challenge fps")
    if extension["games"] != {
        "development": ["Sample_Game_1"], "previously_inspected_diagnostic": ["Sample_Game_2"],
    }:
        raise ValueError("temporal-parity study is limited to the declared development/diagnostic games")
    baseline_path = project / extension["baseline_config"]
    if _file_sha256(baseline_path) != extension["baseline_sha256"]:
        raise ValueError("temporal-parity selected config changed")
    baseline = load_config(baseline_path)
    if extension["historical_source_fps"] != baseline["compliance_provenance"]["target_fps"]:
        raise ValueError("historical interpretation must match its original calibration fps")
    if extension["minimum_objective_improvement"] != scenario["minimum_objective_improvement"]:
        raise ValueError("temporal-parity study must preserve the original improvement threshold")
    corrected = gap_time_correction(
        baseline, extension["fixed_gap_correction"], extension["deployment_source_fps"]
    )
    root = Path(root).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != validation["metrica"]["repository_commit"]:
        raise ValueError("external dataset revision differs from the frozen protocol")
    historical_path = project / extension["historical_game1_report"]
    historical = None
    if historical_path.exists():
        if _file_sha256(historical_path) != extension["historical_game1_report_sha256"]:
            raise ValueError("historical Game 1 evidence changed")
        historical = json.loads(historical_path.read_text())
    elif extension["require_historical_replay"]:
        raise ValueError("required historical Game 1 evidence is missing")
    else:
        print("Historical replay hashes unavailable; that verification will be explicitly unverified.", flush=True)
    results = []
    for role in ("development", "previously_inspected_diagnostic"):
        games = extension["games"][role]
        for game in games:
            deployments = []
            model_results = {}
            inputs = None
            for name, config, source_fps in (
                ("deployment", baseline, extension["deployment_source_fps"]),
                ("historical_calibration", baseline, extension["historical_source_fps"]),
                ("gap_time_corrected", corrected, extension["deployment_source_fps"]),
            ):
                settings = {**scenario, "development_game": game, "source_fps": source_fps}
                prepared = _prepare(root, settings, config)
                data = prepared["data"]
                current_inputs = {path.name: _file_sha256(path) for path in data["input_paths"]}
                if inputs is not None and inputs != current_inputs:
                    raise ValueError("temporal-policy inputs changed within the comparison")
                inputs = current_inputs
                deployments.append(_scenario_identity(prepared))
                if deployments[-1] != deployments[0]:
                    raise ValueError("temporal policies must share exact missingness scenarios")
                events = _scoring_events(
                    data, _load_events(data["event_path"], set(validation["metrica"]["event_control_types"]))
                )
                evaluation = _evaluate(prepared, config, settings, events)
                replay = None
                if name == "historical_calibration" and game == "Sample_Game_1" and historical is not None:
                    old = next(item["evaluation"] for item in historical["experiments"] if item["name"] == "baseline")
                    if current_inputs != historical["input_sha256"]:
                        raise ValueError("historical replay inputs differ")
                    if any(value["label_sha256"] != old["cases"][case]["label_sha256"]
                           for case, value in evaluation["cases"].items()):
                        raise ValueError("historical timing outputs no longer reproduce")
                    replay = True
                checks = guarded_checks(
                    evaluation, model_results["deployment"]["evaluation"] if model_results else evaluation, settings
                )
                effective = config_at_fps(config, source_fps, data["fps"])
                model_results[name] = {
                    "source_fps": source_fps, "timing": temporal_settings(effective, data["fps"]),
                    "evaluation": evaluation, "checks": checks,
                    "safe": all(check["passed"] for check in checks),
                    "historical_game1_hashes_verified": replay,
                }
                print("{} {}: objective={:.5f}, failures={}".format(
                    game, name, evaluation["objective"], sum(not check["passed"] for check in checks)
                ), flush=True)
                if name == "deployment":
                    official = {}
                    labels_by_scenario = {}
                    for case in prepared["cases"]:
                        if case["scenario"] not in labels_by_scenario:
                            ball = data["ball"].copy()
                            for start in case["mask_start_rows"]:
                                ball[start:start + case["mask_length_frames"]] = np.nan
                            labels_by_scenario[case["scenario"]] = _candidate_labels(
                                {**data, "ball": ball}, effective,
                                extension["official_baseline_half_window_seconds"],
                            )
                        official[case["name"]] = evaluate_case(
                            labels_by_scenario[case["scenario"]], prepared, case, settings, events
                        )
                    model_results["organizer_smooth50_2s"] = {
                        "evaluation": summarize_evaluation(official, settings),
                        "tags_used": False,
                    }
            results.append({"game": game, "role": role, "input_sha256": inputs,
                            "scenarios": deployments[0], "models": model_results})
    gains = [
        item["models"]["gap_time_corrected"]["evaluation"]["objective"]
        - item["models"]["deployment"]["evaluation"]["objective"] for item in results
    ]
    safe = all(item["models"]["gap_time_corrected"]["safe"] for item in results)
    improved = safe and gains[0] >= extension["minimum_objective_improvement"] and all(gain >= 0 for gain in gains)
    report = {
        "schema_version": 1, "challenge_runtime_fps": GAMES[GAME]["fps"],
        "baseline_sha256": extension["baseline_sha256"],
        "protocol_sha256": _file_sha256(protocol_path),
        "scenario_protocol_sha256": extension["scenario_protocol_sha256"],
        "validation_protocol_sha256": _file_sha256(project / extension["validation_protocol"]),
        "external_revision": revision, "results": results,
        "fixed_correction": {
            "key": extension["fixed_gap_correction"]["config_key"],
            "before": baseline["max_interpolation_gap_frames"],
            "after": corrected["max_interpolation_gap_frames"],
        },
        "gap_correction_safe": safe, "objective_gains": gains,
        "candidate_ready_for_frozen_release": bool(improved),
        "selected_submission_changed": False, "historical_protocols_changed": False,
        "guards": scenario["guards"],
        "historical_replay_evidence_available": historical is not None,
        "code_sha256": {path.name: _file_sha256(path) for path in (
            Path(__file__), project / "possession" / "external_metrica_validation.py",
            project / "possession" / "external_guarded_calibration.py",
            project / "possession" / "decoder.py",
        )},
        "limitations": [
            "Deployment parity fixes evaluation interpretation; it does not certify inherited parameter provenance.",
            "The fixed candidate changes only interpolation time, not all inherited windows or switch costs.",
            "Games 1-2 are development/reused diagnostic evidence, not fresh holdouts.",
            "Provider references and synthetic tagging are not challenge possession truth.",
            "Historical reports remain historical; no old score or frozen hash is rewritten.",
            "No challenge inference or automatic promotion occurs in this study.",
        ],
    }
    _save(output, report)
    if improved and output_config is not None:
        corrected["temporal_policy_provenance"] = {
            "runtime_fps": GAMES[GAME]["fps"],
            "baseline_sha256": extension["baseline_sha256"],
            "report": str(Path(output).resolve()), "protocol_sha256": report["protocol_sha256"],
            "only_changed_parameter": "max_interpolation_gap_frames",
            "full_active_provenance_cleared": False, "release_checks_completed": False,
        }
        _save(output_config, corrected)
    print("Gap-time correction ready for frozen release: {}".format(improved), flush=True)
    return report


def check_rebased_core(root, protocol_path, parity_path, output, output_config=None,
                       diagnosis_path=None, rebase_path=None, preserve_support_fraction=False,
                       restore_observed_strong_cost=False, touch_control_path=None,
                       soften_touch=False, continuous_touch=False,
                       diagnostic_report_path=None, diagnostic_report_sha256=None,
                       owner_flight_protocol=None, owner_variant_protocol=None,
                       player_reliability_protocol=None, player_reliability_variant=None):
    if player_reliability_variant is not None and player_reliability_protocol is None:
        raise ValueError("player-age resume requires the original reliability protocol")
    if owner_variant_protocol is not None and owner_flight_protocol is None:
        raise ValueError("owner-flight revision requires the original owner-flight protocol")
    if player_reliability_protocol is not None and (
        owner_flight_protocol is not None or touch_control_path is not None or diagnosis_path is not None
    ):
        raise ValueError("player reliability must be tested separately from other interventions")
    if soften_touch and continuous_touch:
        raise ValueError("touch-confidence alternatives must be tested separately")
    if (soften_touch or continuous_touch) and touch_control_path is None:
        raise ValueError("touch softening requires the fixed touch-control protocol")
    if touch_control_path is not None and (
        diagnosis_path is not None or preserve_support_fraction or restore_observed_strong_cost
    ):
        raise ValueError("touch-control pilot must be evaluated separately from support refinements")
    if owner_flight_protocol is not None and (
        touch_control_path is not None or diagnosis_path is not None
        or preserve_support_fraction or restore_observed_strong_cost
        or soften_touch or continuous_touch
    ):
        raise ValueError("owner-flight study must be evaluated separately from other interventions")
    if preserve_support_fraction and restore_observed_strong_cost:
        raise ValueError("residual repairs must be evaluated separately")
    if (preserve_support_fraction or restore_observed_strong_cost) and diagnosis_path is None:
        raise ValueError("residual repairs require diagnostic evidence")
    project = Path(__file__).resolve().parents[1]
    extension = json.loads(Path(protocol_path).read_text())
    validation = load_validation_protocol(project / extension["validation_protocol"])
    parity = json.loads(Path(parity_path).read_text())
    if parity["protocol_sha256"] != _file_sha256(protocol_path):
        raise ValueError("rebase requires the matching temporal-parity report")
    baseline_path = project / extension["baseline_config"]
    if _file_sha256(baseline_path) != parity["baseline_sha256"]:
        raise ValueError("selected deployment config changed before rebase")
    scenario_path = project / extension["scenario_protocol"]
    if _file_sha256(scenario_path) != extension["scenario_protocol_sha256"]:
        raise ValueError("rebase scenario protocol changed")
    scenario = json.loads(scenario_path.read_text())
    baseline = load_config(baseline_path)
    if extension["games"] != {
        "development": ["Sample_Game_1"], "previously_inspected_diagnostic": ["Sample_Game_2"],
    }:
        raise ValueError("rebase is limited to Game 1 development followed by Game 2 diagnostic")
    source_path = project / extension["historical_game1_report"]
    if _file_sha256(source_path) != extension["historical_game1_report_sha256"]:
        raise ValueError("frozen core source evidence changed")
    source_report = json.loads(source_path.read_text())
    source = next(item for item in source_report["experiments"] if item["name"] == "near_miss")
    if set(source["values"]) != set(CORE_CALIBRATION_KEYS):
        raise ValueError("frozen rebase must change only the declared core evidence/support settings")
    values = source["values"]
    diagnosed_rebase = None
    touch_selection = None
    if diagnosis_path is not None:
        if rebase_path is None:
            raise ValueError("support repair requires its original rebase report")
        diagnosis = json.loads(Path(diagnosis_path).read_text())
        if (
            diagnosis["protocol_sha256"] != _file_sha256(protocol_path)
            or diagnosis["baseline_sha256"] != parity["baseline_sha256"]
            or diagnosis["runtime_fps"] != GAMES[GAME]["fps"]
            or not diagnosis["original_model_outputs_reproduced"]
            or diagnosis["rebase_report_sha256"] != _file_sha256(rebase_path)
        ):
            raise ValueError("support repair diagnosis does not match the frozen runtime rebase")
        diagnosed_rebase = json.loads(Path(rebase_path).read_text())
        comparator_values = (
            support_horizon_repair_values(baseline, source["values"])
            if preserve_support_fraction or restore_observed_strong_cost else source["values"]
        )
        if diagnosed_rebase["values"] != comparator_values:
            raise ValueError("support repair source values differ from the diagnosed candidate")
        if preserve_support_fraction or restore_observed_strong_cost:
            primary = next(case for case in diagnosis["cases"] if case["case"] == "recovery_masked:none")
            candidate_result = diagnosed_rebase["results"][0]["candidate"]
            failures = [check for check in candidate_result["checks"] if not check["passed"]]
            if len(failures) != 1 or failures[0]["name"] != "recovery_masked:none:spell_f1_0.75":
                raise ValueError("residual repair requires the diagnosed high-IoU failure")
            score = candidate_result["evaluation"]["cases"]["recovery_masked:none"][
                "possession"]["reference_conditioned_spell_scores"]["0.75"]
            denominator = 2 * score["matched"] + score["missed"] + score["spurious_in_reference_regions"]
            required_removals = math.ceil(denominator - 2 * score["matched"] / failures[0]["minimum"])
            targeted_spurious = primary["high_iou_spurious_lineage"]["introduced_summary"][
                "observed_strong_support" if restore_observed_strong_cost
                else "extension_only_below_baseline_fraction"]
            if required_removals < 1 or targeted_spurious < required_removals:
                raise ValueError("residual diagnosis does not identify enough targeted spurious intervals")
        else:
            primary = next(case for case in diagnosis["cases"]
                           if case["case"] == "recovery_masked:degraded_restart_tags")
            failures = primary["recoveries"]["groups"]["regression"]
            if (
                failures["events"] == 0
                or failures["support_cost_inversion"] < .5 * failures["events"]
                or failures["long_window_would_enable_support"] < .5 * failures["events"]
            ):
                raise ValueError("diagnosis does not concentrate on support-horizon loss")
        values = support_horizon_repair_values(
            baseline, source["values"], preserve_support_fraction, restore_observed_strong_cost
        )
    if touch_control_path is not None:
        values, diagnosed_rebase, touch_selection = select_touch_control(
            project, touch_control_path, baseline, parity, source["values"], soften_touch, continuous_touch
        )
        comparator_values = diagnosed_rebase["values"]
    owner_selection = None
    if owner_flight_protocol is not None:
        values, diagnosed_rebase, owner_selection = select_owner_flight(
            project, owner_flight_protocol, baseline, parity, source["values"], owner_variant_protocol,
        )
        comparator_values = diagnosed_rebase["values"]
    player_selection = None
    if player_reliability_protocol is not None:
        values, diagnosed_rebase, player_selection = select_player_reliability(
            project, player_reliability_protocol, baseline, parity, source["values"],
            player_reliability_variant,
        )
        comparator_values = diagnosed_rebase["values"]
    candidate = {**baseline, **values}
    candidate_name = (
        "short_gap_resumed_player_confidence" if player_reliability_variant is not None else
        "age_confident_near_ball_control" if player_reliability_protocol is not None else
        "stronger_confirmed_owner_flight" if owner_variant_protocol is not None
        and values.get("owner_flight_require_same_team_reception", False)
        and values["owner_flight_weight"] == baseline["restart_weight"] else
        "same_team_reception_owner_flight" if owner_variant_protocol is not None
        and values.get("owner_flight_require_same_team_reception", False) else
        "weaker_sustained_owner_flight_continuity" if owner_variant_protocol is not None
        and values.get("owner_flight_require_sustained_player", False)
        and values["owner_flight_weight"] == baseline["tag_weight"] else
        "sustained_owner_flight_continuity" if owner_variant_protocol is not None
        and values.get("owner_flight_require_sustained_player", False) else
        "confirmed_owner_flight_continuity" if owner_variant_protocol is not None
        and values.get("owner_flight_confirm_decoded_owner", False) else
        "gap_safe_owner_flight_continuity" if owner_variant_protocol is not None else
        "observed_owner_flight_continuity" if owner_flight_protocol is not None else
        "observed_touch_motion_confidence" if continuous_touch else
        "observed_touch_motion_softening" if soften_touch else
        "observed_touch_motion_gate" if touch_control_path is not None else
        "observed_strong_cost_restore" if restore_observed_strong_cost else
        "fraction_preserving_support_repair" if preserve_support_fraction else
        "uncertain_support_horizon_repair" if diagnosis_path is not None else "frozen_core_runtime_rebase"
    )
    frozen_diagnostic = None
    if diagnostic_report_path is not None:
        frozen_diagnostic = load_frozen_diagnostic(
            diagnostic_report_path, diagnostic_report_sha256, values,
            parity["baseline_sha256"], _file_sha256(protocol_path),
        )
    elif diagnostic_report_sha256 is not None:
        raise ValueError("a diagnostic candidate hash requires its report")
    if extension["deployment_source_fps"] != GAMES[GAME]["fps"]:
        raise ValueError("rebased core must use actual runtime fps")
    revision = subprocess.check_output(
        ["git", "-C", str(Path(root).resolve()), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != validation["metrica"]["repository_commit"]:
        raise ValueError("rebase external dataset revision differs")
    results = []
    for role in ("development", "previously_inspected_diagnostic"):
        games = extension["games"][role]
        for game in games:
            if role != "development" and not results[0]["candidate"]["safe"] and frozen_diagnostic is None:
                print("{} not evaluated: fixed candidate failed development guards.".format(game), flush=True)
                continue
            if role != "development" and not results[0]["candidate"]["safe"]:
                print("{}: authorized frozen diagnostic despite development failure; promotion guards remain unchanged.".format(game), flush=True)
            settings = {**scenario, "development_game": game,
                        "source_fps": extension["deployment_source_fps"]}
            if frozen_diagnostic is not None:
                settings["diagnostic_spell_ious"] = [.3, .4, .5, .6, .7]
            prepare_config = {**baseline, **({
                "player_reliability_resume_gap_frames": values["player_reliability_resume_gap_frames"]
            } if player_reliability_variant is not None else {})}
            prepared = _prepare(Path(root).resolve(), settings, prepare_config)
            original = next(item for item in parity["results"] if item["game"] == game)
            if _scenario_identity(prepared) != original["scenarios"]:
                raise ValueError("rebase missingness scenarios differ from deployment benchmark")
            inputs = {path.name: _file_sha256(path) for path in prepared["data"]["input_paths"]}
            if inputs != original["input_sha256"]:
                raise ValueError("rebase inputs differ from deployment benchmark")
            events = _scoring_events(prepared["data"], _load_events(
                prepared["data"]["event_path"], set(validation["metrica"]["event_control_types"])
            ))
            deployment = _evaluate(prepared, baseline, settings, events)
            expected = original["models"]["deployment"]["evaluation"]
            if any(item["label_sha256"] != expected["cases"][case]["label_sha256"]
                   for case, item in deployment["cases"].items()):
                raise ValueError("rebased study failed to reproduce deployment baseline")
            unrepaired = None
            if diagnosed_rebase is not None and game == "Sample_Game_1":
                unrepaired = _evaluate(prepared, {**baseline, **comparator_values}, settings, events)
                original_candidate = diagnosed_rebase["results"][0]["candidate"]["evaluation"]
                if any(item["label_sha256"] != original_candidate["cases"][case]["label_sha256"]
                       for case, item in unrepaired["cases"].items()):
                    raise ValueError("support-repair comparator no longer reproduces")
            evaluation = _evaluate(prepared, candidate, settings, events)
            if game == "Sample_Game_1" and frozen_diagnostic is not None:
                original = next(item for item in frozen_diagnostic["results"] if item["game"] == game)
                if inputs != original["input_sha256"] or any(
                    item["label_sha256"] != original["candidate"]["evaluation"]["cases"][case]["label_sha256"]
                    or item["possession"] != original["candidate"]["evaluation"]["cases"][case]["possession"]
                    for case, item in evaluation["cases"].items()
                ):
                    raise ValueError("frozen diagnostic no longer reproduces Game1 candidate outputs")
            checks = guarded_checks(evaluation, deployment, settings)
            safe = all(check["passed"] for check in checks)
            results.append({
                "game": game, "role": role, "input_sha256": inputs,
                "deployment": deployment,
                "candidate": {"evaluation": evaluation, "checks": checks, "safe": safe},
                **({"unrepaired_candidate": unrepaired} if unrepaired is not None else {}),
            })
            print("{} {}: objective={:.5f}, failures={}".format(
                game, candidate_name,
                evaluation["objective"], sum(not check["passed"] for check in checks)
            ), flush=True)
    all_games = [game for games in extension["games"].values() for game in games]
    gains = [item["candidate"]["evaluation"]["objective"] - item["deployment"]["objective"]
             for item in results]
    ready = (
        len(results) == len(all_games) and all(item["candidate"]["safe"] for item in results)
        and gains[0] >= scenario["minimum_objective_improvement"] and all(gain >= 0 for gain in gains)
    )
    report = {
        "schema_version": 1,
        "candidate": candidate_name,
        "protocol_sha256": _file_sha256(protocol_path),
        "parity_report_sha256": _file_sha256(parity_path),
        "source_report_sha256": extension["historical_game1_report_sha256"],
        "baseline_sha256": parity["baseline_sha256"], "runtime_source_fps": GAMES[GAME]["fps"],
        "values": values, "results": results, "objective_gains": gains,
        "diagnosis_sha256": _file_sha256(diagnosis_path) if diagnosis_path is not None else None,
        **({"touch_control_selection": touch_selection} if touch_selection is not None else {}),
        **({"owner_flight_selection": owner_selection} if owner_selection is not None else {}),
        **({"player_reliability_selection": player_selection} if player_selection is not None else {}),
        "evaluated_games": [item["game"] for item in results],
        "not_evaluated_games": [game for game in all_games if game not in {item["game"] for item in results}],
        "candidate_ready_for_frozen_release": bool(ready),
        "failed_development_diagnostic_allowed": frozen_diagnostic is not None,
        "diagnostic_candidate_sha256": diagnostic_report_sha256,
        "development_guard_passed": results[0]["candidate"]["safe"],
        "selected_submission_changed": False, "parameters_retuned": False,
        "guards": scenario["guards"],
        "code_sha256": {path.name: _file_sha256(path) for path in (
            Path(__file__), project / "possession" / "decoder.py",
            project / "possession" / "external_guarded_calibration.py",
            project / "possession" / "external_metrica_validation.py",
            project / "possession" / "tracking.py",
        )},
        "limitations": [
            "One fixed previously defined core vector; no new search or acceptance-threshold relaxation.",
            "Support repair extends the non-observed-frame horizon. Residual variants separately preserve contact fraction or restore the observed strong cost; uncertain-frame costs remain fixed.",
            "Development rejection blocks promotion; diagnostic transfer after failure requires an explicit frozen-report authorization.",
            "Reference-conditioned scores are not challenge scores.",
            "Inherited origins remain unresolved; no automatic promotion.",
            "Touch motion is an observed-tracking cue, not proof of physical contact; unavailable motion explicitly preserves original support.",
            "Authorized frozen diagnostics also report the official IoU-range mean on provider references; it does not replace any guard or become challenge truth.",
        ],
    }
    _save(output, report)
    if ready and output_config is not None:
        candidate["temporal_policy_provenance"] = {
            "runtime_fps": GAMES[GAME]["fps"],
            "baseline_sha256": parity["baseline_sha256"], "report": str(Path(output).resolve()),
            "source_report_sha256": extension["historical_game1_report_sha256"],
            "full_active_provenance_cleared": False, "release_checks_completed": False,
        }
        if touch_selection is not None:
            candidate["touch_control_provenance"] = {
                **touch_selection, "development_game": "Sample_Game_1",
                "runtime_fps": GAMES[GAME]["fps"], "report": str(Path(output).resolve()),
            }
        if owner_selection is not None:
            candidate["owner_flight_provenance"] = {
                **owner_selection, "development_game": "Sample_Game_1",
                "runtime_fps": GAMES[GAME]["fps"], "report": str(Path(output).resolve()),
            }
        if player_selection is not None:
            candidate["player_reliability_provenance"] = {
                **player_selection, "development_game": "Sample_Game_1",
                "report": str(Path(output).resolve()), "release_checks_completed": False,
            }
        _save(output_config, candidate)
    print("{} ready for release: {}".format(candidate_name, ready), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare deployed temporal policy with historical scaling and one fixed gap correction.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", default=str(
        Path(__file__).resolve().parents[1] / "configs" / "metrica_temporal_parity.json"
    ))
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-config")
    parser.add_argument("--rebase-core", action="store_true")
    parser.add_argument("--parity-report")
    parser.add_argument("--support-horizon-repair", action="store_true")
    parser.add_argument("--diagnosis")
    parser.add_argument("--rebase-report")
    parser.add_argument("--preserve-support-fraction", action="store_true")
    parser.add_argument("--restore-observed-strong-cost", action="store_true")
    parser.add_argument("--touch-control-protocol")
    parser.add_argument("--owner-flight-protocol")
    parser.add_argument("--owner-variant-protocol")
    parser.add_argument("--player-reliability-protocol")
    parser.add_argument("--player-reliability-variant")
    parser.add_argument("--soften-touch-discount", action="store_true")
    parser.add_argument("--continuous-touch-confidence", action="store_true")
    parser.add_argument("--diagnostic-candidate-report")
    parser.add_argument("--diagnostic-candidate-sha256")
    args = parser.parse_args(argv)
    try:
        if (args.diagnostic_candidate_report or args.diagnostic_candidate_sha256) and not (
            args.rebase_core or args.support_horizon_repair or args.touch_control_protocol
            or args.owner_flight_protocol or args.player_reliability_protocol
        ):
            raise ValueError("frozen diagnostic transfer requires an explicit candidate definition")
        if (args.soften_touch_discount or args.continuous_touch_confidence) and not args.touch_control_protocol:
            raise ValueError("touch-confidence alternatives require --touch-control-protocol")
        if args.preserve_support_fraction and not args.support_horizon_repair:
            raise ValueError("--preserve-support-fraction requires --support-horizon-repair")
        if args.restore_observed_strong_cost and not args.support_horizon_repair:
            raise ValueError("--restore-observed-strong-cost requires --support-horizon-repair")
        if (args.rebase_core or args.support_horizon_repair or args.touch_control_protocol
                or args.owner_flight_protocol or args.player_reliability_protocol):
            if not args.parity_report:
                raise ValueError("--rebase-core requires --parity-report")
            if args.support_horizon_repair and (not args.diagnosis or not args.rebase_report):
                raise ValueError("--support-horizon-repair requires --diagnosis and --rebase-report")
            check_rebased_core(
                args.root, args.protocol, args.parity_report, args.output, args.output_config,
                args.diagnosis if args.support_horizon_repair else None,
                args.rebase_report if args.support_horizon_repair else None,
                args.preserve_support_fraction,
                args.restore_observed_strong_cost,
                args.touch_control_protocol,
                args.soften_touch_discount,
                args.continuous_touch_confidence,
                args.diagnostic_candidate_report,
                args.diagnostic_candidate_sha256,
                args.owner_flight_protocol,
                args.owner_variant_protocol,
                args.player_reliability_protocol,
                args.player_reliability_variant,
            )
        else:
            check_temporal_parity(args.root, args.protocol, args.output, args.output_config)
    except (OSError, ValueError, KeyError, TypeError, StopIteration, subprocess.CalledProcessError) as error:
        parser.exit(1, "error: {}\n".format(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
