import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from possession import UNKNOWN
from possession.cli import load_config
from possession.decoder import build_evidence, decode, touch_motion_summary
from possession.external_metrica_validation import (
    _decoded_event_metrics, _file_sha256, _load_events, _load_game, _period_spell_diagnostics,
    _repair_by_period, config_at_fps,
)
from possession.external_possession_events import (
    possession_chain_references, recovery_masks, score_censored_dead_frames,
    score_possession_references,
    score_recovery_transitions,
)
from possession.external_transfer_check import (
    _live_stress_metrics, independent_reference_mask, live_missing_stress, restart_only_tags,
)
from possession.tagging import infer_restart_ownership, localize_restarts
from possession.tracking import control_features_from_arrays


def guarded_candidates(protocol):
    guarded = {**protocol["stoppage_profile"], **protocol["corroboration_profile"]}
    candidates = [
        {"name": "baseline", "overrides": {}, "selection_allowed": True},
        {"name": "uncorroborated_diagnostic",
         "overrides": protocol["stoppage_profile"], "selection_allowed": False},
        {"name": "corroborated_stoppage", "overrides": guarded, "selection_allowed": True},
    ]
    for mode in ("augment", "contact_guarded"):
        gap = {**protocol["gap_profile"], "semi_markov_gap_handover_mode": mode}
        candidates.extend([
            {"name": "gap_" + mode, "overrides": gap, "selection_allowed": True},
            {"name": "corroborated_gap_" + mode,
             "overrides": {**guarded, **gap}, "selection_allowed": True},
        ])
    return candidates


def load_guarded_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if (
        protocol["development_game"] != "Sample_Game_1"
        or set(protocol["excluded_games"]) != {"Sample_Game_2", "Sample_Game_3"}
    ):
        raise ValueError("guarded selection is restricted to Game 1 only")
    if not np.isclose(sum(protocol["selection_weights"].values()), 1.0):
        raise ValueError("guarded objective weights must sum to one")
    project = Path(__file__).resolve().parents[1]
    path = project / protocol["baseline_config"]
    if _file_sha256(path) != protocol["baseline_sha256"]:
        raise ValueError("baseline differs from the frozen guarded protocol")
    return protocol, load_config(path)


def _prepare(root, protocol, baseline, player_arrays=None):
    data = _load_game(root, protocol["development_game"])
    if player_arrays is not None:
        if set(player_arrays) != {"home", "away"}:
            raise ValueError("player stress requires both home and away arrays")
        for team in ("home", "away"):
            positions = player_arrays[team]
            if (not isinstance(positions, np.ndarray) or positions.shape != data[team].shape
                    or np.isinf(positions).any()):
                raise ValueError("player stress requires aligned finite-or-missing player positions")
        data = {**data, "home": player_arrays["home"], "away": player_arrays["away"]}
    events = _load_events(data["event_path"], {
        "PASS", "CARRY", "RECOVERY", "BALL LOST", "BALL OUT", "SHOT", "SET PIECE", "FAULT RECEIVED",
    })
    truth, references = possession_chain_references(
        events, data["frames"], data["periods"], data["fps"], protocol["references"]
    )
    primary_truth, primary_references = independent_reference_mask(
        truth, references, events, data["periods"], data["fps"],
        protocol["restart_reference_exclusion_seconds"],
    )
    if not primary_references["complete_live_spells"]:
        raise ValueError("no independent-endpoint development spells")
    masked, recovery_starts, recovery_length = recovery_masks(
        data["ball"], data["periods"], references["recovery_references"],
        data["fps"], protocol["recovery_scoring"]["mask_seconds"],
    )
    if not recovery_starts:
        raise ValueError("no eligible recovery-centered development masks")
    scenarios = [("clean", data["ball"], [], 0, False),
                 ("recovery_masked", masked, recovery_starts, recovery_length, False)]
    for seconds in protocol["live_missing_durations_seconds"]:
        ball, starts, length = live_missing_stress(
            data["ball"], primary_truth, data["periods"], data["fps"], seconds,
            protocol["maximum_mask_windows"], protocol["minimum_mask_windows"],
        )
        scenarios.append(("live_missing_{}s".format(seconds), ball, starts, length, True))
    config = config_at_fps(baseline, protocol["source_fps"], data["fps"])
    edges = np.r_[0, np.flatnonzero(np.diff(data["periods"])) + 1, len(truth)]
    cases = []
    for scenario, ball, starts, length, live_stress in scenarios:
        prepared = []
        for start, stop in zip(edges[:-1], edges[1:]):
            trajectory = _repair_by_period(ball[start:stop], data["fps"], config)
            control = control_features_from_arrays(
                data["home"][start:stop], data["away"][start:stop],
                trajectory.xy, data["fps"], config,
            )
            period_events = [
                {**event, "frame": event["frame"] - int(start)}
                for event in events if event["period"] == data["periods"][start]
            ]
            prepared.append((start, stop, trajectory, control, period_events))
        for tag_mode, settings in protocol["tag_modes"].items():
            periods = []
            for start, stop, trajectory, control, period_events in prepared:
                tags = restart_only_tags(
                    period_events, data["periods"][start:stop], data["fps"], config,
                    settings, protocol["seed"] + int(data["periods"][start]),
                )
                tags = localize_restarts(tags, trajectory, data["fps"], config)
                tags = infer_restart_ownership(tags, trajectory, control, config)
                periods.append((trajectory, control, tags))
            cases.append({
                "name": scenario + ":" + tag_mode, "scenario": scenario,
                "tag_mode": tag_mode, "periods": periods,
                "mask_start_rows": starts, "mask_length_frames": length,
                "is_live_stress": live_stress,
            })
    return {
        "data": data, "references": references, "truth": primary_truth,
        "full_truth": truth,
        "primary_references": primary_references, "cases": cases,
    }


def evaluate_case(labels, prepared, case, protocol, events=None):
    data = prepared["data"]
    diagnostic_ious = protocol.get("diagnostic_spell_ious", [])
    if any(not np.isfinite(iou) or not 0 < iou <= 1 for iou in diagnostic_ious):
        raise ValueError("diagnostic spell IoUs must be finite and within (0, 1]")
    scoring_ious = list(dict.fromkeys(list(protocol["spell_ious"]) + list(diagnostic_ious)))
    possession = score_possession_references(
        labels, prepared["truth"], prepared["primary_references"], data["periods"], scoring_ious
    )
    diagnostic_scores = possession["reference_conditioned_spell_scores"]
    if diagnostic_ious:
        possession = {
            **possession,
            "reference_conditioned_spell_scores": {
                str(iou): diagnostic_scores[str(iou)] for iou in protocol["spell_ious"]
            },
        }
    result = {
        "possession": possession,
        "recoveries": score_recovery_transitions(
            labels, data["periods"], prepared["references"]["recovery_references"],
            data["fps"], protocol["recovery_scoring"],
        ),
        "live_stress": (
            _live_stress_metrics(labels, case["mask_start_rows"], case["mask_length_frames"])
            if case["is_live_stress"] else None
        ),
        "spells": _period_spell_diagnostics(labels, data["fps"], data["periods"]),
        "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
    }
    if "full_truth" in prepared:
        result["full_timeline_diagnostic"] = {
            "dead": score_censored_dead_frames(labels, prepared["full_truth"]),
            "possession": score_possession_references(
                labels, prepared["full_truth"], prepared["references"],
                data["periods"], (0.3, 0.4, 0.5, 0.6, 0.7),
            ),
            "is_challenge_score": False,
            "provider_events_are_reference_only": True,
        }
    if events is not None:
        result["event_anchors"] = _decoded_event_metrics(labels, events)[0]
    if diagnostic_ious:
        result["official_style_reference_spell"] = {
            "thresholds": diagnostic_ious,
            "mean_f1": float(np.mean([diagnostic_scores[str(iou)]["f1"] for iou in diagnostic_ious])),
            "scores": {str(iou): diagnostic_scores[str(iou)] for iou in diagnostic_ious},
            "is_challenge_score": False,
        }
    return result


def summarize_evaluation(results, protocol):
    weights = protocol["selection_weights"]
    objective = float(np.mean([
        weights["spell_f1"] * np.mean([
            result["possession"]["reference_conditioned_spell_scores"][str(iou)]["f1"]
            for iou in protocol["spell_ious"]
        ])
        + weights["boundary_recall"] * result["recoveries"]["same_direction_boundary_recall"]["0.5"]
        + weights["recovery_accuracy"] * result["recoveries"]["accuracy_at_recovery"]
        + weights["live_macro_f1"] * result["possession"]["known_live_macro_f1"]
        for result in results.values()
    ]))
    return {"objective": objective, "cases": results}


def _evaluate(prepared, source_config, protocol, events=None, include_labels=False):
    data = prepared["data"]
    config = config_at_fps(source_config, protocol["source_fps"], data["fps"])
    results = {}
    labels_by_case = {}
    for case in prepared["cases"]:
        period_labels, quality, owner_quality, player_quality, phase_quality, exit_quality = [], {}, {}, {}, {}, {}
        for trajectory, control, tags in case["periods"]:
            evidence = build_evidence(control, trajectory, tags, (105.0, 68.0), config, fps=data["fps"])
            period_labels.append(decode(evidence, tags, config))
            if config.get("semi_markov_touch_motion_gate_enabled", False):
                for key, value in touch_motion_summary(evidence, config).items():
                    quality[key] = quality.get(key, 0) + value
            if config.get("semi_markov_owner_flight_enabled", False):
                for key, value in evidence.owner_flight_diagnostics.items():
                    owner_quality[key] = owner_quality.get(key, 0) + value
            if config.get("player_reliability_enabled", False):
                for key, value in evidence.player_reliability_diagnostics.items():
                    if key == "age_threshold_frames":
                        player_quality[key] = value
                    else:
                        player_quality[key] = player_quality.get(key, 0) + value
            if config.get("semi_markov_phase_enabled", False):
                for key, value in evidence.control_phase.diagnostics.items():
                    phase_quality[key] = phase_quality.get(key, 0) + value
            if config.get("tracking_exit_bracket_enabled", False):
                for key, value in evidence.exit_bracket_diagnostics.items():
                    exit_quality[key] = exit_quality.get(key, 0) + value
        labels = np.concatenate(period_labels)
        results[case["name"]] = evaluate_case(labels, prepared, case, protocol, events)
        if include_labels:
            labels_by_case[case["name"]] = labels
        if quality:
            results[case["name"]]["touch_motion"] = {
                **quality, "unknown_behavior": "retain_original_support",
                "mode": config.get("touch_motion_mode", "gate"),
            }
        if owner_quality:
            results[case["name"]]["owner_flight"] = owner_quality
        if player_quality:
            results[case["name"]]["player_reliability"] = player_quality
        if phase_quality:
            results[case["name"]]["control_phase"] = phase_quality
        if exit_quality:
            results[case["name"]]["exit_bracket"] = exit_quality
    summary = summarize_evaluation(results, protocol)
    return (summary, labels_by_case) if include_labels else summary


def guarded_checks(evaluation, baseline, protocol):
    guards = protocol["guards"]
    checks = []
    for name, result in evaluation["cases"].items():
        reference = baseline["cases"][name]
        for iou in protocol["spell_ious"]:
            actual = result["possession"]["reference_conditioned_spell_scores"][str(iou)]["f1"]
            minimum = (
                reference["possession"]["reference_conditioned_spell_scores"][str(iou)]["f1"]
                - guards["maximum_case_spell_f1_drop"]
            )
            checks.append({"name": name + ":spell_f1_" + str(iou),
                           "passed": bool(actual >= minimum), "actual": actual, "minimum": minimum})
        for metric, actual, minimum in (
            ("live_f1", result["possession"]["known_live_macro_f1"],
             reference["possession"]["known_live_macro_f1"] - guards["maximum_case_live_macro_f1_drop"]),
            ("recovery_accuracy", result["recoveries"]["accuracy_at_recovery"],
             reference["recoveries"]["accuracy_at_recovery"] - guards["maximum_case_recovery_accuracy_drop"]),
            ("boundary_recall", result["recoveries"]["same_direction_boundary_recall"]["0.5"],
             reference["recoveries"]["same_direction_boundary_recall"]["0.5"]
             - guards["maximum_case_boundary_recall_drop"]),
        ):
            checks.append({"name": name + ":" + metric, "passed": bool(actual >= minimum),
                           "actual": actual, "minimum": minimum})
        actual = result["spells"]["shorter_than_one_second"]
        maximum = reference["spells"]["shorter_than_one_second"] + guards["maximum_short_spell_increase"]
        checks.append({"name": name + ":short_spells", "passed": actual <= maximum,
                       "actual": actual, "maximum": maximum})
        if result["live_stress"] is not None:
            for metric, key in (
                ("false_dead_fraction", "maximum_excess_false_dead_fraction"),
                ("windows_with_false_dead_fraction", "maximum_excess_broken_window_fraction"),
                ("split_window_fraction", "maximum_excess_split_window_fraction"),
            ):
                actual = result["live_stress"][metric] - reference["live_stress"][metric]
                checks.append({"name": name + ":excess_" + metric,
                               "passed": actual <= guards[key], "actual": actual, "maximum": guards[key]})
    return checks


def _save(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(content, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def calibrate_guarded(root, protocol_path, output_config, output_report):
    root = Path(root).resolve()
    protocol_path = Path(protocol_path)
    protocol, baseline = load_guarded_protocol(protocol_path)
    if baseline["video_ball_observations_path"] is not None:
        raise ValueError("guarded development cannot use match-specific video")
    prepared = _prepare(root, protocol, baseline)
    experiments = []
    for candidate in guarded_candidates(protocol):
        evaluation = _evaluate(prepared, {**baseline, **candidate["overrides"]}, protocol)
        checks = guarded_checks(
            evaluation, experiments[0]["evaluation"] if experiments else evaluation, protocol
        )
        safe = all(check["passed"] for check in checks)
        experiments.append({**candidate, "evaluation": evaluation, "checks": checks, "safe": safe})
        print("{}: objective={:.5f}, safe={}, selectable={}".format(
            candidate["name"], evaluation["objective"], safe, candidate["selection_allowed"]
        ), flush=True)
    base_score = experiments[0]["evaluation"]["objective"]
    eligible = [
        candidate for candidate in experiments
        if candidate["safe"] and candidate["selection_allowed"]
        and candidate["evaluation"]["objective"] >= base_score + protocol["minimum_objective_improvement"]
    ]
    selected = max(eligible, key=lambda item: item["evaluation"]["objective"]) if eligible else experiments[0]
    improved = selected["name"] != "baseline"
    config = {**baseline, **selected["overrides"]}
    config["guarded_development_provenance"] = {
        "development_game": protocol["development_game"],
        "excluded_games": protocol["excluded_games"],
        "baseline_sha256": protocol["baseline_sha256"],
        "protocol_sha256": _file_sha256(protocol_path),
        "selected_candidate": selected["name"],
        "selected_overrides": selected["overrides"],
        "fresh_holdout_validation": "pending",
        "inherited_parameters_fully_external_audited": False,
    }
    _save(output_config, config)
    data = prepared["data"]
    project = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1, "external_data": True,
        "development_game": protocol["development_game"],
        "excluded_games": protocol["excluded_games"],
        "protocol_sha256": _file_sha256(protocol_path),
        "baseline_sha256": protocol["baseline_sha256"],
        "selected_config_sha256": _file_sha256(output_config),
        "selected_candidate": selected["name"],
        "development_improvement_found": improved,
        "status": "development_gain_awaiting_fresh_holdout" if improved else "no_safe_development_gain",
        "reference_summary": prepared["references"]["summary"],
        "primary_known_frames": int(np.count_nonzero(prepared["truth"] != UNKNOWN)),
        "primary_reference_spells": len(prepared["primary_references"]["complete_live_spells"]),
        "input_sha256": {path.name: _file_sha256(path) for path in data["input_paths"]},
        "cases": [
            {key: case[key] for key in ("name", "scenario", "tag_mode", "mask_start_rows",
                                      "mask_length_frames", "is_live_stress")}
            for case in prepared["cases"]
        ],
        "experiments": experiments,
        "code_sha256": {
            path.name: _file_sha256(path) for path in (
                Path(__file__), project / "possession" / "decoder.py",
                project / "possession" / "external_metrica_validation.py",
                project / "possession" / "external_possession_events.py",
                project / "possession" / "external_transfer_check.py",
                project / "possession" / "tracking.py", project / "possession" / "trajectory.py",
            )
        },
        "challenge_submission_modified": False,
        "limitations": [
            "Development scores only; Games 2-3 and challenge data are not used or evaluated.",
            "Provider-event references and synthetic restart inputs are not challenge ground truth.",
            "Long known-live masks are hard negatives in selection, not just post-selection tests.",
            "Both IoU thresholds and every scenario must pass non-regression and false-D guards.",
            "Baseline selection is explicit when no safe improvement exists; no automatic promotion.",
            "Fresh external holdout and inherited-parameter audit remain required.",
        ],
    }
    _save(output_report, report)
    print("Selected {}: {}".format(selected["name"], report["status"]), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Select guarded possession ablations on Game 1 only.")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--protocol", default=str(Path(__file__).resolve().parents[1] / "configs" / "metrica_guarded_development.json")
    )
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args(argv)
    try:
        calibrate_guarded(args.root, args.protocol, args.output_config, args.output_report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, "error: {}\n".format(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
