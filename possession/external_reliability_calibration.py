import argparse
import json
import sys
from pathlib import Path

from possession.decoder import CORE_CALIBRATION_KEYS, CORE_RELIABILITY_KEYS
from possession.external_core_calibration import load_core_protocol
from possession.external_guarded_calibration import _evaluate, _prepare, _save, guarded_checks
from possession.external_metrica_validation import _file_sha256


def reliability_overrides(baseline, mode):
    if mode not in ("geometry", "transitions", "both"):
        raise ValueError("reliability ablation must be geometry, transitions, or both")
    result = {"core_reliability_mode": mode}
    if mode in ("geometry", "both"):
        result.update({
            "core_uncertain_closeness_scale_metres": baseline["closeness_scale_metres"],
            "core_uncertain_nearby_two_weight": baseline["nearby_two_weight"],
            "core_uncertain_nearby_four_weight": baseline["nearby_four_weight"],
        })
    if mode in ("transitions", "both"):
        result.update({
            "core_uncertain_live_transition_cost": baseline["semi_markov_live_transition_cost"],
            "core_uncertain_strong_transition_cost": baseline["semi_markov_strong_transition_cost"],
            "core_uncertain_strong_transition_window_frames": baseline["semi_markov_strong_transition_window_frames"],
            "core_uncertain_strong_transition_minimum_frames": baseline["semi_markov_strong_transition_minimum_frames"],
            "core_uncertain_strong_control_distance_metres": baseline["strong_control_distance_metres"],
            "core_uncertain_strong_control_distance_margin_metres": baseline["strong_control_distance_margin_metres"],
        })
    return result


def load_reliability_protocol(path):
    path = Path(path)
    extension = json.loads(path.read_text())
    project = Path(__file__).resolve().parents[1]
    base_path = project / extension["base_protocol"]
    if _file_sha256(base_path) != extension["base_protocol_sha256"]:
        raise ValueError("reliability base protocol changed")
    protocol, baseline = load_core_protocol(base_path)
    if extension["modes"] != ["geometry", "transitions", "both"]:
        raise ValueError("reliability rescue is limited to the three declared ablations")
    report_path = project / extension["source_report"]
    if _file_sha256(report_path) != extension["source_report_sha256"]:
        raise ValueError("reliability source report changed")
    source_report = json.loads(report_path.read_text())
    if (
        source_report["development_game"] != "Sample_Game_1"
        or source_report["baseline_sha256"] != protocol["baseline_sha256"]
        or set(source_report["excluded_games"]) != {"Sample_Game_2", "Sample_Game_3"}
    ):
        raise ValueError("reliability rescue may use only Game 1 development lineage")
    source = next(
        item for item in source_report["experiments"] if item["name"] == extension["source_candidate"]
    )
    if set(source["values"]) != set(CORE_CALIBRATION_KEYS):
        raise ValueError("source candidate changes more than the declared core settings")
    if source["values"]["semi_markov_control_margin_weight"] != baseline["semi_markov_control_margin_weight"]:
        raise ValueError("geometry rescue requires the shared emission margin weight")
    return extension, protocol, baseline, source_report, source


def calibrate_reliability(root, protocol_path, output_config, output_report):
    root = Path(root).resolve()
    protocol_path = Path(protocol_path)
    extension, protocol, baseline, previous, source = load_reliability_protocol(protocol_path)
    prepared = _prepare(root, protocol, baseline)
    data = prepared["data"]
    inputs = {path.name: _file_sha256(path) for path in data["input_paths"]}
    if inputs != previous["input_sha256"]:
        raise ValueError("reliability development inputs differ from the original study")
    cases = [
        {key: case[key] for key in ("name", "scenario", "tag_mode", "mask_start_rows",
                                  "mask_length_frames", "is_live_stress")}
        for case in prepared["cases"]
    ]
    if cases != previous["cases"]:
        raise ValueError("reliability development scenarios differ")
    baseline_values = {key: baseline[key] for key in CORE_CALIBRATION_KEYS}
    candidates = [
        ("baseline", baseline_values, True),
        ("near_miss", source["values"], False),
    ]
    for mode in extension["modes"]:
        candidates.append((mode, {**source["values"], **reliability_overrides(baseline, mode)}, True))
    experiments = []
    for name, values, selectable in candidates:
        evaluation = _evaluate(prepared, {**baseline, **values}, protocol)
        if name in ("baseline", "near_miss"):
            old = previous["experiments"][0]["evaluation"] if name == "baseline" else source["evaluation"]
            if any(
                result["label_sha256"] != old["cases"][case]["label_sha256"]
                for case, result in evaluation["cases"].items()
            ):
                raise ValueError("disabled reliability mode changed original {} outputs".format(name))
        checks = guarded_checks(
            evaluation, experiments[0]["evaluation"] if experiments else evaluation, protocol
        )
        safe = all(check["passed"] for check in checks)
        experiments.append({
            "name": name, "overrides": values, "evaluation": evaluation,
            "checks": checks, "safe": safe, "selection_allowed": selectable,
        })
        print("{}: objective={:.5f}, safe={}, failures={}".format(
            name, evaluation["objective"], safe, sum(not check["passed"] for check in checks)
        ), flush=True)
    eligible = [item for item in experiments if item["safe"] and item["selection_allowed"]]
    best_score = max(item["evaluation"]["objective"] for item in eligible)
    tied = [
        item for item in eligible
        if item["evaluation"]["objective"] >= best_score - protocol["tie_tolerance"]
    ]
    selected = min(tied, key=lambda item: (
        len(set(item["overrides"]) - set(CORE_CALIBRATION_KEYS)), experiments.index(item)
    ))
    gain = selected["evaluation"]["objective"] - experiments[0]["evaluation"]["objective"]
    improved = gain >= protocol["minimum_objective_improvement"]
    if not improved:
        selected = experiments[0]
    config = {**baseline, **selected["overrides"]}
    config["core_development_provenance"] = {
        "development_game": "Sample_Game_1",
        "excluded_games": ["Sample_Game_2", "Sample_Game_3"],
        "baseline_sha256": protocol["baseline_sha256"],
        "protocol_sha256": _file_sha256(protocol_path),
        "development_protocol": str(protocol_path.resolve()),
        "development_report": str(Path(output_report).resolve()),
        "selected_candidate": selected["name"],
        "selected_overrides": selected["overrides"],
        "base_protocol_sha256": extension["base_protocol_sha256"],
        "source_report_sha256": extension["source_report_sha256"],
        "dead_restart_prediction_settings_reselected": False,
        "full_active_provenance_audit_cleared": False,
    }
    _save(output_config, config)
    project = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1, "external_data": True,
        "development_game": "Sample_Game_1", "excluded_games": ["Sample_Game_2", "Sample_Game_3"],
        "baseline_sha256": protocol["baseline_sha256"],
        "protocol_sha256": _file_sha256(protocol_path),
        "base_protocol_sha256": extension["base_protocol_sha256"],
        "source_report_sha256": extension["source_report_sha256"],
        "selected_config_sha256": _file_sha256(output_config),
        "selected_candidate": selected["name"],
        "development_improvement_found": improved,
        "status": "reliability_rescue_ready_for_diagnostic_release" if improved else "no_safe_reliability_rescue",
        "input_sha256": inputs, "cases": cases,
        "reference_summary": prepared["references"]["summary"],
        "primary_reference_spells": len(prepared["primary_references"]["complete_live_spells"]),
        "experiments": experiments,
        "disabled_mode_comparators_reproduced": True,
        "acceptance_thresholds_changed": False,
        "challenge_submission_modified": False,
        "code_sha256": {
            path.name: _file_sha256(path) for path in (
                Path(__file__), project / "possession" / "decoder.py",
                project / "possession" / "external_guarded_calibration.py",
                project / "possession" / "external_metrica_validation.py",
                project / "possession" / "external_transfer_check.py",
            )
        },
        "limitations": [
            "Three policy ablations on Game 1 only; no new parameter grid or validation feedback.",
            "Fallback values equal the current baseline; no guard is relaxed to force acceptance.",
            "Observed source-1 input is treated as reliable; other sources use the selected fallback policy.",
            "Provider references and simulated restart tags are not challenge ground truth.",
            "A development gain is not a release decision; inherited provenance remains partly unresolved.",
        ],
    }
    _save(output_report, report)
    print("Selected {}: {}".format(selected["name"], report["status"]), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Test three reliability-conditioned rescues on Game 1.")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--protocol", default=str(Path(__file__).resolve().parents[1] / "configs" / "metrica_reliability_development.json")
    )
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args(argv)
    try:
        calibrate_reliability(args.root, args.protocol, args.output_config, args.output_report)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        parser.exit(1, "error: {}\n".format(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
