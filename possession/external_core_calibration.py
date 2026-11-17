import argparse
import json
import sys
from pathlib import Path

import numpy as np

from possession import decoder
from possession.decoder import CORE_CALIBRATION_KEYS
from possession.external_guarded_calibration import _evaluate, _prepare, _save, guarded_checks
from possession.external_metrica_validation import _file_sha256
from possession.cli import load_config


def load_core_protocol(path):
    path = Path(path)
    protocol = json.loads(path.read_text())
    project = Path(__file__).resolve().parents[1]
    if "base_protocol" in protocol:
        extension = protocol
        base_path = project / extension["base_protocol"]
        if _file_sha256(base_path) != extension["base_protocol_sha256"]:
            raise ValueError("interaction base protocol hash differs")
        protocol = json.loads(base_path.read_text())
        if "base_protocol" in protocol:
            raise ValueError("core interaction protocols may have only one base level")
        protocol.update({
            "interaction_profiles": extension["interaction_profiles"],
            "resume_report": extension["resume_report"],
            "resume_report_sha256": extension["resume_report_sha256"],
            "base_protocol_sha256": extension["base_protocol_sha256"],
        })
        if len(protocol["interaction_profiles"]) > 6:
            raise ValueError("interaction pass exceeds its six-vector bound")
    if (
        protocol["development_game"] != "Sample_Game_1"
        or set(protocol["excluded_games"]) != {"Sample_Game_2", "Sample_Game_3"}
    ):
        raise ValueError("core calibration is restricted to Game 1")
    baseline_path = project / protocol["baseline_config"]
    if _file_sha256(baseline_path) != protocol["baseline_sha256"]:
        raise ValueError("core baseline does not match the frozen protocol")
    baseline = load_config(baseline_path)
    if baseline["decoder_mode"] != "semi_markov" or baseline["video_ball_observations_path"] is not None:
        raise ValueError("core calibration requires a tracking-only semi-Markov baseline")
    if not np.isclose(sum(protocol["selection_weights"].values()), 1.0):
        raise ValueError("core objective weights must sum to one")
    for group in ("evidence_profiles", "transition_profiles"):
        profiles = protocol[group]
        if not profiles or profiles[0]["name"] != "baseline":
            raise ValueError("{} must begin with baseline".format(group))
        for profile in profiles:
            if not set(profile["values"]).issubset(CORE_CALIBRATION_KEYS):
                raise ValueError("profile changes a fixed dead/restart or preprocessing parameter")
        if any(baseline[key] != value for key, value in profiles[0]["values"].items()):
            raise ValueError("{} baseline values are inconsistent".format(group))
    for evidence, transition in protocol.get("interaction_profiles", []):
        if (
            evidence not in {item["name"] for item in protocol["evidence_profiles"]}
            or transition not in {item["name"] for item in protocol["transition_profiles"]}
        ):
            raise ValueError("interaction references an unknown profile")
    return protocol, baseline


def choose_core_candidate(experiments, baseline, tolerance):
    safe = [item for item in experiments if item["safe"]]
    if not safe:
        raise ValueError("no safe core candidate, including baseline")
    best = max(item["evaluation"]["objective"] for item in safe)
    tied = [item for item in safe if item["evaluation"]["objective"] >= best - tolerance]
    return min(tied, key=lambda item: (
        sum(item["values"][key] != baseline[key] for key in CORE_CALIBRATION_KEYS),
        experiments.index(item),
    ))


def calibrate_core(root, protocol_path, output_config, output_report):
    root = Path(root).resolve()
    protocol_path = Path(protocol_path)
    protocol, baseline = load_core_protocol(protocol_path)
    prepared = _prepare(root, protocol, baseline)
    experiments = []
    seen = set()
    project = Path(__file__).resolve().parents[1]
    resumed_hash = None
    if "resume_report" in protocol:
        resume_path = project / protocol["resume_report"]
        resumed_hash = _file_sha256(resume_path)
        if resumed_hash != protocol["resume_report_sha256"]:
            raise ValueError("core resume report hash differs")
        previous_report = json.loads(resume_path.read_text())
        if (
            previous_report["protocol_sha256"] != protocol["base_protocol_sha256"]
            or previous_report["baseline_sha256"] != protocol["baseline_sha256"]
            or previous_report["development_game"] != "Sample_Game_1"
            or previous_report["input_sha256"] != {
                path.name: _file_sha256(path) for path in prepared["data"]["input_paths"]
            }
            or previous_report["cases"] != [
                {key: case[key] for key in ("name", "scenario", "tag_mode", "mask_start_rows",
                                          "mask_length_frames", "is_live_stress")}
                for case in prepared["cases"]
            ]
        ):
            raise ValueError("resume data, protocol, or scenarios are not identical")
        for name, digest in previous_report["code_sha256"].items():
            if name == Path(__file__).name:
                continue
            if _file_sha256(project / "possession" / name) != digest:
                raise ValueError("model-relevant code changed since core screening: {}".format(name))
        experiments = previous_report["experiments"]
        seen = {
            tuple(item["values"][key] for key in CORE_CALIBRATION_KEYS)
            for item in experiments
        }

    def evaluate(name, overrides, stage):
        config = {**baseline, **overrides}
        values = {key: config[key] for key in CORE_CALIBRATION_KEYS}
        vector = tuple(values[key] for key in CORE_CALIBRATION_KEYS)
        if vector in seen:
            return
        seen.add(vector)
        result = _evaluate(prepared, config, protocol)
        checks = guarded_checks(
            result, experiments[0]["evaluation"] if experiments else result, protocol
        )
        safe = all(check["passed"] for check in checks)
        experiments.append({
            "name": name, "stage": stage, "values": values,
            "overrides": values, "evaluation": result, "checks": checks, "safe": safe,
        })
        print("{}: objective={:.5f}, safe={}".format(
            name, result["objective"], safe
        ), flush=True)

    if resumed_hash is None:
        for profile in protocol["evidence_profiles"]:
            evaluate(profile["name"], profile["values"], "evidence")
        seed = choose_core_candidate(experiments, baseline, protocol["tie_tolerance"])
        evidence_values = {
            key: seed["values"][key]
            for key in protocol["evidence_profiles"][0]["values"]
        }
        for profile in protocol["transition_profiles"]:
            evaluate(
                seed["name"] + "+" + profile["name"],
                {**evidence_values, **profile["values"]}, "transition",
            )
    for evidence_name, transition_name in protocol.get("interaction_profiles", []):
        evidence = next(item["values"] for item in protocol["evidence_profiles"] if item["name"] == evidence_name)
        transition = next(item["values"] for item in protocol["transition_profiles"] if item["name"] == transition_name)
        evaluate(evidence_name + "+" + transition_name, {**evidence, **transition}, "interaction")
    selected = choose_core_candidate(experiments, baseline, protocol["tie_tolerance"])
    gain = selected["evaluation"]["objective"] - experiments[0]["evaluation"]["objective"]
    improved = gain >= protocol["minimum_objective_improvement"]
    if not improved:
        selected = experiments[0]
    config = {**baseline, **selected["values"]}
    config["core_development_provenance"] = {
        "development_game": "Sample_Game_1",
        "excluded_games": protocol["excluded_games"],
        "baseline_sha256": protocol["baseline_sha256"],
        "protocol_sha256": _file_sha256(protocol_path),
        "development_protocol": str(protocol_path.resolve()),
        "development_report": str(Path(output_report).resolve()),
        "selected_candidate": selected["name"],
        "selected_overrides": selected["values"],
        "externally_reselected_keys": list(CORE_CALIBRATION_KEYS),
        "dead_restart_prediction_settings_reselected": False,
        "full_active_provenance_audit_cleared": False,
    }
    _save(output_config, config)
    data = prepared["data"]
    report = {
        "schema_version": 1, "external_data": True,
        "development_game": "Sample_Game_1", "excluded_games": protocol["excluded_games"],
        "baseline_sha256": protocol["baseline_sha256"],
        "protocol_sha256": _file_sha256(protocol_path),
        "selected_config_sha256": _file_sha256(output_config),
        "selected_candidate": selected["name"],
        "development_improvement_found": improved,
        "status": "frozen_for_diagnostic_release" if improved else "no_safe_core_gain",
        "externally_reselected_keys": list(CORE_CALIBRATION_KEYS),
        "reference_summary": prepared["references"]["summary"],
        "primary_reference_spells": len(prepared["primary_references"]["complete_live_spells"]),
        "input_sha256": {path.name: _file_sha256(path) for path in data["input_paths"]},
        "cases": [
            {key: case[key] for key in ("name", "scenario", "tag_mode", "mask_start_rows",
                                      "mask_length_frames", "is_live_stress")}
            for case in prepared["cases"]
        ],
        "experiments": experiments,
        "resumed_report_sha256": resumed_hash,
        "resumed_model_code_verified": resumed_hash is not None,
        "changed_core_parameters": {
            key: {"baseline": baseline[key], "selected": config[key]}
            for key in CORE_CALIBRATION_KEYS if baseline[key] != config[key]
        },
        "dead_and_restart_parameters_changed": False,
        "challenge_submission_modified": False,
        "code_sha256": {
            path.name: _file_sha256(path) for path in (
                Path(__file__), Path(decoder.__file__),
                project / "possession" / "external_guarded_calibration.py",
                project / "possession" / "external_metrica_validation.py",
                project / "possession" / "external_possession_events.py",
                project / "possession" / "external_transfer_check.py",
            )
        },
        "limitations": [
            "Game 1 is development only; no inspected external or challenge outcomes enter selection.",
            "Provider references and synthetic restart features are not challenge ground truth.",
            "Dead/restart formulas and other inherited settings are fixed, not externally reselected.",
            "Higher development score is not a release decision; a frozen diagnostic pass remains necessary.",
            "Near-tied scores prefer fewer parameter changes rather than microscopic score differences.",
        ],
    }
    _save(output_report, report)
    print("Selected {}: {}".format(selected["name"], report["status"]), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate active core evidence and transition parameters externally.")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--protocol", default=str(Path(__file__).resolve().parents[1] / "configs" / "metrica_core_development.json")
    )
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args(argv)
    try:
        calibrate_core(args.root, args.protocol, args.output_config, args.output_report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, "error: {}\n".format(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
