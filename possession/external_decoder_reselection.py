import argparse
import json
import subprocess
from pathlib import Path

from possession.cli import load_config
from possession.external_guarded_calibration import (
    _evaluate, _prepare, _save, guarded_checks,
)
from possession.external_metrica_validation import _file_sha256


DECODER_KEYS = {
    "semi_markov_live_transition_cost",
    "semi_markov_strong_transition_cost",
    "semi_markov_strong_transition_window_frames",
    "semi_markov_strong_transition_minimum_frames",
    "semi_markov_minimum_live_frames",
}


def choose_decoder_policy(experiments, minimum_gain):
    if not experiments or experiments[0]["name"] != "current_25fps" or not experiments[0]["safe"]:
        raise ValueError("decoder selection requires a guarded 25-fps current comparator")
    current = experiments[0]["evaluation"]["objective"]
    eligible = [
        item for item in experiments[1:]
        if item["safe"] and item["evaluation"]["objective"] >= current + minimum_gain
    ]
    return max(eligible, key=lambda item: item["evaluation"]["objective"]) if eligible else experiments[0]


def reselect_decoder(root, protocol_path, output):
    project = Path(__file__).resolve().parents[1]
    root = Path(root).resolve()
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    base_path = project / protocol["baseline_config"]
    exit_path = project / protocol["exit_candidate_config"]
    prediction_path = project / "artifacts" / "external_prediction_reselection_game1.json"
    if (
        protocol.get("schema_version") != 1
        or protocol["source_fps"] != 25
        or protocol["development_game"] != "Sample_Game_1"
        or protocol["previously_inspected_diagnostics"] != ["Sample_Game_2", "Sample_Game_3"]
        or protocol["original_guards_unchanged"] is not True
        or protocol["challenge_possession_truth_or_video_used"] is not False
        or _file_sha256(base_path) != protocol["baseline_sha256"]
        or _file_sha256(exit_path) != protocol["exit_candidate_sha256"]
        or _file_sha256(prediction_path) != protocol["prediction_source_report_sha256"]
        or subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip() != protocol["external_revision"]
    ):
        raise ValueError("decoder reselection requires frozen external-only inputs and guards")
    vectors = protocol["candidate_vectors"]
    if (
        not vectors or vectors[0] != {"name": "current_25fps", "values": {}}
        or any(not set(item["values"]).issubset(DECODER_KEYS) for item in vectors)
        or any(set(item["values"]) & set(protocol["forbidden_overrides"]) for item in vectors)
    ):
        raise ValueError("decoder vectors change a fixed non-decoder parameter")
    chosen_prediction = json.loads(prediction_path.read_text())["selected"]
    if (
        chosen_prediction["history_frames"] != protocol["prediction_history_frames"]
        or chosen_prediction["horizon_frames"] != protocol["prediction_horizon_frames"]
    ):
        raise ValueError("decoder reselection changed frozen external prediction policy")
    baseline = load_config(base_path)
    source = load_config(exit_path)
    settings = {
        **json.loads((project / protocol["source_scenario_protocol"]).read_text()),
        "source_fps": protocol["source_fps"],
    }
    prepared = _prepare(root, settings, baseline)
    deployment = _evaluate(prepared, baseline, settings)
    core = {
        **source,
        "prediction_history_frames": chosen_prediction["history_frames"],
        "prediction_horizon_frames": chosen_prediction["horizon_frames"],
    }
    experiments = []
    for vector in vectors:
        evaluated = _evaluate(prepared, {**core, **vector["values"]}, settings)
        checks = guarded_checks(evaluated, deployment, settings)
        experiments.append({
            "name": vector["name"], "values": vector["values"],
            "evaluation": evaluated, "checks": checks,
            "safe": all(item["passed"] for item in checks),
        })
        print("{} objective={:.6f} guards={}/{}".format(
            vector["name"], evaluated["objective"],
            sum(item["passed"] for item in checks), len(checks),
        ), flush=True)
    selected = choose_decoder_policy(experiments, protocol["minimum_objective_gain_vs_exit_candidate"])
    report = {
        "schema_version": 1, "development_game": "Sample_Game_1",
        "source_protocol_sha256": _file_sha256(protocol_path),
        "base_config_sha256": protocol["baseline_sha256"],
        "exit_candidate_sha256": protocol["exit_candidate_sha256"],
        "prediction_source_report_sha256": protocol["prediction_source_report_sha256"],
        "external_revision": protocol["external_revision"],
        "input_sha256": {path.name: _file_sha256(path) for path in prepared["data"]["input_paths"]},
        "code_sha256": {
            path.name: _file_sha256(path) for path in (
                project / "possession" / "decoder.py",
                project / "possession" / "external_guarded_calibration.py",
                project / "possession" / "external_possession_events.py",
            )
        },
        "baseline_evaluation": deployment,
        "experiments": experiments, "selected": selected["name"],
        "original_guards_unmodified": True,
        "challenge_video_or_possession_truth_used": False,
        "selected_submission_changed": False,
    }
    _save(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reselect semi-Markov block on corrected external Game 1")
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", default="configs/external_decoder_reselection.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    reselect_decoder(args.root, args.protocol, args.output)


if __name__ == "__main__":
    main()
