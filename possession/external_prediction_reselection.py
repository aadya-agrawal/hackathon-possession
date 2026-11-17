import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from possession import AWAY, HOME
from possession.cli import load_config
from possession.external_guarded_calibration import _prepare, _save
from possession.external_metrica_validation import (
    _file_sha256, _repair_by_period, _select_windows,
)


def choose_prediction_policy(candidates):
    if not candidates or any(
        not np.isfinite(item["objective"])
        or item["history_frames"] < 1 or item["horizon_frames"] < 1
        for item in candidates
    ):
        raise ValueError("prediction selection needs finite, positive candidate measurements")
    return max(
        candidates,
        key=lambda item: (
            item["objective"], -item["horizon_frames"], -item["history_frames"],
        ),
    )


def calibrate_prediction(root, protocol_path, output):
    project = Path(__file__).resolve().parents[1]
    root = Path(root).resolve()
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    base_path = project / protocol["baseline_config"]
    profile_path = project / "artifacts" / "challenge_input_distribution.json"
    if (
        protocol.get("schema_version") != 1
        or protocol["source_fps"] != 25
        or protocol["development_game"] != "Sample_Game_1"
        or protocol["frozen_diagnostics"] != ["Sample_Game_2", "Sample_Game_3"]
        or protocol["challenge_possession_truth_used"] is not False
        or protocol["challenge_video_output_review_used"] is not False
        or _file_sha256(base_path) != protocol["baseline_sha256"]
        or _file_sha256(profile_path) != protocol["challenge_input_profile_sha256"]
        or subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip() != protocol["external_revision"]
    ):
        raise ValueError("prediction reselection requires frozen external and input-only protocols")
    for key in ("prediction_history_frames", "prediction_horizon_frames", "masked_gap_lengths_frames"):
        values = protocol[key]
        if not values or any(not isinstance(value, int) or isinstance(value, bool) or value < 1
                             for value in values) or len(set(values)) != len(values):
            raise ValueError("{} requires unique positive frame counts".format(key))
    radius = protocol["correct_position_radius_metres"]
    penalty = protocol["wrong_position_penalty_relative_to_abstention"]
    if not np.isfinite([radius, penalty]).all() or radius <= 0 or penalty <= 0:
        raise ValueError("prediction scoring needs finite positive radius and error penalty")
    baseline = load_config(base_path)
    settings = {**json.loads((project / protocol["source_scenario_protocol"]).read_text()),
                "source_fps": protocol["source_fps"]}
    prepared = _prepare(root, settings, baseline)
    data = prepared["data"]
    if data["fps"] != 25:
        raise ValueError("prediction calibration requires physical 25-fps external tracking")
    ball, periods = data["ball"], data["periods"]
    eligible = ball.copy()
    eligible[~np.isin(prepared["truth"], (HOME, AWAY))] = np.nan
    masked_cases = {}
    for length in protocol["masked_gap_lengths_frames"]:
        starts = _select_windows(eligible, length, protocol["windows_per_length"], periods)
        if len(starts) != protocol["windows_per_length"]:
            raise ValueError("{}-frame gap lacks requested independent known-live windows".format(length))
        masked = ball.copy()
        selected = np.zeros(len(ball), dtype=bool)
        for start in starts:
            masked[start:start + length] = np.nan
            selected[start:start + length] = True
        if not np.isfinite(ball[selected]).all():
            raise ValueError("masked ball reference contains unobserved external frames")
        masked_cases[str(length)] = (masked, selected)
    candidates = []
    for history in protocol["prediction_history_frames"]:
        for horizon in protocol["prediction_horizon_frames"]:
            config = {
                **baseline, "prediction_history_frames": history,
                "prediction_horizon_frames": horizon,
            }
            results = {}
            for length, (masked, selected) in masked_cases.items():
                repaired = _repair_by_period(masked, data["fps"], config, periods)
                recovered = selected & np.isfinite(repaired.xy).all(axis=1)
                error = np.linalg.norm(repaired.xy[recovered] - ball[recovered], axis=1)
                correct = int(np.count_nonzero(error <= radius))
                wrong = int(np.count_nonzero(error > radius))
                results[length] = {
                    "masked_frames": int(np.count_nonzero(selected)),
                    "correct_within_radius": correct, "wrong_outside_radius": wrong,
                    "unfilled_frames": int(np.count_nonzero(selected)) - correct - wrong,
                    "utility_per_masked_frame": (
                        correct - penalty * wrong
                    ) / int(np.count_nonzero(selected)),
                }
            candidates.append({
                "history_frames": history, "horizon_frames": horizon,
                "objective": float(np.mean([
                    result["utility_per_masked_frame"] for result in results.values()
                ])),
                "cases": results,
            })
    selected = choose_prediction_policy(candidates)
    report = {
        "schema_version": 1, "development_game": "Sample_Game_1",
        "source_fps": 25, "external_revision": protocol["external_revision"],
        "source_protocol_sha256": _file_sha256(protocol_path),
        "baseline_sha256": _file_sha256(base_path),
        "challenge_input_profile_sha256": _file_sha256(profile_path),
        "input_sha256": {path.name: _file_sha256(path) for path in data["input_paths"]},
        "code_sha256": {
            path.name: _file_sha256(path) for path in (
                Path(__file__), project / "possession" / "trajectory.py",
                project / "possession" / "external_metrica_validation.py",
            )
        },
        "candidates": candidates,
        "selected": {
            key: selected[key] for key in ("history_frames", "horizon_frames", "objective")
        },
        "label_data_used_in_inference": False,
        "provider_known_live_used_only_for_mask_selection": True,
        "challenge_video_or_possession_truth_used": False,
        "candidate_automatically_released": False,
    }
    _save(output, report)
    print("External prediction selection: history={} horizon={} objective={:.5f}".format(
        selected["history_frames"], selected["horizon_frames"], selected["objective"],
    ))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reselect ball prediction history/horizon externally")
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", default="configs/external_prediction_reselection.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    calibrate_prediction(args.root, args.protocol, args.output)


if __name__ == "__main__":
    main()
