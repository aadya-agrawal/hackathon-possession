import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def _round_up(value, step):
    return math.ceil(float(value) / float(step)) * float(step)


def select_external_parameters(metrics, target_fps):
    development = metrics["splits"]["development"]["distributions"]
    validation = metrics["splits"]["validation"]["distributions"]
    selected = {
        "max_ball_speed_metres_per_second": _round_up(
            development["ball_speeds_mps"]["p99"] * 1.10, 5.0
        ),
        "max_ball_acceleration_metres_per_second_squared": _round_up(
            development["ball_accelerations_mps2"]["p90"] * 1.10, 5.0
        ),
        "max_interpolation_gap_seconds": _round_up(
            development["missing_runs_seconds"]["p75"], 0.25
        ),
        "video_recovery_path_maximum_acceleration_mps2": _round_up(
            development["ball_accelerations_mps2"]["p99"] * 1.10, 10.0
        ),
        "video_recovery_kalman_process_acceleration_std_mps2": (
            _round_up(
                development["ball_accelerations_mps2"]["p75"], 5.0
            )
        ),
    }
    selected["max_interpolation_gap_frames"] = int(round(
        selected["max_interpolation_gap_seconds"] * float(target_fps)
    ))
    selected["video_recovery_path_maximum_speed_mps"] = selected[
        "max_ball_speed_metres_per_second"
    ]
    selected["video_recovery_kalman_maximum_missing_frames"] = selected[
        "max_interpolation_gap_frames"
    ]

    checks = [
        {
            "name": "validation_speed_p99",
            "actual": validation["ball_speeds_mps"]["p99"],
            "maximum": selected["max_ball_speed_metres_per_second"],
        },
        {
            "name": "validation_acceleration_p90",
            "actual": validation["ball_accelerations_mps2"]["p90"],
            "maximum": selected[
                "max_ball_acceleration_metres_per_second_squared"
            ],
        },
        {
            "name": "validation_missing_gap_p75",
            "actual": validation["missing_runs_seconds"]["p75"],
            "maximum": selected["max_interpolation_gap_seconds"],
        },
        {
            "name": "validation_process_acceleration_p75",
            "actual": validation["ball_accelerations_mps2"]["p75"],
            "maximum": (
                selected[
                    "video_recovery_kalman_process_acceleration_std_mps2"
                ] * 1.25
            ),
        },
    ]
    for check in checks:
        check["passed"] = check["actual"] <= check["maximum"]
    return selected, checks


def build_compliance_outputs(metrics_path, base_config_path, target_fps):
    metrics_path = Path(metrics_path)
    base_config_path = Path(base_config_path)
    metrics_bytes = metrics_path.read_bytes()
    base_bytes = base_config_path.read_bytes()
    metrics = json.loads(metrics_bytes)
    base_config = json.loads(base_bytes)
    if metrics["policy"].get("test_split_used"):
        raise ValueError("external test split must remain untouched")
    if metrics["policy"].get("detector_outputs_used"):
        raise ValueError(
            "parameter selection must not use detector outputs"
        )

    selected, checks = select_external_parameters(metrics, target_fps)
    failed = [check["name"] for check in checks if not check["passed"]]
    if failed:
        raise ValueError(
            "external validation checks failed: {}".format(failed)
        )

    config = dict(base_config)
    for key, value in selected.items():
        if key == "max_interpolation_gap_seconds":
            continue
        config[key] = value
    config["video_ball_observations_path"] = None
    config["compliance_provenance"] = {
        "mode": "external_calibration_tracking_only",
        "external_metrics": str(metrics_path.resolve()),
        "external_metrics_sha256": hashlib.sha256(
            metrics_bytes
        ).hexdigest(),
        "base_config": str(base_config_path.resolve()),
        "base_config_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "target_fps": float(target_fps),
        "challenge_video_used_for_parameter_selection": False,
        "challenge_ground_truth_used_for_parameter_selection": False,
        "match_specific_video_observations_enabled": False,
        "external_test_split_used": False,
    }
    report = {
        "schema_version": 1,
        "external_metrics": str(metrics_path.resolve()),
        "external_metrics_sha256": hashlib.sha256(
            metrics_bytes
        ).hexdigest(),
        "selection_policy": {
            "speed": "development p99 with 10% margin, rounded up to 5 m/s",
            "acceleration": (
                "development p90 with 10% margin, rounded up to 5 m/s2"
            ),
            "missing_gap": (
                "development p75 rounded up to 0.25 seconds"
            ),
            "kalman_process_acceleration": (
                "development p75 rounded up to 5 m/s2"
            ),
            "path_acceleration": (
                "development p99 with 10% margin, rounded up to 10 m/s2"
            ),
        },
        "target_fps": float(target_fps),
        "selected": selected,
        "validation_checks": checks,
        "passed": True,
        "unused_for_selection": [
            "SoccerNet test split",
            "challenge video",
            "challenge possession ground truth",
            "match-specific video observations",
            "detector outputs",
        ],
    }
    return config, report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Select externally calibrated parameters and build a "
            "tracking-only compliance config."
        )
    )
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--target-fps", type=float, required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args(argv)
    try:
        config, report = build_compliance_outputs(
            args.metrics, args.base_config, args.target_fps
        )
        for path, content in (
            (Path(args.output_config), config),
            (Path(args.output_report), report),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as handle:
                json.dump(content, handle, indent=2, sort_keys=True)
                handle.write("\n")
    except (IOError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, "error: {}\n".format(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
