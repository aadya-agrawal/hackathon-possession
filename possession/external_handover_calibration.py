import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from possession import UNKNOWN
from possession.cli import load_config
from possession.decoder import build_evidence, decode_semi_markov
from possession.external_metrica_validation import (
    _file_sha256,
    _load_events,
    _load_game,
    _period_spell_diagnostics,
    _repair_by_period,
    config_at_fps,
)
from possession.external_possession_events import (
    possession_chain_references,
    recovery_masks,
    score_possession_references,
    score_recovery_transitions,
)
from possession.tagging import TagFeatures
from possession.tracking import control_features_from_arrays


def load_handover_protocol(path):
    protocol = json.loads(Path(path).read_text())
    splits = protocol["splits"]
    if set(splits) != {"development", "diagnostic_validation", "unevaluated_test"}:
        raise ValueError("handover protocol requires three explicit split roles")
    games = [game for values in splits.values() for game in values]
    if (
        any(not values for values in splits.values())
        or len(games) != len(set(games))
        or set(games) != {"Sample_Game_1", "Sample_Game_2", "Sample_Game_3"}
    ):
        raise ValueError("Metrica calibration games must be nonempty and split-isolated")
    if not np.isclose(sum(protocol["selection_weights"].values()), 1.0):
        raise ValueError("selection objective weights must sum to one")
    if protocol["source_fps"] <= 0:
        raise ValueError("calibration source fps must be positive")
    return protocol


def handover_candidates(protocol):
    local = protocol["local_handover"]
    grid = protocol["candidate_grid"]
    result = [("baseline", {})]
    for radius, duration, cost in itertools.product(
        grid["distance_metres"], grid["minimum_live_frames"], grid["strong_transition_cost"]
    ):
        result.append((
            "local-r{}-live{}-cost{}".format(radius, duration, cost),
            {
                "semi_markov_local_handover_enabled": True,
                "semi_markov_local_handover_window_frames": local["window_frames"],
                "semi_markov_local_handover_minimum_frames": local["minimum_frames"],
                "semi_markov_local_handover_distance_metres": radius,
                "semi_markov_local_handover_distance_margin_metres": local["distance_margin_metres"],
                "semi_markov_local_handover_minimum_confidence": local["minimum_confidence"],
                "semi_markov_minimum_live_frames": duration,
                "semi_markov_strong_transition_cost": cost,
            },
        ))
    return result


def stoppage_candidates(protocol):
    settings = protocol["tracking_stoppage"]
    result = [("baseline", {})]
    for speed, missing in itertools.product(
        settings["maximum_median_speed_mps"], settings["minimum_missing_frames"]
    ):
        result.append((
            "stoppage-speed{}-missing{}".format(speed, missing),
            {
                "tracking_stoppage_enabled": True,
                "tracking_stoppage_maximum_median_speed_mps": speed,
                "tracking_stoppage_minimum_missing_frames": missing,
                "tracking_stoppage_motion_window_frames": settings["motion_window_frames"],
                "tracking_stoppage_minimum_slow_fraction": settings["minimum_slow_fraction"],
                "tracking_stoppage_minimum_stationary_fraction": settings["minimum_stationary_fraction"],
                "tracking_stoppage_dead_weight": settings["dead_weight"],
            },
        ))
    return result


def _prepare_game(root, game, config, protocol):
    data = _load_game(root, game)
    effective = config_at_fps(config, protocol["source_fps"], data["fps"])
    events = _load_events(data["event_path"], {
        "PASS", "RECOVERY", "BALL LOST", "BALL OUT", "SHOT", "SET PIECE",
        "FAULT RECEIVED",
    })
    truth, references = possession_chain_references(
        events, data["frames"], data["periods"], data["fps"], protocol["references"]
    )
    if not references["complete_live_spells"]:
        raise ValueError("{}: no uncensored complete provider-reference spells".format(game))
    if any(references["summary"]["frames_by_state"][state] == 0 for state in ("H", "A", "D")):
        raise ValueError("{}: calibration requires supported H/A/D reference frames".format(game))
    recovery = references["recovery_references"]
    masked, starts, length = recovery_masks(
        data["ball"], data["periods"], recovery, data["fps"],
        protocol["recovery_scoring"]["mask_seconds"],
    )
    if not starts:
        raise ValueError("{}: no observed recovery-centred masks".format(game))
    targeted = [
        reference for reference in recovery
        if not np.isfinite(masked[reference["row"]]).all()
    ]
    edges = np.r_[0, np.flatnonzero(np.diff(data["periods"])) + 1, len(data["ball"])]
    prepared = {}
    for mode, ball in (("clean", data["ball"]), ("recovery_masked", masked)):
        periods = []
        for start, stop in zip(edges[:-1], edges[1:]):
            trajectory = _repair_by_period(ball[start:stop], data["fps"], effective)
            control = control_features_from_arrays(
                data["home"][start:stop], data["away"][start:stop],
                trajectory.xy, data["fps"], effective,
            )
            n_frames = stop - start
            tags = TagFeatures(
                team_prior=np.zeros(n_frames, dtype=np.float32),
                dead_hint=np.zeros(n_frames, dtype=bool),
                pre_restart_window=np.zeros(n_frames, dtype=bool),
                restart_team=np.full(n_frames, UNKNOWN, dtype=np.int8),
            )
            periods.append((trajectory, control, tags))
        prepared[mode] = periods
    return {
        "game": game, "fps": data["fps"], "periods": data["periods"],
        "truth": truth, "references": references, "prepared": prepared,
        "targeted_recoveries": targeted, "mask_start_rows": starts,
        "mask_length_frames": length,
        "input_sha256": {
            path.name: _file_sha256(path) for path in (
                data["event_path"],
                root / "data" / game / (game + "_RawTrackingData_Home_Team.csv"),
                root / "data" / game / (game + "_RawTrackingData_Away_Team.csv"),
            )
        },
    }


def _evaluate(game, source_config, protocol):
    config = config_at_fps(source_config, protocol["source_fps"], game["fps"])
    modes = {}
    for mode, periods in game["prepared"].items():
        labels = np.concatenate([
            decode_semi_markov(
                build_evidence(control, trajectory, tags, (105.0, 68.0), config, fps=game["fps"]),
                config,
            ) for trajectory, control, tags in periods
        ])
        modes[mode] = {
            "possession": score_possession_references(
                labels, game["truth"], game["references"], game["periods"]
            ),
            "recovery": score_recovery_transitions(
                labels, game["periods"], game["references"]["recovery_references"],
                game["fps"], protocol["recovery_scoring"],
            ),
            "targeted_recovery": score_recovery_transitions(
                labels, game["periods"], game["targeted_recoveries"],
                game["fps"], protocol["recovery_scoring"],
            ),
            "spells": _period_spell_diagnostics(labels, game["fps"], game["periods"]),
        }
    weights = protocol["selection_weights"]
    objective = []
    for mode, metrics in modes.items():
        objective.append(
            weights["reference_spell_f1"]
            * metrics["possession"]["reference_conditioned_spell_scores"]["0.5"]["f1"]
            + weights["boundary_recall"]
            * metrics["recovery"]["same_direction_boundary_recall"]["0.5"]
            + weights["recovery_accuracy"]
            * (
                metrics["targeted_recovery"]["accuracy_at_recovery"]
                if mode == "recovery_masked" else metrics["recovery"]["accuracy_at_recovery"]
            )
            + weights["live_macro_f1"] * metrics["possession"]["known_live_macro_f1"]
            + weights.get("frame_macro_f1", 0.0)
            * metrics["possession"]["known_frame_macro_f1"]
        )
    return {"objective": float(np.mean(objective)), "modes": modes}


def _guards(result, baseline, protocol, validation=False):
    limits = protocol["guards"]
    checks = []
    for mode in ("clean", "recovery_masked"):
        current = result["modes"][mode]
        reference = baseline["modes"][mode]
        for name, actual, minimum in (
            ("live_macro_f1", current["possession"]["known_live_macro_f1"],
             reference["possession"]["known_live_macro_f1"] - limits["maximum_live_macro_f1_drop"]),
            ("dead_recall", current["possession"]["per_state"]["D"]["recall"],
             reference["possession"]["per_state"]["D"]["recall"] - limits["maximum_dead_recall_drop"]),
        ):
            checks.append({
                "name": "{}_{}".format(mode, name),
                "passed": bool(actual >= minimum), "actual": actual, "minimum": minimum,
            })
        actual = current["spells"]["shorter_than_one_second"]
        maximum = reference["spells"]["shorter_than_one_second"] + limits["maximum_short_spell_increase"]
        checks.append({
            "name": "{}_short_spells".format(mode),
            "passed": actual <= maximum, "actual": actual, "maximum": maximum,
        })
        if validation or protocol.get("require_development_recovery_non_regression", False):
            for name, actual, minimum in (
                ("boundary_recall", current["recovery"]["same_direction_boundary_recall"]["0.5"],
                 reference["recovery"]["same_direction_boundary_recall"]["0.5"] - limits["maximum_boundary_recall_drop"]),
                ("targeted_recovery", current["targeted_recovery"]["accuracy_at_recovery"],
                 reference["targeted_recovery"]["accuracy_at_recovery"] - limits["maximum_masked_recovery_drop"]),
            ):
                checks.append({
                    "name": "{}_{}".format(mode, name),
                    "passed": bool(actual >= minimum), "actual": actual, "minimum": minimum,
                })
            actual = current["recovery"]["accuracy_at_recovery"]
            minimum = (
                reference["recovery"]["accuracy_at_recovery"]
                - limits.get("maximum_clean_recovery_drop", 0.02)
            )
            checks.append({
                "name": "{}_recovery_accuracy".format(mode),
                "passed": bool(actual >= minimum), "actual": actual, "minimum": minimum,
            })
    if validation:
        minimum = baseline["objective"] - limits["maximum_validation_objective_drop"]
        checks.append({
            "name": "validation_objective",
            "passed": result["objective"] >= minimum,
            "actual": result["objective"], "minimum": minimum,
        })
    return checks


def _save_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(content, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def calibrate_handover(root, protocol_path, config_output, report_output, references_output):
    root = Path(root).resolve()
    protocol_path = Path(protocol_path)
    protocol = load_handover_protocol(protocol_path)
    project = Path(__file__).resolve().parents[1]
    base_path = project / protocol["base_config"]
    base = load_config(base_path)
    if base["video_ball_observations_path"] is not None:
        raise ValueError("external handover calibration must not load match-specific video")
    if len(protocol["splits"]["development"]) != 1 or len(protocol["splits"]["diagnostic_validation"]) != 1:
        raise ValueError("this calibration protocol requires one development and one diagnostic game")
    development = _prepare_game(
        root, protocol["splits"]["development"][0], base, protocol
    )
    print("Development reference coverage: {:.2%}, complete spells: {}".format(
        development["references"]["summary"]["coverage"],
        development["references"]["summary"]["complete_live_spells"],
    ), flush=True)
    experiments = []
    specs = (
        stoppage_candidates(protocol) if "tracking_stoppage" in protocol
        else handover_candidates(protocol)
    )
    for name, overrides in specs:
        result = _evaluate(development, {**base, **overrides}, protocol)
        checks = _guards(result, experiments[0]["evaluation"] if experiments else result, protocol)
        eligible = all(check["passed"] for check in checks)
        experiments.append({
            "name": name, "overrides": overrides, "evaluation": result,
            "development_guards": checks, "eligible": eligible,
        })
        print("{}: objective={:.4f} eligible={}".format(
            name, result["objective"], eligible
        ), flush=True)
    if "tracking_stoppage" in protocol:
        seed = max(
            [item for item in experiments if item["eligible"]],
            key=lambda item: item["evaluation"]["objective"],
        )
        for name, handover in handover_candidates(protocol)[1:]:
            overrides = {**seed["overrides"], **handover}
            result = _evaluate(development, {**base, **overrides}, protocol)
            checks = _guards(result, experiments[0]["evaluation"], protocol)
            eligible = all(check["passed"] for check in checks)
            full_name = seed["name"] + "+" + name
            experiments.append({
                "name": full_name, "overrides": overrides, "evaluation": result,
                "development_guards": checks, "eligible": eligible,
            })
            print("{}: objective={:.4f} eligible={}".format(
                full_name, result["objective"], eligible
            ), flush=True)
    eligible = [item for item in experiments if item["eligible"]]
    selected = max(eligible, key=lambda item: item["evaluation"]["objective"])
    candidate = {**base, **selected["overrides"]}
    candidate["external_handover_provenance"] = {
        "development_game": development["game"],
        "protocol_sha256": _file_sha256(protocol_path),
        "base_config_sha256": _file_sha256(base_path),
        "selected_candidate": selected["name"],
        "externally_selected_overrides": selected["overrides"],
        "inherited_parameters_fully_external_audited": False,
        "challenge_data_used": False,
        "provider_references_are_challenge_ground_truth": False,
    }
    _save_json(config_output, candidate)
    frozen_hash = _file_sha256(config_output)
    print("Frozen {} before diagnostic validation".format(selected["name"]), flush=True)
    validation = _prepare_game(
        root, protocol["splits"]["diagnostic_validation"][0], base, protocol
    )
    validation_baseline = _evaluate(validation, base, protocol)
    validation_candidate = _evaluate(validation, candidate, protocol)
    validation_checks = _guards(validation_candidate, validation_baseline, protocol, validation=True)
    reference_report = {
        "schema_version": 1,
        "dataset": "Metrica Sports provider-event-derived possession",
        "label_codes": {"-1": "unknown", "0": "H", "1": "A", "2": "D"},
        "protocol_sha256": _file_sha256(protocol_path),
        "unevaluated_games": protocol["splits"]["unevaluated_test"],
        "games": [
            {"game": game["game"], "role": role, "fps": game["fps"],
             "input_sha256": game["input_sha256"], "references": game["references"]}
            for role, game in (("development", development), ("diagnostic_validation", validation))
        ],
    }
    _save_json(references_output, reference_report)
    report = {
        "schema_version": 1,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": _file_sha256(protocol_path),
        "split_authorization": protocol["split_authorization"],
        "selected_candidate": selected["name"],
        "candidate_config_sha256": frozen_hash,
        "development": {
            "game": development["game"], "experiments": experiments,
            "reference_summary": development["references"]["summary"],
        },
        "diagnostic_validation": {
            "game": validation["game"], "baseline": validation_baseline,
            "candidate": validation_candidate, "checks": validation_checks,
            "passed": all(check["passed"] for check in validation_checks),
            "reference_summary": validation["references"]["summary"],
        },
        "unevaluated_games": protocol["splits"]["unevaluated_test"],
        "references_sha256": _file_sha256(references_output),
        "code_sha256": {
            path.name: _file_sha256(path) for path in (
                Path(__file__), project / "possession" / "external_possession_events.py",
                project / "possession" / "decoder.py", project / "possession" / "tracking.py",
                project / "possession" / "external_metrica_validation.py",
                project / "possession" / "trajectory.py",
            )
        },
        "challenge_submission_modified": False,
        "limitations": [
            "Provider-event-derived references are not challenge-equivalent ground truth.",
            "Unknown/censored regions and incomplete spells are explicitly excluded.",
            "Game 2 is previously inspected diagnostic validation, not a pristine test.",
            "Events are scoring-only; inference has no provider Smart Tagging inputs.",
            "Only local handover overrides are externally selected; inherited parameters remain unaudited.",
            "No automatic candidate promotion; Game 3 and SoccerNet test remain unevaluated.",
        ],
    }
    _save_json(report_output, report)
    print("Diagnostic validation passed: {}".format(
        report["diagnostic_validation"]["passed"]
    ), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate handovers on external provider references.")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--protocol",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "metrica_handover_calibration.json"),
    )
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-references", required=True)
    args = parser.parse_args(argv)
    try:
        report = calibrate_handover(
            args.root, args.protocol, args.output_config, args.output_report,
            args.output_references,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, "error: {}\n".format(error))
    return 0 if report["diagnostic_validation"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
