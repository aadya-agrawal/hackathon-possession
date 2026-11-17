import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from possession import AWAY, DEAD, HOME, UNKNOWN
from possession.cli import load_config
from possession.decoder import CORE_CALIBRATION_KEYS, CORE_RELIABILITY_KEYS, build_evidence, decode
from possession.external_metrica_validation import (
    _file_sha256, _load_events, _load_game, _period_spell_diagnostics,
    _repair_by_period, _select_windows, config_at_fps,
)
from possession.external_possession_events import (
    possession_chain_references, recovery_masks, score_possession_references, score_recovery_transitions,
)
from possession.tagging import TagFeatures, infer_restart_ownership, localize_restarts
from possession.tracking import control_features_from_arrays


def load_frozen_transfer_protocol(path):
    protocol = json.loads(Path(path).read_text())
    project = Path(__file__).resolve().parents[1]
    configs = {}
    for name in ("baseline", "candidate"):
        config_path = project / protocol[name + "_config"]
        if _file_sha256(config_path) != protocol[name + "_sha256"]:
            raise ValueError("{} config differs from the frozen protocol".format(name))
        configs[name] = load_config(config_path)
        if configs[name]["video_ball_observations_path"] is not None:
            raise ValueError("frozen transfer checks cannot use match-specific video")
    changes = {
        key for key in set(configs["baseline"]) | set(configs["candidate"])
        if configs["baseline"].get(key) != configs["candidate"].get(key)
    }
    scope = protocol.get("candidate_scope")
    diagnostic_release = scope in ("gap_release", "core_release")
    gap_release = scope == "gap_release"
    prefixes = ("semi_markov_gap_handover_",) if gap_release else ("tracking_stoppage_",)
    metadata_key = "guarded_development_provenance" if gap_release else "external_handover_provenance"
    if scope == "core_release":
        if not changes.issubset(set(CORE_CALIBRATION_KEYS) | set(CORE_RELIABILITY_KEYS) | {"core_development_provenance"}):
            raise ValueError("core candidate changes fixed preprocessing or dead/restart parameters")
    elif any(not key.startswith(prefixes) and key != metadata_key for key in changes):
        raise ValueError("candidate changes are incompatible with shared preprocessing")
    expected_games = (
        {"reused_diagnostic": ["Sample_Game_2", "Sample_Game_3"]} if diagnostic_release else
        {"diagnostic_validation": ["Sample_Game_2"], "final_test": ["Sample_Game_3"]}
    )
    if protocol["games"] != expected_games:
        raise ValueError("frozen check has invalid external game roles")
    return protocol, configs


def restart_only_tags(events, periods, fps, config, settings, seed):
    n_frames = len(periods)
    team_delta = np.zeros(n_frames + 1, dtype=np.int32)
    count_delta = np.zeros(n_frames + 1, dtype=np.int32)
    dead_delta = np.zeros(n_frames + 1, dtype=np.int32)
    pre_delta = np.zeros(n_frames + 1, dtype=np.int32)
    restart_team = np.full(n_frames, UNKNOWN, dtype=np.int8)
    restart_events, restart_types = [], []
    if settings is not None:
        rng = np.random.default_rng(seed)
        period_starts = {
            int(period): int(np.flatnonzero(periods == period)[0])
            for period in np.unique(periods)
        }
        period_stops = {
            int(period): int(np.flatnonzero(periods == period)[-1]) + 1
            for period in np.unique(periods)
        }
        first_kickoff = {}
        for event in sorted(
            [item for item in events if item["type"] == "SET PIECE"],
            key=lambda item: (item["frame"], item["team"], item["subtype"]),
        ):
            if "RETAKEN" in event["subtype"]:
                continue
            is_first = event["period"] not in first_kickoff
            if event["subtype"] == "KICK OFF":
                first_kickoff[event["period"]] = event["frame"]
            if rng.random() < settings["drop_fraction"]:
                continue
            shift = int(round(rng.uniform(
                -settings["jitter_seconds"], settings["jitter_seconds"]
            ) * fps))
            lower = period_starts[event["period"]]
            upper = period_stops[event["period"]]
            restart = max(lower, min(upper - 1, event["frame"] - 1 + shift))
            start = max(lower, restart - config["restart_offset_frames"])
            end = min(upper, restart + config["restart_prior_frames"])
            if end <= start:
                raise ValueError("synthetic restart tag has empty bounds")
            team = HOME if event["team"] == "home" else AWAY
            if rng.random() < settings["unknown_team_fraction"]:
                team = UNKNOWN
            if team != UNKNOWN:
                team_delta[start] += 1 if team == HOME else -1
                team_delta[end] -= 1 if team == HOME else -1
                count_delta[start] += 1
                count_delta[end] -= 1
                restart_team[restart:end] = team
            dead_delta[start] += 1
            dead_delta[restart] -= 1
            lookback = max(lower, restart - config["dead_lookback_frames"])
            pre_delta[lookback] += 1
            pre_delta[restart] -= 1
            restart_events.append((start, end, restart, team))
            if event["subtype"] == "KICK OFF":
                restart_type = "kick off (start)" if is_first else "kick off (after goal)"
            elif event["subtype"] == "THROW IN":
                restart_type = "throw-in"
            else:
                restart_type = event["subtype"].lower()
            restart_types.append(restart_type)
    sums = np.cumsum(team_delta[:-1])
    counts = np.cumsum(count_delta[:-1])
    prior = np.zeros(n_frames, dtype=np.float32)
    np.divide(sums, counts, out=prior, where=counts > 0)
    return TagFeatures(
        team_prior=prior,
        dead_hint=np.cumsum(dead_delta[:-1]) > 0,
        pre_restart_window=np.cumsum(pre_delta[:-1]) > 0,
        restart_team=restart_team,
        restart_events=tuple(restart_events), restart_types=tuple(restart_types),
        restart_frames=np.asarray([item[2] for item in restart_events], dtype=np.int32),
        restart_localized=np.zeros(len(restart_events), dtype=bool),
        restart_teams=np.asarray([item[3] for item in restart_events], dtype=np.int8),
        restart_team_inferred=np.zeros(len(restart_events), dtype=bool),
    )


def independent_reference_mask(truth, references, events, periods, fps, exclusion_seconds):
    excluded = np.zeros(len(truth), dtype=bool)
    radius = int(round(exclusion_seconds * fps))
    restart_rows = []
    for event in events:
        if event["type"] != "SET PIECE":
            continue
        row = event["frame"] - 1
        restart_rows.append(row)
        low, high = max(0, row - radius), min(len(truth), row + radius + 1)
        indexes = np.arange(low, high)
        excluded[indexes[periods[indexes] == event["period"]]] = True
    scoring_truth = truth.copy()
    scoring_truth[excluded] = UNKNOWN
    scoring_references = dict(references)
    scoring_references["complete_live_spells"] = [
        spell for spell in references["complete_live_spells"]
        if all(
            abs(spell["start_row"] - row) > radius
            and abs(spell["stop_row"] - row) > radius
            for row in restart_rows
        )
    ]
    return scoring_truth, scoring_references


def live_missing_stress(ball, truth, periods, fps, seconds, maximum_windows, minimum_windows):
    length = max(1, int(round(seconds * fps)))
    selection = ball.copy()
    live = np.isin(truth, [HOME, AWAY])
    selection[~live] = np.nan
    groups = periods.astype(np.int32) * 3 + truth
    starts = _select_windows(selection, length, maximum_windows, groups)
    if len(starts) < minimum_windows:
        raise ValueError("only {} eligible {}s live windows; need {}".format(
            len(starts), seconds, minimum_windows
        ))
    masked = ball.copy()
    for start in starts:
        masked[start:start + length] = np.nan
    return masked, starts, length


def _live_stress_metrics(labels, starts, length):
    if not starts:
        return None
    windows = np.asarray([labels[start:start + length] for start in starts])
    return {
        "windows": len(starts), "masked_frames": len(starts) * length,
        "false_dead_fraction": float(np.mean(windows == DEAD)),
        "windows_with_false_dead_fraction": float(np.mean(np.any(windows == DEAD, axis=1))),
        "split_window_fraction": float(np.mean(np.any(
            windows[:, 1:] != windows[:, :-1], axis=1
        ))),
    }


def _case_checks(game, tag_mode, scenario, models, protocol):
    baseline, candidate = models["baseline"], models["candidate"]
    limits = protocol["gates"]
    checks = []
    for metric, actual, minimum in (
        ("macro_f1", candidate["primary_possession"]["known_frame_macro_f1"],
         baseline["primary_possession"]["known_frame_macro_f1"] - limits["maximum_macro_f1_drop"]),
        ("independent_endpoint_spell_f1",
         candidate["primary_possession"]["reference_conditioned_spell_scores"]["0.5"]["f1"],
         baseline["primary_possession"]["reference_conditioned_spell_scores"]["0.5"]["f1"]
         - limits["maximum_reference_spell_f1_drop"]),
        ("recovery_boundary_recall", candidate["recoveries"]["same_direction_boundary_recall"]["0.5"],
         baseline["recoveries"]["same_direction_boundary_recall"]["0.5"]
         - limits["maximum_recovery_boundary_recall_drop"]),
    ):
        checks.append({
            "name": "{}:{}:{}:{}".format(game, tag_mode, scenario, metric),
            "passed": bool(actual >= minimum), "actual": actual, "minimum": minimum,
        })
    if protocol.get("candidate_scope") in ("gap_release", "core_release"):
        for iou in protocol["spell_ious"]:
            if iou == 0.5:
                continue
            actual = candidate["primary_possession"]["reference_conditioned_spell_scores"][str(iou)]["f1"]
            minimum = (
                baseline["primary_possession"]["reference_conditioned_spell_scores"][str(iou)]["f1"]
                - limits["maximum_reference_spell_f1_drop"]
            )
            checks.append({
                "name": "{}:{}:{}:spell_f1_{}".format(game, tag_mode, scenario, iou),
                "passed": bool(actual >= minimum), "actual": actual, "minimum": minimum,
            })
        actual = candidate["recoveries"]["accuracy_at_recovery"]
        minimum = baseline["recoveries"]["accuracy_at_recovery"] - limits["maximum_recovery_accuracy_drop"]
        checks.append({
            "name": "{}:{}:{}:recovery_accuracy".format(game, tag_mode, scenario),
            "passed": bool(actual >= minimum), "actual": actual, "minimum": minimum,
        })
    elif scenario == "clean" and tag_mode != "none":
        gain = (
            candidate["primary_possession"]["per_state"]["D"]["recall"]
            - baseline["primary_possession"]["per_state"]["D"]["recall"]
        )
        checks.append({
            "name": "{}:{}:incremental_dead_recall".format(game, tag_mode),
            "passed": gain >= limits["minimum_tagged_dead_recall_gain"],
            "actual": gain, "minimum": limits["minimum_tagged_dead_recall_gain"],
        })
    if candidate["live_stress"] is not None:
        for metric, maximum in (
            ("false_dead_fraction", limits["maximum_excess_masked_live_dead_fraction"]),
            ("windows_with_false_dead_fraction", limits["maximum_excess_broken_live_window_fraction"]),
            ("split_window_fraction", limits["maximum_excess_broken_live_window_fraction"]),
        ):
            excess = candidate["live_stress"][metric] - baseline["live_stress"][metric]
            checks.append({
                "name": "{}:{}:{}:excess_{}".format(game, tag_mode, scenario, metric),
                "passed": excess <= maximum, "actual": excess, "maximum": maximum,
            })
    return checks


def gap_release_spell_summary(results, protocol):
    summaries = []
    for game in results:
        values = {"baseline": [], "candidate": []}
        for case in game["cases"]:
            if case["scenario"] != "clean" or case["tag_mode"] == "none":
                continue
            for name in values:
                values[name].extend(
                    case["models"][name]["primary_possession"][
                        "reference_conditioned_spell_scores"
                    ][str(iou)]["f1"]
                    for iou in protocol["spell_ious"]
                )
        if not values["baseline"] or any(
            value is None for items in values.values() for value in items
        ):
            raise ValueError("release comparison requires measurable tagged spell scores")
        baseline = float(np.mean(values["baseline"]))
        candidate = float(np.mean(values["candidate"]))
        summaries.append({
            "game": game["game"], "baseline_mean": baseline,
            "candidate_mean": candidate, "gain": candidate - baseline,
        })
    return summaries


def check_external_transfer(root, protocol_path, output_path):
    root = Path(root).resolve()
    protocol_path = Path(protocol_path)
    protocol, configs = load_frozen_transfer_protocol(protocol_path)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != protocol["external_repository_commit"]:
        raise ValueError("external dataset revision differs from frozen protocol")
    results, checks = [], []
    for role, games in protocol["games"].items():
        for game in games:
            print("Frozen transfer check: {} ({})".format(game, role), flush=True)
            data = _load_game(root, game)
            events = _load_events(data["event_path"], {
                "PASS", "CARRY", "RECOVERY", "BALL LOST", "BALL OUT", "SHOT",
                "SET PIECE", "FAULT RECEIVED",
            })
            truth, references = possession_chain_references(
                events, data["frames"], data["periods"], data["fps"], protocol["references"]
            )
            primary_truth, primary_references = independent_reference_mask(
                truth, references, events, data["periods"], data["fps"],
                protocol["restart_reference_exclusion_seconds"],
            )
            if len(primary_references["complete_live_spells"]) < protocol["minimum_independent_reference_spells"]:
                raise ValueError("insufficient reference spells without shared restart endpoints")
            if any(not np.any(primary_truth == state) for state in (HOME, AWAY, DEAD)):
                raise ValueError("primary transfer references must support H/A/D")
            scaled = {
                name: config_at_fps(config, protocol["source_fps"], data["fps"])
                for name, config in configs.items()
            }
            scenarios = [("clean", data["ball"], [], 0, False)]
            if "recovery_mask_seconds" in protocol:
                ball, starts, length = recovery_masks(
                    data["ball"], data["periods"], references["recovery_references"],
                    data["fps"], protocol["recovery_mask_seconds"],
                )
                if not starts:
                    raise ValueError("no eligible release recovery masks")
                scenarios.append(("recovery_masked", ball, starts, length, False))
            for seconds in protocol["missing_durations_seconds"]:
                ball, starts, length = live_missing_stress(
                    data["ball"], primary_truth, data["periods"], data["fps"], seconds,
                    protocol["maximum_mask_windows"], protocol["minimum_mask_windows"],
                )
                scenarios.append(("missing_{}s".format(seconds), ball, starts, length, True))
            edges = np.r_[0, np.flatnonzero(np.diff(data["periods"])) + 1, len(truth)]
            cases = []
            for scenario, ball, starts, length, is_live_stress in scenarios:
                prepared = []
                for start, stop in zip(edges[:-1], edges[1:]):
                    trajectory = _repair_by_period(ball[start:stop], data["fps"], scaled["baseline"])
                    control = control_features_from_arrays(
                        data["home"][start:stop], data["away"][start:stop],
                        trajectory.xy, data["fps"], scaled["baseline"],
                    )
                    period_events = [
                        {**event, "frame": event["frame"] - int(start)}
                        for event in events if event["period"] == data["periods"][start]
                    ]
                    prepared.append((start, stop, trajectory, control, period_events))
                for tag_mode, settings in protocol["tag_modes"].items():
                    predictions = {"baseline": [], "candidate": []}
                    tag_count = 0
                    for start, stop, trajectory, control, period_events in prepared:
                        tags = restart_only_tags(
                            period_events, data["periods"][start:stop], data["fps"],
                            scaled["baseline"], settings, protocol["seed"] + int(data["periods"][start]),
                        )
                        tag_count += len(tags.restart_events)
                        tags = localize_restarts(tags, trajectory, data["fps"], scaled["baseline"])
                        tags = infer_restart_ownership(tags, trajectory, control, scaled["baseline"])
                        for name, config in scaled.items():
                            predictions[name].append(decode(
                                build_evidence(control, trajectory, tags, (105.0, 68.0), config, fps=data["fps"]),
                                tags, config,
                            ))
                    models = {}
                    for name, pieces in predictions.items():
                        labels = np.concatenate(pieces)
                        models[name] = {
                            "primary_possession": score_possession_references(
                                labels, primary_truth, primary_references, data["periods"]
                            ),
                            "shared_endpoint_provider_proxy": score_possession_references(
                                labels, truth, references, data["periods"]
                            ),
                            "recoveries": score_recovery_transitions(
                                labels, data["periods"], references["recovery_references"],
                                data["fps"], protocol["recovery_scoring"],
                            ),
                            "live_stress": (
                                _live_stress_metrics(labels, starts, length) if is_live_stress else None
                            ),
                            "spell_diagnostics": _period_spell_diagnostics(labels, data["fps"], data["periods"]),
                            "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                        }
                    case_checks = _case_checks(game, tag_mode, scenario, models, protocol)
                    checks.extend(case_checks)
                    cases.append({
                        "scenario": scenario, "tag_mode": tag_mode,
                        "simulated_restart_tags": tag_count, "mask_start_rows": starts,
                        "mask_duration_frames": length, "models": models,
                        "passed": all(check["passed"] for check in case_checks),
                    })
                    print("  {} / {}: passed={}".format(
                        scenario, tag_mode, cases[-1]["passed"]
                    ), flush=True)
            results.append({
                "game": game, "role": role, "fps": data["fps"],
                "input_sha256": {path.name: _file_sha256(path) for path in data["input_paths"]},
                "reference_summary": references["summary"],
                "primary_known_frames": int(np.count_nonzero(primary_truth != UNKNOWN)),
                "primary_complete_spells": len(primary_references["complete_live_spells"]),
                "cases": cases,
            })
            _save_transfer_report(output_path, protocol_path, protocol, results, checks)
    for name in ("baseline", "candidate"):
        project = Path(__file__).resolve().parents[1]
        if _file_sha256(project / protocol[name + "_config"]) != protocol[name + "_sha256"]:
            raise ValueError("frozen config changed during evaluation")
    if protocol.get("candidate_scope") in ("gap_release", "core_release"):
        summaries = gap_release_spell_summary(results, protocol)
        for summary in summaries:
            minimum = protocol["gates"]["minimum_per_game_tagged_clean_spell_gain"]
            checks.append({
                "name": summary["game"] + ":tagged_clean_spell_gain",
                "passed": summary["gain"] >= minimum,
                "actual": summary["gain"], "minimum": minimum,
            })
        gain = float(np.mean([summary["gain"] for summary in summaries]))
        minimum = protocol["gates"]["minimum_mean_tagged_clean_spell_gain"]
        checks.append({
            "name": "release_mean_tagged_clean_spell_gain",
            "passed": gain >= minimum, "actual": gain, "minimum": minimum,
        })
    return _save_transfer_report(output_path, protocol_path, protocol, results, checks, finalized=True)


def _save_transfer_report(path, protocol_path, protocol, results, checks, finalized=False):
    project = Path(__file__).resolve().parents[1]
    complete = finalized and len(results) == sum(len(games) for games in protocol["games"].values())
    report = {
        "schema_version": 1, "external_data": True,
        "protocol_sha256": _file_sha256(protocol_path),
        "baseline_sha256": protocol["baseline_sha256"], "candidate_sha256": protocol["candidate_sha256"],
        "results": results, "checks": checks,
        "passed": complete and all(check["passed"] for check in checks),
        "completed_games": [result["game"] for result in results],
        "complete": complete,
        "gap_release_spell_summary": (
            gap_release_spell_summary(results, protocol)
            if protocol.get("candidate_scope") in ("gap_release", "core_release") else None
        ),
        "code_sha256": {
            source.name: _file_sha256(source) for source in (
                Path(__file__), project / "possession" / "external_metrica_validation.py",
                project / "possession" / "external_possession_events.py",
                project / "possession" / "decoder.py", project / "possession" / "tagging.py",
                project / "possession" / "tracking.py", project / "possession" / "trajectory.py",
            )
        },
        "limitations": [
            "Provider-derived references and synthetic restart tags, not independent automatic tagging truth.",
            "Primary frame scores exclude restart neighborhoods; primary spells exclude shared restart endpoints.",
            "Tag inputs use only SET PIECE records; full reference labels and loss/recovery events are not input features.",
            "Missingness masks are chosen from external known-live references, not challenge outcomes.",
            "Two games; no inference of challenge spell F1 or generalization confidence interval.",
            (
                "Games 2-3 are reused diagnostics, not an independent or fresh holdout."
                if protocol.get("candidate_scope") in ("gap_release", "core_release") else
                "Game 3 is now consumed by this frozen test and must not be reused for tuning."
            ),
            "No candidate parameters are changed or automatically promoted.",
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check frozen stoppage transfer and missing-ball safety.")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--protocol", default=str(Path(__file__).resolve().parents[1] / "configs" / "frozen_stoppage_transfer_check.json")
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        report = check_external_transfer(args.root, args.protocol, args.output)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, "error: {}\n".format(error))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
