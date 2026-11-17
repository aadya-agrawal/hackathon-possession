import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from possession import AWAY, DEAD, HOME, UNKNOWN
from games import GAME, GAMES, require
from evaluation.possession_metrics import match_spells, spell_overlaps
from possession.decoder import build_evidence, decode, _semi_markov_strong_transition_support
from possession.external_guarded_calibration import _evaluate, _prepare, _save, guarded_checks
from possession.external_metrica_validation import (
    PITCH_LENGTH, PITCH_WIDTH, _file_sha256, _load_game, config_at_fps,
)
from possession.external_possession_events import (
    _match_boundaries, reference_spell_inputs, score_possession_references,
)
from possession.external_reliability_calibration import load_reliability_protocol
from possession.tracking import (
    _nearest_player_velocity, _parse_group, contact_motion_features,
    load_tracking_metadata, player_velocity_steps,
)


def matched_spell_pairs(reference, predictions, threshold):
    overlaps = spell_overlaps(reference, predictions)
    used_reference, used_prediction, pairs = set(), set(), []
    for iou, ref_index, pred_index in overlaps:
        if iou < threshold:
            break
        if ref_index in used_reference or pred_index in used_prediction:
            continue
        used_reference.add(ref_index)
        used_prediction.add(pred_index)
        pairs.append((ref_index, pred_index, iou))
    if len(pairs) != match_spells(overlaps, len(reference), len(predictions), threshold)[0]:
        raise ValueError("localization matching differs from the reference scorer")
    return pairs


def spurious_lineage(reference, baseline, candidate, metric_iou, lineage_iou):
    baseline_matched = {item[1] for item in matched_spell_pairs(reference, baseline, metric_iou)}
    candidate_matched = {item[1] for item in matched_spell_pairs(reference, candidate, metric_iou)}
    baseline_spurious = set(range(len(baseline))) - baseline_matched
    candidate_spurious = set(range(len(candidate))) - candidate_matched
    counterparts = {
        item[1]: item[0] for item in matched_spell_pairs(baseline, candidate, lineage_iou)
    }
    details = []
    shared = 0
    for index in sorted(candidate_spurious):
        previous = counterparts.get(index)
        if previous in baseline_spurious:
            kind = "shared_spurious"
            shared += 1
        elif previous is not None:
            kind = "previously_matched_now_spurious"
        else:
            kind = "new_prediction"
        start, stop, state = candidate[index]
        details.append({
            "prediction_index": index, "start_row": int(start), "stop_row": int(stop),
            "state": int(state), "lineage": kind, "baseline_prediction_index": previous,
        })
    added = len(candidate_spurious) - shared
    removed = len(baseline_spurious) - shared
    return {
        "baseline_spurious": len(baseline_spurious),
        "candidate_spurious": len(candidate_spurious),
        "shared_spurious": shared,
        "introduced_or_degraded": added,
        "removed_or_fixed": removed,
        "net_increase": added - removed,
        "details": details,
    }


def _previous_offset(row, events, periods, fps):
    selected = [int(event) for event in events if event <= row and periods[event] == periods[row]]
    return (row - max(selected)) / fps if selected else None


def _nearest_offset(row, events, periods, fps):
    selected = [int(event) for event in events if periods[event] == periods[row]]
    if not selected:
        return None
    nearest = min(selected, key=lambda event: (abs(row - event), event))
    return (row - nearest) / fps


def annotate_spurious(lineage, source, periods, truth, reappearances, mask_ends, tag_boundaries, fps):
    details = []
    for item in lineage["details"]:
        row = item["start_row"]
        details.append({
            **item, "duration_seconds": (item["stop_row"] - row) / fps,
            "period": int(periods[row]), "start_source": int(source[row]),
            "reference_state_at_start": int(truth[row]),
            "seconds_after_last_reappearance": _previous_offset(row, reappearances, periods, fps),
            "signed_seconds_from_mask_end": _nearest_offset(row, mask_ends, periods, fps),
            "signed_seconds_from_tag_boundary": _nearest_offset(row, tag_boundaries, periods, fps),
        })
    return details


def localization_summary(details, protocol):
    introduced = [item for item in details if item["lineage"] != "shared_spurious"]
    groups = {}
    for name, items in (("all_spurious", details), ("introduced_or_degraded", introduced)):
        counts = {
            "events": len(items),
            "starts_in_interpolation": sum(item["start_source"] == 2 for item in items),
            "starts_unobserved_or_predicted": sum(item["start_source"] in (0, 3) for item in items),
            "starts_observed": sum(item["start_source"] == 1 for item in items),
            "correct_reference_team_at_start": sum(
                item["reference_state_at_start"] == item["state"] for item in items
            ),
            "wrong_reference_team_at_start": sum(
                item["reference_state_at_start"] != UNKNOWN
                and item["reference_state_at_start"] != item["state"] for item in items
            ),
            "unknown_reference_at_start": sum(
                item["reference_state_at_start"] == UNKNOWN for item in items
            ),
        }
        for seconds in protocol["reappearance_bins_seconds"]:
            counts["post_reappearance_{}s".format(seconds)] = sum(
                item["seconds_after_last_reappearance"] is not None
                and 0 <= item["seconds_after_last_reappearance"] <= seconds
                for item in items
            )
        for seconds in protocol["tag_bins_seconds"]:
            counts["near_tag_boundary_{}s".format(seconds)] = sum(
                item["signed_seconds_from_tag_boundary"] is not None
                and abs(item["signed_seconds_from_tag_boundary"]) <= seconds
                for item in items
            )
        groups[name] = counts
    return groups


def control_cue_arrays(home, away, trajectory, control, evidence, config, fps, window):
    n = len(trajectory.source)
    if n < 1 or fps <= 0 or window < 1:
        raise ValueError("control diagnosis requires frames, positive fps, and a positive window")
    rows = np.arange(n)
    nearest = control.nearest_distances[:, :, 0]
    observed = trajectory.source == 1
    ball_velocity = np.full((n, 2), np.nan, dtype=np.float32)
    steps = np.diff(trajectory.xy, axis=0) * fps
    valid_steps = observed[1:] & observed[:-1] & np.isfinite(steps).all(axis=1)
    ball_velocity[1:][valid_steps] = steps[valid_steps]
    relative_speed = np.full((n, 2), np.nan, dtype=np.float32)
    proxy_error = np.full((n, 2), np.nan, dtype=np.float32)
    legacy_velocity = _nearest_player_velocity(
        control.nearest_player_xy, fps, config["max_player_speed_metres_per_second"]
    )
    identity_switch = np.full((n, 2), np.nan, dtype=np.float32)
    contact = np.zeros((n, 2), dtype=np.float32)
    team_contact = np.zeros((n, 2), dtype=np.float32)
    available = np.zeros(n, dtype=np.int32)
    radius = config["strong_control_distance_metres"]
    for lag in range(min(window, n)):
        available[:n - lag] += observed[lag:]
    for team, players in enumerate((home, away)):
        if (players.ndim != 3 or players.shape[0] != n
                or players.shape[1] < 1 or players.shape[2] != 2):
            raise ValueError("control diagnosis requires aligned player tracks")
        distances = np.linalg.norm(players - trajectory.xy[:, None, :], axis=2)
        distances[~np.isfinite(distances)] = np.inf
        index = np.argsort(distances, axis=1)[:, 0]
        velocity = np.full((n, 2), np.nan, dtype=np.float32)
        # Follow the current player's track, not the moving nearest-player identity.
        player_steps = (players[rows[1:], index[1:]] - players[rows[:-1], index[1:]]) * fps
        valid = (
            np.isfinite(player_steps).all(axis=1)
            & np.isfinite(nearest[1:, team])
            & (np.linalg.norm(player_steps, axis=1) <= config["max_player_speed_metres_per_second"])
        )
        velocity[1:][valid] = player_steps[valid]
        relative_speed[:, team] = np.linalg.norm(ball_velocity - velocity, axis=1)
        proxy_error[:, team] = np.linalg.norm(
            legacy_velocity[:, team] - velocity, axis=1
        )
        visible = np.isfinite(nearest[1:, team]) & np.isfinite(nearest[:-1, team])
        identity_switch[1:, team] = np.where(visible, index[1:] != index[:-1], np.nan)
        for lag in range(min(window, n)):
            stop = n - lag
            future_distance = np.linalg.norm(
                players[rows[lag:], index[:stop]] - trajectory.xy[lag:], axis=1
            )
            contact[:stop, team] += observed[lag:] & (future_distance <= radius)
            team_contact[:stop, team] += observed[lag:] & (nearest[lag:, team] <= radius)
    fractions = {}
    for name, values in (("same_player_contact_fraction", contact),
                         ("team_contact_fraction", team_contact)):
        fraction = np.full((n, 2), np.nan, dtype=np.float32)
        np.divide(values, available[:, None], out=fraction, where=available[:, None] > 0)
        fraction[~observed] = np.nan
        fractions[name] = fraction
    advantage = evidence.control > evidence.control[:, ::-1]
    dominance_count = _future_strong_counts(observed[:, None] & advantage, window)
    dominance = np.full((n, 2), np.nan, dtype=np.float32)
    np.divide(dominance_count, available[:, None], out=dominance, where=available[:, None] > 0)
    dominance[~observed] = np.nan
    motion = contact_motion_features(
        trajectory.xy, trajectory.source, nearest, fps, radius, window,
        config["semi_markov_strong_transition_minimum_frames"],
        config["max_ball_speed_metres_per_second"],
    )
    margin = np.full((n, 2), np.nan, dtype=np.float32)
    np.subtract(nearest[:, ::-1], nearest, out=margin,
                where=np.isfinite(nearest).all(axis=1)[:, None])
    strong_now = evidence.strong_control.astype(np.float32)
    opposing_strong_now = evidence.strong_control[:, ::-1].astype(np.float32)
    strong_future = _semi_markov_strong_transition_support(evidence, config).astype(np.float32)
    future_without_current = strong_future * (1 - strong_now)
    for values in (strong_now, opposing_strong_now, strong_future, future_without_current):
        values[~observed] = np.nan
    fast_proximity = np.full((n, 2), np.nan, dtype=np.float32)
    valid_motion = observed[:, None] & np.isfinite(relative_speed)
    fast_proximity[valid_motion] = (
        (nearest <= radius)
        & (relative_speed > config["strong_control_relative_speed_metres_per_second"])
    )[valid_motion]
    minimum = config["semi_markov_strong_transition_minimum_frames"]
    low_contact = np.full((n, 2), np.nan, dtype=np.float32)
    supported_window = observed & (available >= minimum)
    low_contact[supported_window] = (
        fractions["same_player_contact_fraction"][supported_window] < minimum
        / config["semi_markov_strong_transition_window_frames"]
    )
    return {
        "nearest_distance_metres": np.where(observed[:, None], nearest, np.nan),
        "distance_margin_metres": np.where(observed[:, None], margin, np.nan),
        "relative_speed_mps": relative_speed,
        "ball_speed_mps": np.repeat(np.linalg.norm(ball_velocity, axis=1)[:, None], 2, axis=1),
        "velocity_proxy_error_mps": proxy_error,
        "nearest_identity_switch": identity_switch,
        "strong_current": strong_now, "strong_future": strong_future,
        "opposing_strong_current": opposing_strong_now,
        "future_without_current_contact": future_without_current,
        "future_team_advantage_fraction": dominance,
        **motion,
        "fast_proximity": fast_proximity, "low_same_player_contact": low_contact,
        **fractions,
    }


def cue_group_summary(items, arrays):
    summary = {"events": len(items), "metrics": {}}
    for name, array in arrays.items():
        values = np.asarray(
            [array[item["start_row"], item["state"]] for item in items], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        summary["metrics"][name] = {
            "available": int(len(finite)),
            "mean": float(np.mean(finite)) if len(finite) else None,
            "quantiles": dict(zip(("q25", "median", "q75"),
                                  np.quantile(finite, [0.25, 0.5, 0.75]).tolist())) if len(finite) else None,
        }
    return summary


def cue_rank_contrasts(errors, matched, arrays):
    contrasts = {}
    for name, array in arrays.items():
        bad = np.asarray([array[item["start_row"], item["state"]] for item in errors])
        good = np.asarray([array[item["start_row"], item["state"]] for item in matched])
        bad, good = bad[np.isfinite(bad)], good[np.isfinite(good)]
        contrasts[name] = {
            "error_available": len(bad), "matched_available": len(good),
            "probability_error_greater": float(np.mean(
                (bad[:, None] > good[None, :]) + 0.5 * (bad[:, None] == good[None, :])
            )) if len(bad) and len(good) else None,
        }
    return contrasts


def control_start_diagnosis(reference, predictions, labels, details, arrays, source, truth, periods, fps, iou):
    matched = []
    for ref_index, pred_index, overlap in matched_spell_pairs(reference, predictions, iou):
        start, stop, state = predictions[pred_index]
        matched.append({
            "prediction_index": pred_index, "start_row": int(start), "stop_row": int(stop),
            "state": int(state), "reference_index": ref_index, "matched_iou": overlap,
            "boundary_offset_seconds": (start - reference[ref_index][0]) / fps,
        })
    all_items = matched + details
    for item in all_items:
        row = item["start_row"]
        item["start_source"] = int(source[row])
        item["reference_state_at_start"] = int(truth[row])
        item["period"] = int(periods[row])
        item["previous_predicted_state"] = (
            int(labels[row - 1]) if row > 0 and periods[row - 1] == periods[row] else UNKNOWN
        )
        item["cues"] = {
            name: float(values[row, item["state"]]) if np.isfinite(values[row, item["state"]]) else None
            for name, values in arrays.items()
        }
    introduced = [item for item in details if item["lineage"] != "shared_spurious"]
    wrong = [item for item in introduced if item["reference_state_at_start"] in (HOME, AWAY)
             and item["reference_state_at_start"] != item["state"]]
    low_matched = {pair[1] for pair in matched_spell_pairs(reference, predictions, .5)}
    high_matched = {item["prediction_index"] for item in matched}
    partial = low_matched - high_matched
    for item in details:
        item["matched_at_iou_0.5"] = item["prediction_index"] in low_matched
    groups = {
        "matched": matched, "all_spurious": details, "introduced_or_degraded": introduced,
        "introduced_wrong_live_team": wrong,
        "introduced_wrong_live_unmatched_0.5": [
            item for item in wrong if item["prediction_index"] not in low_matched
        ],
        "introduced_partial_reference_spell": [
            item for item in introduced if item["prediction_index"] in partial
        ],
        "introduced_during_reference_dead": [item for item in introduced
                                            if item["reference_state_at_start"] not in (HOME, AWAY, UNKNOWN)],
    }
    handovers = {
        name: [item for item in items if item["start_source"] == 1
               and item["previous_predicted_state"] in (HOME, AWAY)
               and item["previous_predicted_state"] != item["state"]]
        for name, items in groups.items()
    }
    return {
        "groups": {name: cue_group_summary(items, arrays) for name, items in groups.items()},
        "observed_handover_groups": {name: cue_group_summary(items, arrays) for name, items in handovers.items()},
        "observed_handover_contrasts": {
            name: cue_rank_contrasts(handovers[name], handovers["matched"], arrays)
            for name in ("introduced_or_degraded", "introduced_wrong_live_team",
                         "introduced_wrong_live_unmatched_0.5")
        },
        "by_period": {
            str(period): {
                name: cue_group_summary([item for item in items if item["period"] == period], arrays)
                for name, items in handovers.items()
            } for period in np.unique(periods)
        },
        "matched_starts": matched,
    }


def localize_external_spells(root, protocol_path, output, control_cues=False):
    protocol_path = Path(protocol_path)
    localization = json.loads(protocol_path.read_text())
    if localization["development_game"] != "Sample_Game_1":
        raise ValueError("localization is restricted to Game 1")
    project = Path(__file__).resolve().parents[2]
    extension, protocol, baseline, previous, source_candidate = load_reliability_protocol(
        project / localization["source_protocol"]
    )
    prepared = _prepare(Path(root).resolve(), protocol, baseline)
    data = prepared["data"]
    if {path.name: _file_sha256(path) for path in data["input_paths"]} != previous["input_sha256"]:
        raise ValueError("localization inputs differ from the original development study")
    edges = np.r_[0, np.flatnonzero(np.diff(data["periods"])) + 1, len(data["periods"])]
    results = []
    for name in localization["cases"]:
        case = next(item for item in prepared["cases"] if item["name"] == name)
        source = np.concatenate([trajectory.source for trajectory, _, _ in case["periods"]])
        reappearances = np.flatnonzero(
            (source[1:] == 1) & (source[:-1] != 1)
            & (data["periods"][1:] == data["periods"][:-1])
        ) + 1
        tag_boundaries = []
        for start, (trajectory, control, tags) in zip(edges[:-1], case["periods"]):
            tag_boundaries.extend(
                int(start + row) for event in tags.restart_events for row in event[:2]
                if start + row < len(source)
            )
            tag_boundaries.extend(
                int(start + row) for row in tags.restart_frames if start + row < len(source)
            )
        tag_boundaries = np.asarray(sorted(set(tag_boundaries)), dtype=np.int64)
        mask_ends = np.asarray(case["mask_start_rows"], dtype=np.int64) + case["mask_length_frames"]
        models = {}
        for model, config in (("baseline", baseline), ("candidate", {**baseline, **source_candidate["values"]})):
            scaled = config_at_fps(config, protocol["source_fps"], data["fps"])
            period_labels, period_cues = [], []
            for start, stop, (trajectory, control, tags) in zip(edges[:-1], edges[1:], case["periods"]):
                evidence = build_evidence(control, trajectory, tags, (105, 68), scaled, fps=data["fps"])
                period_labels.append(decode(evidence, tags, scaled))
                if control_cues:
                    period_cues.append(control_cue_arrays(
                        data["home"][start:stop], data["away"][start:stop],
                        trajectory, control, evidence, scaled, data["fps"],
                        scaled["semi_markov_strong_transition_window_frames"],
                    ))
            labels = np.concatenate(period_labels)
            expected = (
                previous["experiments"][0]["evaluation"] if model == "baseline"
                else source_candidate["evaluation"]
            )
            label_hash = hashlib.sha256(labels.tobytes()).hexdigest()
            if label_hash != expected["cases"][name]["label_sha256"]:
                raise ValueError("localization replay differs from original model output")
            reference, predictions = reference_spell_inputs(
                labels, prepared["primary_references"], data["periods"]
            )
            models[model] = {
                "predictions": predictions, "label_sha256": label_hash,
                "score": score_possession_references(
                    labels, prepared["truth"], prepared["primary_references"], data["periods"]
                ),
            }
            if control_cues:
                models[model]["labels"] = labels
                models[model]["cue_arrays"] = {
                    key: np.concatenate([cues[key] for cues in period_cues])
                    for key in period_cues[0]
                }
        lineage = spurious_lineage(
            reference, models["baseline"]["predictions"], models["candidate"]["predictions"],
            localization["metric_iou"], localization["lineage_iou"],
        )
        details = annotate_spurious(
            lineage, source, data["periods"], prepared["truth"], reappearances,
            mask_ends, tag_boundaries, data["fps"],
        )
        baseline_matched = {
            pair[1] for pair in matched_spell_pairs(
                reference, models["baseline"]["predictions"], localization["metric_iou"]
            )
        }
        baseline_details = {"details": [
            {"start_row": int(start), "stop_row": int(stop), "state": int(state),
             "lineage": "shared_spurious", "prediction_index": index}
            for index, (start, stop, state) in enumerate(models["baseline"]["predictions"])
            if index not in baseline_matched
        ]}
        baseline_annotated = annotate_spurious(
            baseline_details, source, data["periods"], prepared["truth"], reappearances,
            mask_ends, tag_boundaries, data["fps"],
        )
        diagnoses = {}
        if control_cues:
            for model, annotated in (("baseline", baseline_annotated), ("candidate", details)):
                diagnoses[model] = control_start_diagnosis(
                    reference, models[model]["predictions"], models[model]["labels"],
                    annotated, models[model]["cue_arrays"], source, prepared["truth"],
                    data["periods"], data["fps"], localization["metric_iou"],
                )
        results.append({
            "case": name, "lineage": {key: value for key, value in lineage.items() if key != "details"},
            "summary": localization_summary(details, localization), "spurious_intervals": details,
            "baseline_spurious_summary": localization_summary(baseline_annotated, localization)["all_spurious"],
            "mask_end_rows": mask_ends.tolist(), "reappearance_rows": reappearances.tolist(),
            "tag_boundary_rows": tag_boundaries.tolist(),
            "models": {
                model: {key: value for key, value in item.items()
                        if key not in ("predictions", "labels", "cue_arrays")}
                for model, item in models.items()
            },
            **({"control_diagnosis": diagnoses} if control_cues else {}),
        })
        print("{}: net spurious increase {}; introduced/degraded {}".format(
            name, lineage["net_increase"], lineage["introduced_or_degraded"]
        ), flush=True)
    primary = next(result for result in results if result["case"] == localization["primary_case"])
    counts = primary["summary"]["introduced_or_degraded"]
    n = counts["events"]
    fractions = {
        "post_reappearance_1s": counts["post_reappearance_1.0s"] / n if n else 0,
        "interpolation": counts["starts_in_interpolation"] / n if n else 0,
        "near_tag_boundary_3s": counts["near_tag_boundary_3.0s"] / n if n else 0,
    }
    supported = [key for key, value in fractions.items()
                 if value >= localization["repair_evidence_minimum_fraction"]]
    report = {
        "schema_version": 1, "development_game": "Sample_Game_1",
        "excluded_games": ["Sample_Game_2", "Sample_Game_3"],
        "candidate": extension["source_candidate"], "protocol_sha256": _file_sha256(protocol_path),
        "input_sha256": {path.name: _file_sha256(path) for path in data["input_paths"]},
        "cases": results, "repair_evidence_fractions": fractions, "supported_hypotheses": supported,
        "targeted_rule_justified_by_localization": bool(supported),
        "unsupported_rule_added": False,
        "scoring_definition_changed": False, "inference_parameters_changed": False,
        "original_outputs_reproduced": True,
        "control_cues_measured": control_cues,
        **({"control_cue_definitions": {
            "ball_velocity": "Backward differences of consecutive observed ball points, within period.",
            "player_velocity": "Backward differences of the current nearest player's same track; speed-capped at baseline maximum.",
            "contact": "Observed ball within the model's existing strong-control distance; future window and minimum fraction from its unchanged support settings.",
            "fast_proximity": "Within existing strong-control distance and above existing inactive relative-speed gate (2 m/s); not a selected threshold.",
            "comparison": "Observed decoded H/A handovers; provider-IoU matched starts versus introduced/degraded starts, also split by period.",
            "rank_probability": "P(error cue > matched cue), half weight for ties, finite values only. Descriptive within Game 1, not held-out evidence.",
        }} if control_cues else {}),
        "code_sha256": {
            path.name: _file_sha256(path) for path in (
                Path(__file__), project / "possession" / "external_possession_events.py",
                project / "possession" / "decoder.py",
                project / "possession" / "external_guarded_calibration.py",
                project / "possession" / "tracking.py",
            )
        },
        "limitations": [
            "Provider-reference spurious spells, not challenge ground truth.",
            "Introduced/degraded predictions use one-to-one same-team IoU lineage; net count is not an interval identity.",
            "Cue neighborhoods overlap; localization is association, not causal proof.",
            "Only Game 1 is inspected; no inference parameters or acceptance thresholds are changed.",
            "High relative ball speed is a pass-flight hypothesis, not independent proof of a pass or absent control.",
            "Nearest-player position differences can switch identity; diagnostic velocities follow stable tracks without changing production inference.",
        ],
    }
    _save(output, report)
    print("Supported localization hypotheses: {}".format(supported), flush=True)
    return report


def weak_transition_restore_values(baseline, source):
    key = "semi_markov_live_transition_cost"
    if source[key] >= baseline[key]:
        raise ValueError("weak-transition restore requires a reduced source switch cost")
    return {**source, key: baseline[key]}


def _future_strong_counts(strong, window):
    cumulative = np.vstack((np.zeros((1, 2), dtype=np.int32),
                            np.cumsum(strong, axis=0, dtype=np.int32)))
    starts = np.arange(len(strong))
    return cumulative[np.minimum(starts + window, len(strong))] - cumulative[starts]


def recovery_regressions(labels, cues, source, periods, references, configs, fps):
    boundaries = {
        name: _match_boundaries(values, periods, references, int(round(2 * fps)))
        for name, values in labels.items()
    }
    details = []
    for index, reference in enumerate(references):
        row, team = reference["row"], reference["incoming"]
        baseline_correct = labels["baseline"][row] == team
        candidate_correct = labels["candidate"][row] == team
        group = (
            "regression" if baseline_correct and not candidate_correct else
            "fix" if candidate_correct and not baseline_correct else
            "shared_correct" if candidate_correct else "shared_wrong"
        )
        item = {
            "reference_index": index, "row": row, "period": reference["period"],
            "incoming": team, "outgoing": reference["outgoing"], "group": group,
            "source": int(source[row]),
            "reference_support_seconds": reference["reference_support_seconds"],
            "models": {},
        }
        for name, values in labels.items():
            boundary = boundaries[name].get(index)
            support = bool(cues[name]["strong_future"][row, team])
            model = {
                "label_at_recovery": int(values[row]), "strong_support": support,
                "effective_switch_cost": configs[name][
                    "semi_markov_strong_transition_cost" if support else "semi_markov_live_transition_cost"
                ],
                "boundary_offset_seconds": (boundary - row) / fps if boundary is not None else None,
                "boundary_within_reference_support": (
                    boundary is not None and row <= boundary < reference["support_end_row"]
                ),
                "incoming_support_long_count": int(cues[name]["long_strong_count"][row, team]),
                "incoming_control_margin": float(cues[name]["control_margin"][row, team]),
                "nearest_distance_metres": (
                    float(cues[name]["nearest_distance_metres"][row, team])
                    if np.isfinite(cues[name]["nearest_distance_metres"][row, team]) else None
                ),
                "post_recovery": {},
            }
            for seconds in (0.2, 0.5, 1.0, 2.0):
                target = row + int(round(seconds * fps))
                model["post_recovery"][str(seconds)] = (
                    bool(values[target] == team)
                    if target < len(values) and target < reference["support_end_row"]
                    and periods[target] == periods[row] else None
                )
            item["models"][name] = model
        base, candidate = item["models"]["baseline"], item["models"]["candidate"]
        item["support_cost_inversion"] = (
            base["strong_support"] and not candidate["strong_support"]
            and base["effective_switch_cost"] < candidate["effective_switch_cost"]
        )
        item["long_window_would_enable_candidate_support"] = (
            not candidate["strong_support"]
            and candidate["incoming_support_long_count"]
            >= configs["candidate"]["semi_markov_strong_transition_minimum_frames"]
        )
        item["long_window_at_baseline_fraction_would_enable"] = (
            not candidate["strong_support"]
            and candidate["incoming_support_long_count"]
            >= configs["baseline"]["semi_markov_strong_transition_minimum_frames"]
        )
        item["candidate_delayed_handover"] = (
            not candidate_correct and candidate["boundary_offset_seconds"] is not None
            and candidate["boundary_offset_seconds"] > 0
            and candidate["boundary_within_reference_support"]
        )
        details.append(item)
    groups = {}
    for name in ("regression", "fix", "shared_correct", "shared_wrong"):
        items = [item for item in details if item["group"] == name]
        groups[name] = {
            "events": len(items),
            "interpolated": sum(item["source"] == 2 for item in items),
            "support_cost_inversion": sum(item["support_cost_inversion"] for item in items),
            "long_window_would_enable_support": sum(
                item["long_window_would_enable_candidate_support"] for item in items
            ),
            "long_window_at_baseline_fraction": sum(
                item["long_window_at_baseline_fraction_would_enable"] for item in items
            ),
            "candidate_delayed_handover": sum(item["candidate_delayed_handover"] for item in items),
            "candidate_dead_at_recovery": sum(
                item["models"]["candidate"]["label_at_recovery"] == DEAD for item in items
            ),
            "short_reference_support": sum(item["reference_support_seconds"] < 1 for item in items),
            "candidate_correct_at_0.5s": sum(
                item["models"]["candidate"]["post_recovery"]["0.5"] is True for item in items
            ),
            "scorable_at_0.5s": sum(
                item["models"]["candidate"]["post_recovery"]["0.5"] is not None for item in items
            ),
        }
    return {"groups": groups, "details": details}


def high_iou_regressions(reference, baseline, candidate, source, fps):
    base_high = {ref: (pred, iou) for ref, pred, iou in matched_spell_pairs(reference, baseline, 0.75)}
    candidate_high = {ref: (pred, iou) for ref, pred, iou in matched_spell_pairs(reference, candidate, 0.75)}
    candidate_low = {ref: (pred, iou) for ref, pred, iou in matched_spell_pairs(reference, candidate, 0.5)}
    losses = []
    for index in sorted(set(base_high) - set(candidate_high)):
        start, stop, state = map(int, reference[index])
        base_index, base_iou = base_high[index]
        candidate_match = candidate_low.get(index)
        item = {
            "reference_index": index, "reference_start": start, "reference_stop": stop,
            "state": state, "baseline_prediction": [int(value) for value in baseline[base_index]],
            "baseline_iou": float(base_iou), "candidate_still_matched_at_0.5": candidate_match is not None,
        }
        if candidate_match is not None:
            pred_index, iou = candidate_match
            pred_start, pred_stop, _ = map(int, candidate[pred_index])
            item.update({
                "candidate_prediction": [int(value) for value in candidate[pred_index]], "candidate_iou": float(iou),
                "start_error_seconds": (pred_start - start) / fps,
                "stop_error_seconds": (pred_stop - stop) / fps,
                "start_source": int(source[pred_start]),
                "stop_source": int(source[min(pred_stop, len(source) - 1)]),
            })
        losses.append(item)
    partial = [item for item in losses if item["candidate_still_matched_at_0.5"]]
    return {
        "baseline_matches": len(base_high), "candidate_matches": len(candidate_high),
        "lost": len(losses), "gained": len(set(candidate_high) - set(base_high)),
        "lost_but_still_matched_at_0.5": len(partial),
        "late_starts": sum(item["start_error_seconds"] > 0 for item in partial),
        "early_starts": sum(item["start_error_seconds"] < 0 for item in partial),
        "early_ends": sum(item["stop_error_seconds"] < 0 for item in partial),
        "late_ends": sum(item["stop_error_seconds"] > 0 for item in partial),
        "details": losses,
    }


def diagnose_rebased_regressions(root, rebase_path, output, control_cues=False,
                                game="Sample_Game_1"):
    from possession.external_temporal_parity import temporal_settings

    if game not in ("Sample_Game_1", "Sample_Game_2"):
        raise ValueError("regression diagnosis is limited to previously inspected Games 1-2")
    project = Path(__file__).resolve().parents[2]
    rebase = json.loads(Path(rebase_path).read_text())
    protocol_path = project / "configs" / "metrica_temporal_parity.json"
    if _file_sha256(protocol_path) != rebase["protocol_sha256"]:
        raise ValueError("regression diagnosis requires the matching corrected-clock protocol")
    extension = json.loads(protocol_path.read_text())
    if rebase["runtime_source_fps"] != extension["deployment_source_fps"]:
        raise ValueError("regression diagnosis must use the deployment time policy")
    scenario_path = project / extension["scenario_protocol"]
    if _file_sha256(scenario_path) != extension["scenario_protocol_sha256"]:
        raise ValueError("regression-diagnosis scenario protocol changed")
    protocol = {**json.loads(scenario_path.read_text()), "source_fps": rebase["runtime_source_fps"],
                "development_game": game}
    baseline_path = project / extension["baseline_config"]
    if _file_sha256(baseline_path) != rebase["baseline_sha256"]:
        raise ValueError("regression-diagnosis baseline changed")
    baseline = json.loads(baseline_path.read_text())
    prepared = _prepare(Path(root).resolve(), protocol, baseline)
    if game == "Sample_Game_2" and not rebase.get("failed_development_diagnostic_allowed", False):
        raise ValueError("Game2 localization requires an explicitly authorized frozen diagnostic")
    expected = next(item for item in rebase["results"] if item["game"] == game)
    data = prepared["data"]
    if {path.name: _file_sha256(path) for path in data["input_paths"]} != expected["input_sha256"]:
        raise ValueError("regression diagnosis inputs differ from the frozen rebase")
    configs = {
        "baseline": config_at_fps(baseline, protocol["source_fps"], data["fps"]),
        "candidate": config_at_fps({**baseline, **rebase["values"]}, protocol["source_fps"], data["fps"]),
    }
    results = []
    edges = np.r_[0, np.flatnonzero(np.diff(data["periods"])) + 1, len(data["periods"])]
    for name in ("clean:none", "clean:degraded_restart_tags",
                 "recovery_masked:none", "recovery_masked:degraded_restart_tags"):
        case = next(item for item in prepared["cases"] if item["name"] == name)
        models = {}
        for model, config in configs.items():
            labels, cues = [], []
            for start, stop, (trajectory, control, tags) in zip(edges[:-1], edges[1:], case["periods"]):
                evidence = build_evidence(control, trajectory, tags, (105, 68), config, fps=data["fps"])
                labels.append(decode(evidence, tags, config))
                window = configs["baseline"]["semi_markov_strong_transition_window_frames"]
                values = {
                    "strong_future": _semi_markov_strong_transition_support(evidence, config),
                    "long_strong_count": _future_strong_counts(evidence.strong_control, window),
                    "short_strong_count": _future_strong_counts(
                        evidence.strong_control, config["semi_markov_strong_transition_window_frames"]
                    ),
                    "control_margin": evidence.control - evidence.control[:, ::-1],
                    "nearest_distance_metres": control.nearest_distances[:, :, 0],
                }
                if control_cues:
                    full_cues = control_cue_arrays(
                        data["home"][start:stop], data["away"][start:stop],
                        trajectory, control, evidence, config, data["fps"], window,
                    )
                    observed = trajectory.source == 1
                    full_cues["positive_control_advantage"] = np.where(
                        observed[:, None], values["control_margin"] > 0, np.nan
                    )
                    full_cues["positive_distance_advantage"] = np.where(
                        observed[:, None] & np.isfinite(full_cues["distance_margin_metres"]),
                        full_cues["distance_margin_metres"] > 0, np.nan
                    )
                    values = {**full_cues, **values}
                cues.append(values)
            labels = np.concatenate(labels)
            reference_evaluation = (
                expected["deployment"] if model == "baseline" else expected["candidate"]["evaluation"]
            )
            if hashlib.sha256(labels.tobytes()).hexdigest() != reference_evaluation["cases"][name]["label_sha256"]:
                raise ValueError("regression-diagnosis model replay differs")
            models[model] = {
                "labels": labels,
                "cues": {key: np.concatenate([item[key] for item in cues]) for key in cues[0]},
            }
        source = np.concatenate([trajectory.source for trajectory, _, _ in case["periods"]])
        reference, base_predictions = reference_spell_inputs(
            models["baseline"]["labels"], prepared["primary_references"], data["periods"]
        )
        _, candidate_predictions = reference_spell_inputs(
            models["candidate"]["labels"], prepared["primary_references"], data["periods"]
        )
        recoveries = recovery_regressions(
            {key: item["labels"] for key, item in models.items()},
            {key: item["cues"] for key, item in models.items()},
            source, data["periods"], prepared["references"]["recovery_references"], configs, data["fps"],
        )
        high_iou = high_iou_regressions(reference, base_predictions, candidate_predictions, source, data["fps"])
        lineage = spurious_lineage(reference, base_predictions, candidate_predictions, .75, .5)
        for interval in lineage["details"]:
            row, state = interval["start_row"], interval["state"]
            cues = models["candidate"]["cues"]
            minimum = configs["candidate"]["semi_markov_strong_transition_minimum_frames"]
            interval.update({
                "start_source": int(source[row]),
                "reference_state_at_start": int(prepared["truth"][row]),
                "duration_seconds": (interval["stop_row"] - row) / data["fps"],
                "support_only_from_extension": bool(
                    source[row] != 1 and cues["strong_future"][row, state]
                    and cues["short_strong_count"][row, state] < minimum
                ),
                "long_support_count": int(cues["long_strong_count"][row, state]),
                "candidate_strong_support": bool(cues["strong_future"][row, state]),
            })
        introduced = [item for item in lineage["details"] if item["lineage"] != "shared_spurious"]
        lineage["introduced_summary"] = {
            "events": len(introduced),
            "interpolated_starts": sum(item["start_source"] == 2 for item in introduced),
            "extension_only_support": sum(item["support_only_from_extension"] for item in introduced),
            "extension_only_below_baseline_fraction": sum(
                item["support_only_from_extension"]
                and item["long_support_count"]
                < configs["baseline"]["semi_markov_strong_transition_minimum_frames"]
                for item in introduced
            ),
            "observed_weak_support": sum(
                item["start_source"] == 1 and not item["candidate_strong_support"]
                for item in introduced
            ),
            "observed_strong_support": sum(
                item["start_source"] == 1 and item["candidate_strong_support"]
                for item in introduced
            ),
            "subsecond": sum(item["duration_seconds"] < 1 for item in introduced),
        }
        control_diagnosis = {}
        if control_cues:
            control_diagnosis = control_start_diagnosis(
                reference, candidate_predictions, models["candidate"]["labels"], lineage["details"],
                models["candidate"]["cues"], source, prepared["truth"], data["periods"], data["fps"], .75,
            )
        results.append({"case": name, "recoveries": recoveries, "high_iou_spells": high_iou,
                        "high_iou_spurious_lineage": lineage,
                        **({"control_diagnosis": control_diagnosis} if control_cues else {})})
        print("{}: recovery regressions={}, fixes={}, cost inversions={}; high-IoU lost={}, gained={}".format(
            name, recoveries["groups"]["regression"]["events"], recoveries["groups"]["fix"]["events"],
            recoveries["groups"]["regression"]["support_cost_inversion"], high_iou["lost"], high_iou["gained"],
        ), flush=True)
    report = {
        "schema_version": 1,
        "development_game": "Sample_Game_1" if game == "Sample_Game_1" else None,
        "previously_inspected_diagnostic_game": game if game == "Sample_Game_2" else None,
        "excluded_games": (["Sample_Game_2", "Sample_Game_3"] if game == "Sample_Game_1"
                           else ["Sample_Game_3"]),
        "selection_from_this_report_permitted": game == "Sample_Game_1",
        "runtime_fps": data["fps"], "rebase_report_sha256": _file_sha256(rebase_path),
        "protocol_sha256": _file_sha256(protocol_path), "baseline_sha256": rebase["baseline_sha256"],
        "original_model_outputs_reproduced": True, "input_sha256": expected["input_sha256"],
        "timing": {name: temporal_settings(config, data["fps"]) for name, config in configs.items()},
        "cases": results, "parameters_changed": False, "acceptance_guards_changed": False,
        "control_cues_measured": control_cues,
        "limitations": [
            "External Game1 diagnosis only; provider references are not independent continuous truth.",
            "Same one-to-one matching and full prediction extents as the scorer.",
            "Support-cost inversions and boundary drift are diagnostic associations, not causal proof.",
            "Long-window support counts are measured at the baseline's existing horizon, not a tuned new horizon.",
            "Post-recovery labels are excluded after reference support ends or a period boundary.",
        ],
        "code_sha256": {path.name: _file_sha256(path) for path in (
            Path(__file__), project / "possession" / "decoder.py",
            project / "possession" / "external_guarded_calibration.py",
        )},
    }
    _save(output, report)
    return report


def player_quality_features(players, ball, periods, fps, max_speed, pitch_margin):
    n = len(ball)
    if (n < 1 or ball.shape != (n, 2) or periods.shape != (n,)
            or players.ndim != 3 or players.shape[0] != n or players.shape[1] < 1 or players.shape[2] != 2):
        raise ValueError("player-quality audit requires aligned nonempty player tracks")
    if (not np.isfinite(fps) or fps <= 0 or not np.isfinite(max_speed) or max_speed <= 0
            or not np.isfinite(pitch_margin) or pitch_margin < 0):
        raise ValueError("player-quality audit requires valid fps, speed and pitch margin")
    finite = np.isfinite(players).all(axis=2)
    counts = finite.sum(axis=1)
    same_period = periods[1:] == periods[:-1]
    steps = (players[1:] - players[:-1]) * fps
    valid_steps = np.isfinite(steps).all(axis=2) & same_period[:, None]
    speeds = np.full(finite.shape, np.nan)
    speeds[1:][valid_steps] = np.linalg.norm(steps, axis=2)[valid_steps]
    outside = finite & (
        (players[:, :, 0] < -pitch_margin) | (players[:, :, 0] > PITCH_LENGTH + pitch_margin)
        | (players[:, :, 1] < -pitch_margin) | (players[:, :, 1] > PITCH_WIDTH + pitch_margin)
    )
    distances = np.linalg.norm(players - ball[:, None, :], axis=2)
    distances[~np.isfinite(distances)] = np.inf
    index = np.argmin(distances, axis=1)
    rows = np.arange(n)
    visible = np.isfinite(distances[rows, index])
    nearest_xy = np.full((n, 2), np.nan)
    nearest_xy[visible] = players[rows[visible], index[visible]]
    selected_speed = np.full(n, np.nan)
    selected_speed[visible] = speeds[rows[visible], index[visible]]
    correct_velocity = np.full((n, 2), np.nan)
    correct_available = visible & np.isfinite(selected_speed) & (selected_speed <= max_speed)
    selected_rows = rows[correct_available]
    correct_velocity[correct_available] = player_velocity_steps(
        players[selected_rows - 1, index[correct_available]],
        players[selected_rows, index[correct_available]], fps, max_speed,
    )
    legacy = _nearest_player_velocity(nearest_xy[:, None, :], fps, max_speed)[:, 0]
    legacy[1:][~same_period] = np.nan
    legacy_available = np.isfinite(legacy).all(axis=1)
    both = correct_available & legacy_available
    proxy_error = np.full(n, np.nan)
    proxy_error[both] = np.linalg.norm(legacy[both] - correct_velocity[both], axis=1)
    missing_proxy = np.full(n, np.nan)
    missing_proxy[correct_available] = ~legacy_available[correct_available]
    identity_switch = np.full(n, np.nan)
    comparable = visible[1:] & visible[:-1] & same_period
    identity_switch[1:][comparable] = (index[1:] != index[:-1])[comparable]
    implausible = np.full(n, np.nan)
    implausible[np.isfinite(selected_speed)] = selected_speed[np.isfinite(selected_speed)] > max_speed
    finite_speeds = speeds[np.isfinite(speeds)]
    summary = {
        "frames": n, "track_slots": players.shape[1],
        "valid_player_count_quantiles": dict(zip(
            ("q10", "median", "q90"), np.quantile(counts, [.1, .5, .9]).tolist()
        )),
        "missing_slot_fraction": float(1 - finite.mean()),
        "outside_pitch_margin_fraction_of_valid": float(outside.sum() / finite.sum()) if finite.sum() else None,
        "valid_track_steps": len(finite_speeds),
        "implausible_step_fraction": float(np.mean(finite_speeds > max_speed)) if len(finite_speeds) else None,
        "player_speed_p99_mps": float(np.quantile(finite_speeds, .99)) if len(finite_speeds) else None,
        "raw_nearest_visible_frames": int(visible.sum()),
        "stable_track_velocity_available_frames": int(correct_available.sum()),
        "legacy_proxy_missing_when_track_valid": int(np.count_nonzero(correct_available & ~legacy_available)),
    }
    return summary, {
        "visible_player_count": counts, "nearest_actual_speed_mps": selected_speed,
        "nearest_identity_changed": identity_switch, "implausible_nearest_step": implausible,
        "legacy_proxy_error_mps": proxy_error, "legacy_proxy_missing_when_track_valid": missing_proxy,
        "nearest_distance_metres": distances[rows, index],
    }


def audit_player_control_quality(root, diagnosis_path, output):
    diagnosis = json.loads(Path(diagnosis_path).read_text())
    if diagnosis["development_game"] != "Sample_Game_1" or not diagnosis["control_cues_measured"]:
        raise ValueError("player-quality cohorts require the Game1 control diagnosis")
    project = Path(__file__).resolve().parents[2]
    baseline_path = project / "configs" / "compliant_tracking.json"
    if _file_sha256(baseline_path) != diagnosis["baseline_sha256"]:
        raise ValueError("player-quality baseline differs from diagnosed control")
    baseline = json.loads(baseline_path.read_text())
    globals_by_game = []
    cohorts = []
    for game in ("Sample_Game_1", "Sample_Game_2"):
        data = _load_game(Path(root).resolve(), game)
        inputs = {path.name: _file_sha256(path) for path in data["input_paths"]}
        if game == "Sample_Game_1" and inputs != diagnosis["input_sha256"]:
            raise ValueError("player-quality inputs differ from diagnosed control")
        teams, features = {}, []
        for name, players in (("home", data["home"]), ("away", data["away"])):
            summary, arrays = player_quality_features(
                players, data["ball"], data["periods"], data["fps"],
                baseline["max_player_speed_metres_per_second"], baseline["pitch_margin_metres"],
            )
            teams[name] = summary
            features.append(arrays)
        globals_by_game.append({"game": game, "fps": data["fps"], "teams": teams, "input_sha256": inputs})
        if game == "Sample_Game_1":
            arrays = {key: np.column_stack([team[key] for team in features]) for key in features[0]}
            arrays["opposing_visible_player_count"] = arrays["visible_player_count"][:, ::-1]
            for case in diagnosis["cases"]:
                control = case["control_diagnosis"]
                matched = [
                    item for item in control["matched_starts"]
                    if item["start_source"] == 1 and item["previous_predicted_state"] in (HOME, AWAY)
                ]
                wrong = [
                    item for item in case["high_iou_spurious_lineage"]["details"]
                    if item["lineage"] != "shared_spurious" and item["start_source"] == 1
                    and item["previous_predicted_state"] in (HOME, AWAY)
                    and item["reference_state_at_start"] in (HOME, AWAY)
                    and item["reference_state_at_start"] != item["state"]
                    and not item["matched_at_iou_0.5"]
                ]
                cohorts.append({
                    "case": case["case"],
                    "matched_observed_handovers": cue_group_summary(matched, arrays),
                    "wrong_live_unmatched_observed_handovers": cue_group_summary(wrong, arrays),
                })
    report = {
        "schema_version": 1, "diagnosis_sha256": _file_sha256(diagnosis_path),
        "baseline_sha256": diagnosis["baseline_sha256"],
        "global_tracking": globals_by_game, "game1_control_cohorts": cohorts,
        "positions_or_teams_corrected": False,
        "code_sha256": {path.name: _file_sha256(path) for path in (
            Path(__file__), project / "possession" / "tracking.py",
            project / "possession" / "external_metrica_validation.py",
        )},
        "limitations": [
            "Raw external tracking geometry; no video review or challenge possession truth.",
            "Error cohorts are from the horizon-repaired Game1 component, not a new Game2 fit.",
            "Missing roster slots include substitutes/inactive tracks; interpret visible-player counts, not missing fraction alone.",
            "Outside nominal pitch is not automatically invalid; players may legitimately leave the field.",
            "Team metadata consistency does not prove visual team-assignment correctness.",
            "Legacy nearest-coordinate velocity is compared with the current selected player's own track step.",
        ],
    }
    _save(output, report)
    print("Player-quality audit saved; no positions or team assignments changed.", flush=True)
    return report


def audit_atd_player_quality(root, output):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], data_dir=str(Path(root).resolve()))
    metadata = load_tracking_metadata(meta_path)
    fps = float(ET.parse(meta_path).getroot().findtext(".//FrameRate"))
    if fps != GAMES[GAME]["fps"]:
        raise ValueError("ATD player audit metadata fps differs from the challenge contract")
    pitch = GAMES[GAME]["pitch"]
    n = GAMES[GAME]["n_video_frames"]
    project = Path(__file__).resolve().parents[2]
    config = json.loads((project / "configs" / "compliant_tracking.json").read_text())
    counts = np.zeros((n, 2), dtype=np.int16)
    valid_positions = np.zeros(2, dtype=np.int64)
    outside_positions = np.zeros(2, dtype=np.int64)
    valid_steps = np.zeros(2, dtype=np.int64)
    implausible_steps = np.zeros(2, dtype=np.int64)
    duplicate_frames = np.zeros(2, dtype=np.int64)
    previous = np.full((len(metadata.team_of_track), 2), np.nan)
    previous_frame = None
    rows_read = 0
    with open(raw_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split(":", 2)
            if len(parts) != 3:
                raise ValueError("ATD quality audit expected three fields at line {}".format(line_number))
            frame = int(parts[0]) - 1
            groups = parts[1].split(";")
            if (not 0 <= frame < n or len(groups) != len(previous)
                    or (previous_frame is not None and frame <= previous_frame)):
                raise ValueError("ATD quality audit found an invalid frame/track layout at line {}".format(line_number))
            current = np.full_like(previous, np.nan)
            for slot, group in enumerate(groups):
                player = _parse_group(group)
                if player is not None:
                    current[slot] = (player[0] * pitch[0], player[1] * pitch[1])
            finite = np.isfinite(current).all(axis=1)
            outside = finite & (
                (current[:, 0] < -config["pitch_margin_metres"])
                | (current[:, 0] > pitch[0] + config["pitch_margin_metres"])
                | (current[:, 1] < -config["pitch_margin_metres"])
                | (current[:, 1] > pitch[1] + config["pitch_margin_metres"])
            )
            speeds = np.full(len(previous), np.nan)
            if previous_frame is not None and frame == previous_frame + 1:
                comparable = finite & np.isfinite(previous).all(axis=1)
                speeds[comparable] = np.linalg.norm(current[comparable] - previous[comparable], axis=1) * fps
            for team in (HOME, AWAY):
                selected = metadata.team_of_track == team
                visible = selected & finite
                counts[frame, team] = visible.sum()
                valid_positions[team] += visible.sum()
                outside_positions[team] += np.count_nonzero(selected & outside)
                comparable = selected & np.isfinite(speeds)
                valid_steps[team] += comparable.sum()
                implausible_steps[team] += np.count_nonzero(
                    comparable & (speeds > config["max_player_speed_metres_per_second"])
                )
                if visible.any() and len(np.unique(current[visible], axis=0)) < visible.sum():
                    duplicate_frames[team] += 1
            previous, previous_frame = current, frame
            rows_read += 1
    active = np.all(counts > 0, axis=1)
    teams = {}
    for team, name in ((HOME, "home"), (AWAY, "away")):
        teams[name] = {
            "track_slots": int(np.count_nonzero(metadata.team_of_track == team)),
            "active_count_quantiles": dict(zip(
                ("q10", "median", "q90"), np.quantile(counts[active, team], [.1, .5, .9]).tolist()
            )) if active.any() else None,
            "active_frames_more_than_11": float(np.mean(counts[active, team] > 11)) if active.any() else None,
            "outside_margin_fraction": float(outside_positions[team] / valid_positions[team]) if valid_positions[team] else None,
            "implausible_step_fraction": float(implausible_steps[team] / valid_steps[team]) if valid_steps[team] else None,
            "frames_with_identical_track_coordinates": int(duplicate_frames[team]),
            "valid_position_entries": int(valid_positions[team]),
            "valid_consecutive_track_steps": int(valid_steps[team]),
        }
    report = {
        "schema_version": 1, "game": GAME, "fps": fps, "raw_rows": rows_read,
        "frames_with_both_teams_tracked": int(active.sum()), "teams": teams,
        "input_sha256": {Path(path).name: _file_sha256(path) for path in (meta_path, raw_path)},
        "parameters_selected_from_challenge_data": False,
        "positions_or_team_assignments_changed": False,
        "limitations": [
            "Automatic structural audit only: no video review or possession-truth comparison.",
            "Plausible positions and recognized team IDs are not proof of tracking or team-label accuracy.",
            "Outside-pitch tracks and more than11 visible players may include substitutes or people outside play.",
            "Thresholds are existing physical/geometry limits, not new challenge-fitted parameters.",
        ],
    }
    _save(output, report)
    print("ATD player-quality audit saved; no challenge-derived tuning or corrections.", flush=True)
    return report


def check_weak_transition_restore(root, protocol_path, diagnosis_path, output, output_config):
    localization = json.loads(Path(protocol_path).read_text())
    project = Path(__file__).resolve().parents[2]
    extension, protocol, baseline, previous, source = load_reliability_protocol(
        project / localization["source_protocol"]
    )
    diagnosis = json.loads(Path(diagnosis_path).read_text())
    if (not diagnosis.get("control_cues_measured") or not diagnosis["original_outputs_reproduced"]
            or diagnosis["candidate"] != extension["source_candidate"]
            or diagnosis["protocol_sha256"] != _file_sha256(protocol_path)):
        raise ValueError("weak-transition restore requires the matching frozen control diagnosis")
    for name, digest in diagnosis["code_sha256"].items():
        if _file_sha256(project / "possession" / name) != digest:
            raise ValueError("control-diagnosis source changed: {}".format(name))
    primary = next(case for case in diagnosis["cases"] if case["case"] == localization["primary_case"])
    groups = primary["control_diagnosis"]["candidate"]["observed_handover_groups"]
    wrong = groups["introduced_wrong_live_team"]["metrics"]["strong_future"]["mean"]
    matched = groups["matched"]["metrics"]["strong_future"]["mean"]
    if wrong is None or matched is None or wrong >= matched:
        raise ValueError("diagnosis does not support restoring unsupported-switch regularization")
    values = weak_transition_restore_values(baseline, source["values"])
    if any(item["values"] == values for item in previous["experiments"]):
        raise ValueError("fixed restore vector was already evaluated; do not repeat the search")
    prepared = _prepare(Path(root).resolve(), protocol, baseline)
    inputs = {path.name: _file_sha256(path) for path in prepared["data"]["input_paths"]}
    if inputs != previous["input_sha256"] or inputs != diagnosis["input_sha256"]:
        raise ValueError("weak-transition restore inputs differ from the original development study")
    experiments = []
    for name, candidate_values in (
        ("baseline", {key: baseline[key] for key in source["values"]}),
        ("near_miss", source["values"]),
        ("weak_transition_restore", values),
    ):
        evaluation = _evaluate(prepared, {**baseline, **candidate_values}, protocol)
        if name != "weak_transition_restore":
            expected = previous["experiments"][0]["evaluation"] if name == "baseline" else source["evaluation"]
            if any(result["label_sha256"] != expected["cases"][case]["label_sha256"]
                   for case, result in evaluation["cases"].items()):
                raise ValueError("restore replay changed the original {} outputs".format(name))
        checks = guarded_checks(
            evaluation, experiments[0]["evaluation"] if experiments else evaluation, protocol
        )
        experiments.append({
            "name": name, "values": candidate_values, "evaluation": evaluation, "checks": checks,
            "safe": all(check["passed"] for check in checks),
        })
        print("{}: objective={:.5f}, failures={}".format(
            name, evaluation["objective"], sum(not check["passed"] for check in checks)
        ), flush=True)
    restored = experiments[-1]
    gain = restored["evaluation"]["objective"] - experiments[0]["evaluation"]["objective"]
    improved = restored["safe"] and gain >= protocol["minimum_objective_improvement"]
    report = {
        "schema_version": 1, "development_game": "Sample_Game_1",
        "excluded_games": ["Sample_Game_2", "Sample_Game_3"],
        "source_candidate": extension["source_candidate"],
        "source_report_sha256": extension["source_report_sha256"],
        "base_protocol_sha256": extension["base_protocol_sha256"],
        "diagnosis_sha256": _file_sha256(diagnosis_path),
        "protocol_sha256": _file_sha256(protocol_path),
        "baseline_sha256": protocol["baseline_sha256"],
        "input_sha256": inputs, "experiments": experiments,
        "repair_changes_from_near_miss": {
            "semi_markov_live_transition_cost": {
                "before": source["values"]["semi_markov_live_transition_cost"],
                "after": values["semi_markov_live_transition_cost"],
            },
        },
        "guards": protocol["guards"],
        "original_comparators_reproduced_all_cases": True,
        "development_improvement_found": bool(improved),
        "selected_candidate": "weak_transition_restore" if improved else "baseline",
        "status": "ready_for_frozen_release_check" if improved else "no_safe_weak_transition_restore",
        "submission_changed": False, "full_active_provenance_audit_cleared": False,
        "code_sha256": {
            name: _file_sha256(project / "possession" / name)
            for name in diagnosis["code_sha256"]
        },
        "limitations": [
            "Diagnostic association motivates one fixed ablation, not a causal claim.",
            "Only the regular H/A cost is restored; supported-switch, dead, restart, and prediction settings are fixed.",
            "No speed veto, new feature, or acceptance-threshold relaxation.",
            "Game 1 development scores are not challenge scores or independent generalization evidence.",
            "No automatic promotion; a frozen release check and active-provenance audit remain necessary.",
        ],
    }
    _save(output, report)
    if improved:
        config = {**baseline, **values}
        config["core_development_provenance"] = {
            "development_game": "Sample_Game_1", "excluded_games": ["Sample_Game_2", "Sample_Game_3"],
            "baseline_sha256": protocol["baseline_sha256"],
            "development_report": str(Path(output).resolve()),
            "diagnosis_sha256": report["diagnosis_sha256"],
            "selected_candidate": "weak_transition_restore", "selected_overrides": values,
            "dead_restart_prediction_settings_reselected": False,
            "full_active_provenance_audit_cleared": False,
        }
        _save(output_config, config)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Localize external possession-spell errors before repair.")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--protocol", default=str(Path(__file__).resolve().parents[2] / "configs" / "archive" / "metrica_error_localization.json")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--control-cues", action="store_true")
    parser.add_argument("--weak-transition-restore", action="store_true")
    parser.add_argument("--diagnosis")
    parser.add_argument("--output-config")
    parser.add_argument("--rebased-regressions", action="store_true")
    parser.add_argument("--rebase-report")
    parser.add_argument("--game", choices=["Sample_Game_1", "Sample_Game_2"],
                        default="Sample_Game_1")
    parser.add_argument("--player-track-audit", action="store_true")
    parser.add_argument("--atd-player-audit", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.atd_player_audit:
            if args.player_track_audit or args.rebased_regressions or args.weak_transition_restore:
                raise ValueError("--atd-player-audit cannot be combined with other diagnostic modes")
            audit_atd_player_quality(args.root, args.output)
        elif args.player_track_audit:
            if not args.diagnosis or args.rebased_regressions or args.weak_transition_restore:
                raise ValueError("--player-track-audit requires --diagnosis and no other diagnostic mode")
            audit_player_control_quality(args.root, args.diagnosis, args.output)
        elif args.rebased_regressions:
            if not args.rebase_report or args.weak_transition_restore:
                raise ValueError("--rebased-regressions requires --rebase-report and no other diagnostic mode")
            diagnose_rebased_regressions(
                args.root, args.rebase_report, args.output, args.control_cues, args.game
            )
        elif args.weak_transition_restore:
            if not args.diagnosis or not args.output_config or args.control_cues:
                raise ValueError("restore requires --diagnosis and --output-config, not --control-cues")
            check_weak_transition_restore(
                args.root, args.protocol, args.diagnosis, args.output, args.output_config
            )
        else:
            localize_external_spells(args.root, args.protocol, args.output, args.control_cues)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        parser.exit(1, "error: {}\n".format(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
