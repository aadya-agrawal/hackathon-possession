import numpy as np

from possession import AWAY, DEAD, HOME, UNKNOWN
from evaluation.possession_metrics import match_spells, possession_spells, spell_overlaps


def recovery_references(events, frames, periods, fps, tolerance_seconds):
    row_by_frame = {int(frame): row for row, frame in enumerate(frames)}
    tolerance = int(round(tolerance_seconds * fps))
    losses = [
        event for event in events
        if event["type"] == "BALL LOST" and event["end_frame"] >= event["frame"]
    ]
    recoveries = sorted(
        [event for event in events if event["type"] == "RECOVERY"],
        key=lambda event: event["frame"],
    )
    teams_by_frame = {}
    for event in recoveries:
        teams_by_frame.setdefault(event["frame"], set()).add(event["team"])
    counts = {
        "source_recoveries": len(recoveries),
        "conflicting_recoveries": 0,
        "ambiguous_loss_matches": 0,
        "unpaired_recoveries": 0,
        "duplicate_recoveries": 0,
    }
    references = []
    seen = set()
    used_losses = set()
    for recovery in recoveries:
        key = (recovery["period"], recovery["frame"], recovery["team"])
        if key in seen:
            counts["duplicate_recoveries"] += 1
            continue
        seen.add(key)
        if len(teams_by_frame[recovery["frame"]]) > 1:
            counts["conflicting_recoveries"] += 1
            continue
        possible = [
            (index, loss) for index, loss in enumerate(losses)
            if loss["period"] == recovery["period"]
            and loss["team"] != recovery["team"]
            and loss["frame"] <= recovery["frame"]
            and abs(loss["end_frame"] - recovery["frame"]) <= tolerance
        ]
        if len(possible) != 1 or possible[0][0] in used_losses:
            counts[
                "ambiguous_loss_matches" if possible else "unpaired_recoveries"
            ] += 1
            continue
        loss_index, loss = possible[0]
        row = row_by_frame.get(recovery["frame"])
        loss_end = row_by_frame.get(loss["end_frame"])
        if row is None or loss_end is None:
            raise ValueError("loss/recovery reference is outside tracking frames")
        if periods[row] != recovery["period"] or periods[loss_end] != recovery["period"]:
            raise ValueError("loss/recovery reference crosses a period boundary")
        used_losses.add(loss_index)
        support_end = row + 1
        while support_end < len(periods) and periods[support_end] == periods[row]:
            support_end += 1
        for event in events:
            if event["period"] != recovery["period"]:
                continue
            if (
                event["type"] in {"RECOVERY", "PASS", "CARRY", "SET PIECE", "SHOT"}
                and event["team"] != recovery["team"]
            ):
                terminal_frame = event["frame"]
            elif event["type"] in {"BALL OUT", "BALL LOST"} and event["team"] == recovery["team"]:
                terminal_frame = event["end_frame"]
            elif event["type"] == "FAULT RECEIVED" and event["team"] == recovery["team"]:
                terminal_frame = event["frame"]
            else:
                continue
            terminal_row = row_by_frame.get(terminal_frame)
            if terminal_row is not None and terminal_row >= row:
                support_end = min(support_end, terminal_row)
        references.append({
            "row": row,
            "frame": recovery["frame"],
            "period": recovery["period"],
            "incoming": HOME if recovery["team"] == "home" else AWAY,
            "outgoing": HOME if loss["team"] == "home" else AWAY,
            "subtype": recovery["subtype"],
            "boundary_start_row": min(row, loss_end),
            "boundary_end_row": max(row, loss_end),
            "support_end_row": support_end,
            "reference_support_seconds": (support_end - row) / fps,
        })
    counts["paired_recoveries"] = len(references)
    return references, counts


def recovery_masks(ball, periods, references, fps, duration_seconds):
    length = max(1, int(round(duration_seconds * fps)))
    observed = np.isfinite(ball).all(axis=1)
    starts = []
    for reference in references:
        start = reference["row"] - length // 2
        stop = start + length
        if (
            start < 1 or stop >= len(ball)
            or not observed[start - 1:stop + 1].all()
            or not np.all(periods[start - 1:stop + 1] == reference["period"])
            or (starts and start < starts[-1] + length + 2)
        ):
            continue
        starts.append(start)
    masked = ball.copy()
    for start in starts:
        masked[start:start + length] = np.nan
    return masked, starts, length


def _match_boundaries(labels, periods, references, tolerance_frames):
    transitions = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    possible = []
    for index, reference in enumerate(references):
        matching = transitions[
            (periods[transitions] == reference["period"])
            & (periods[transitions - 1] == reference["period"])
            & (labels[transitions - 1] == reference["outgoing"])
            & (labels[transitions] == reference["incoming"])
        ]
        for row in matching:
            distance = max(
                reference["boundary_start_row"] - int(row),
                int(row) - reference["boundary_end_row"], 0,
            )
            if distance <= tolerance_frames:
                possible.append((
                    distance, abs(int(row) - reference["row"]), index, int(row)
                ))
    matched = {}
    used_rows = set()
    for _, _, index, row in sorted(possible):
        if index not in matched and row not in used_rows:
            matched[index] = row
            used_rows.add(row)
    return matched


def score_recovery_transitions(labels, periods, references, fps, settings):
    if not references:
        raise ValueError("recovery scoring requires paired loss/recovery references")
    rows = np.asarray([reference["row"] for reference in references])
    incoming = np.asarray([reference["incoming"] for reference in references])
    correct = labels[rows] == incoming
    matched = _match_boundaries(
        labels, periods, references,
        int(round(settings["boundary_search_seconds"] * fps)),
    )
    offsets = np.asarray([
        (row - references[index]["row"]) / fps for index, row in matched.items()
    ])
    if offsets.size:
        timing = {
            "matched": len(offsets),
            "median_signed_offset_seconds": float(np.median(offsets)),
            "p90_absolute_offset_seconds": float(np.percentile(np.abs(offsets), 90)),
            "early_fraction": float(np.mean(offsets < 0)),
            "late_fraction": float(np.mean(offsets > 0)),
        }
    else:
        timing = {"matched": 0}
    cohorts = {}
    short_limit = settings["short_reference_seconds"]
    for name, selected in (
        ("short_reference", np.asarray([
            reference["reference_support_seconds"] < short_limit
            for reference in references
        ])),
        ("longer_reference", np.asarray([
            reference["reference_support_seconds"] >= short_limit
            for reference in references
        ])),
    ):
        cohorts[name] = {
            "events": int(selected.sum()),
            "accuracy_at_recovery": (
                float(correct[selected].mean()) if selected.any() else None
            ),
        }
    persistence = {}
    for seconds in settings["post_recovery_seconds"]:
        outcomes = []
        for reference in references:
            row = reference["row"] + int(round(seconds * fps))
            if row < reference["support_end_row"]:
                outcomes.append(labels[row] == reference["incoming"])
        persistence[str(seconds)] = {
            "events": len(outcomes),
            "incoming_team_fraction": float(np.mean(outcomes)) if outcomes else None,
        }
    return {
        "references": len(references),
        "accuracy_at_recovery": float(correct.mean()),
        "previous_team_at_recovery_fraction": float(np.mean(
            labels[rows] == np.asarray([reference["outgoing"] for reference in references])
        )),
        "dead_at_recovery_fraction": float(np.mean(labels[rows] == DEAD)),
        "same_direction_boundary_recall": {
            str(seconds): len(_match_boundaries(
                labels, periods, references, int(round(seconds * fps))
            )) / len(references)
            for seconds in settings["boundary_tolerance_seconds"]
        },
        "timing_within_search_window": timing,
        "cohorts": cohorts,
        "event_derived_post_recovery_support": persistence,
    }


def possession_chain_references(events, frames, periods, fps, settings):
    if len(frames) != len(periods) or len(frames) == 0:
        raise ValueError("reference timeline must contain aligned frames and periods")
    row_by_frame = {int(frame): row for row, frame in enumerate(frames)}
    paired, recovery_counts = recovery_references(
        events, frames, periods, fps, settings["loss_pair_tolerance_seconds"]
    )
    votes = np.zeros((len(frames) + 1, 3), dtype=np.int32)
    boundaries = []
    owners = sorted(
        [event for event in events if event["type"] in {
            "PASS", "CARRY", "RECOVERY", "BALL LOST", "BALL OUT", "SHOT", "SET PIECE",
        }],
        key=lambda event: event["frame"],
    )
    stops = []
    restarts = []

    def claim(start, stop, state):
        if stop > start:
            votes[start, state] += 1
            votes[stop, state] -= 1

    for event in events:
        row = row_by_frame.get(event["frame"])
        if row is None or periods[row] != event["period"]:
            raise ValueError("event reference has an invalid frame or period")
        state = HOME if event["team"] == "home" else AWAY
        tokens = set(event["subtype"].split("-"))
        end = row_by_frame.get(event["end_frame"])
        if event["type"] in {"PASS", "CARRY", "BALL LOST", "BALL OUT", "SHOT"}:
            if end is None or end < row or periods[end] != periods[row]:
                raise ValueError("live event reference has an invalid endpoint")
            claim(row, max(row + 1, end), state)
        elif event["type"] in {"RECOVERY", "SET PIECE"} and "RETAKEN" not in tokens:
            claim(row, row + 1, state)
            restarts.append((row, event, state))
        if event["type"] == "BALL OUT":
            stops.append((end, event, state))
        elif event["type"] == "FAULT RECEIVED":
            stops.append((row, event, state))
        elif event["type"] == "SHOT" and tokens & {"OUT", "GOAL"}:
            stops.append((end, event, state))
        elif event["type"] == "BALL LOST" and tokens & {
            "FORCED", "VOLUNTARY", "OFFSIDE", "END HALF",
        }:
            stops.append((end, event, state))

    stop_rows = {row for row, _, _ in stops}
    for first, second in zip(owners[:-1], owners[1:]):
        if (
            first["team"] != second["team"] or first["period"] != second["period"]
            or first["type"] not in {"PASS", "CARRY", "RECOVERY", "SET PIECE"}
            or "RETAKEN" in first["subtype"]
        ):
            continue
        start = row_by_frame[first["frame"]]
        stop = row_by_frame[second["frame"]]
        if not any(start <= row < stop for row in stop_rows):
            claim(start, stop, HOME if first["team"] == "home" else AWAY)

    rejected_dead_intervals = 0
    for stop, event, outgoing in stops:
        boundaries.append({
            "row": stop, "lower": stop, "upper": stop, "old": outgoing,
            "new": DEAD, "period": event["period"], "source": event["type"],
        })
        possible = [
            item for item in restarts
            if item[0] > stop and item[1]["period"] == event["period"]
        ]
        if not possible:
            rejected_dead_intervals += 1
            continue
        restart, restart_event, incoming = min(possible, key=lambda item: item[0])
        interrupted = any(
            other["type"] in {"PASS", "CARRY", "BALL LOST", "SHOT"}
            and other["period"] == event["period"]
            and stop < row_by_frame[other["frame"]] < restart
            for other in events
        )
        if interrupted:
            rejected_dead_intervals += 1
            continue
        claim(stop, restart, DEAD)
        boundaries.extend([
            {"row": restart, "lower": restart, "upper": restart, "old": DEAD,
             "new": incoming, "period": restart_event["period"], "source": "restart"},
        ])
    for reference in paired:
        boundaries.append({
            "row": reference["row"],
            "lower": reference["boundary_start_row"],
            "upper": reference["boundary_end_row"],
            "old": reference["outgoing"], "new": reference["incoming"],
            "period": reference["period"], "source": "loss_recovery_pair",
        })
    # Kickoff is a confirmed start even when the pre-kickoff dead interval is censored.
    for row, event, state in restarts:
        if event["type"] == "SET PIECE" and event["subtype"] == "KICK OFF":
            boundaries.append({
                "row": row, "lower": row, "upper": row, "old": DEAD,
                "new": state, "period": event["period"], "source": "kickoff",
            })
    by_row = {}
    for boundary in boundaries:
        by_row.setdefault(boundary["row"], []).append(boundary)
    unambiguous = []
    conflicting_boundaries = 0
    for group in by_row.values():
        if len({(item["old"], item["new"]) for item in group}) == 1:
            unambiguous.append(group[0])
        else:
            conflicting_boundaries += 1
    boundaries = sorted(unambiguous, key=lambda item: item["row"])
    support = np.cumsum(votes[:-1], axis=0) > 0
    unique = support.sum(axis=1) == 1
    raw = np.full(len(frames), UNKNOWN, dtype=np.int8)
    raw[unique] = np.argmax(support[unique], axis=1)
    truth = raw.copy()
    guard = int(round(settings["boundary_guard_seconds"] * fps))
    for boundary in boundaries:
        start = max(0, boundary["lower"] - guard)
        stop = min(len(frames), boundary["upper"] + guard + 1)
        selected = np.arange(start, stop)
        truth[selected[periods[selected] == boundary["period"]]] = UNKNOWN
    spells = []
    for first, second in zip(boundaries[:-1], boundaries[1:]):
        if (
            first["period"] != second["period"]
            or first["new"] not in (HOME, AWAY)
            or first["new"] != second["old"]
        ):
            continue
        core_start = first["upper"] + guard + 1
        core_stop = second["lower"] - guard
        if core_stop <= core_start or not np.all(truth[core_start:core_stop] == first["new"]):
            continue
        spells.append({
            "start_row": first["row"], "stop_row": second["row"],
            "core_start_row": core_start, "core_stop_row": core_stop,
            "state": first["new"], "period": first["period"],
            "start_uncertainty_rows": [first["lower"], first["upper"]],
            "end_uncertainty_rows": [second["lower"], second["upper"]],
        })
    segments = []
    edges = np.r_[0, np.flatnonzero(np.diff(truth)) + 1, len(truth)]
    period_edges = np.flatnonzero(np.diff(periods)) + 1
    edges = np.unique(np.r_[edges, period_edges])
    for start, stop in zip(edges[:-1], edges[1:]):
        segments.append({
            "start_row": int(start), "stop_row": int(stop),
            "state": int(truth[start]), "period": int(periods[start]),
        })
    return truth, {
        "source": "Metrica provider-event-derived references",
        "challenge_equivalent_ground_truth": False,
        "segments": segments,
        "complete_live_spells": spells,
        "boundaries": boundaries,
        "recovery_references": paired,
        "summary": {
            "frames": len(truth),
            "known_frames": int((truth != UNKNOWN).sum()),
            "coverage": float(np.mean(truth != UNKNOWN)),
            "frames_by_state": {
                name: int((truth == code).sum())
                for code, name in ((HOME, "H"), (AWAY, "A"), (DEAD, "D"), (UNKNOWN, "unknown"))
            },
            "conflicting_claim_frames": int(np.count_nonzero(support.sum(axis=1) > 1)),
            "conflicting_boundaries": conflicting_boundaries,
            "rejected_dead_intervals": rejected_dead_intervals,
            "complete_live_spells": len(spells),
            "recoveries": recovery_counts,
        },
    }


def reference_spell_inputs(labels, references, periods):
    runs = list(zip(
        np.r_[0, np.flatnonzero(np.diff(periods)) + 1],
        np.r_[np.flatnonzero(np.diff(periods)) + 1, len(labels)],
    ))
    complete = references["complete_live_spells"]
    gt_spells = [(item["start_row"], item["stop_row"], item["state"]) for item in complete]
    predictions = [
        (start, stop, state) for start, stop, state in possession_spells(labels, runs)
        if any(
            start < item["core_stop_row"] and stop > item["core_start_row"]
            for item in complete
        )
    ]
    return gt_spells, predictions


def score_censored_dead_frames(labels, truth):
    if (
        labels.shape != truth.shape
        or not np.isin(labels, (HOME, AWAY, DEAD)).all()
        or not np.isin(truth, (HOME, AWAY, DEAD, UNKNOWN)).all()
    ):
        raise ValueError("dead-state scoring requires aligned H/A/D predictions and censored references")
    known_dead = truth == DEAD
    known_live = np.isin(truth, (HOME, AWAY))
    if not known_dead.any() or not known_live.any():
        raise ValueError("dead-state scoring needs both known dead and known live external frames")
    predicted_dead = labels == DEAD
    tp = int(np.count_nonzero(predicted_dead & known_dead))
    fp = int(np.count_nonzero(predicted_dead & known_live))
    denominator = tp + fp
    return {
        "known_dead_frames": int(np.count_nonzero(known_dead)),
        "known_live_frames": int(np.count_nonzero(known_live)),
        "unknown_frames_excluded": int(np.count_nonzero(truth == UNKNOWN)),
        "predicted_dead_on_known_frames": denominator,
        "correct_dead_frames": tp,
        "false_dead_on_known_live_frames": fp,
        "dead_precision_on_known_frames": tp / denominator if denominator else None,
        "dead_recall_on_known_frames": tp / int(np.count_nonzero(known_dead)),
        "false_dead_fraction_of_known_live": fp / int(np.count_nonzero(known_live)),
    }


def score_possession_references(labels, truth, references, periods, ious=(0.5, 0.75)):
    known = truth != UNKNOWN
    if not known.any() or labels.shape != truth.shape:
        raise ValueError("possession scoring requires aligned labels and known references")
    if not np.isin(labels, [HOME, AWAY, DEAD]).all():
        raise ValueError("predictions must contain only H/A/D codes")
    confusion = np.bincount(
        truth[known].astype(int) * 3 + labels[known].astype(int), minlength=9
    ).reshape(3, 3)
    per_state = {}
    f1s = []
    for state, name in enumerate(("H", "A", "D")):
        tp = int(confusion[state, state])
        actual = int(confusion[state].sum())
        predicted = int(confusion[:, state].sum())
        denominator = actual + predicted
        f1 = 2 * tp / denominator if denominator else 0.0
        per_state[name] = {
            "support": actual, "f1": f1,
            "recall": tp / actual if actual else None,
        }
        f1s.append(f1)
    gt_spells, predictions = reference_spell_inputs(labels, references, periods)
    overlaps = spell_overlaps(gt_spells, predictions)
    spell_metrics = {}
    for iou in ious:
        tp, fp, fn = match_spells(overlaps, len(gt_spells), len(predictions), iou)
        denominator = 2 * tp + fp + fn
        spell_metrics[str(iou)] = {
            "matched": tp, "spurious_in_reference_regions": fp, "missed": fn,
            "f1": 2 * tp / denominator if denominator else None,
        }
    return {
        "known_frame_accuracy": float(np.mean(labels[known] == truth[known])),
        "known_frame_macro_f1": float(np.mean(f1s)),
        "known_live_macro_f1": float(np.mean(f1s[:2])),
        "per_state": per_state,
        "reference_conditioned_spell_scores": spell_metrics,
        "complete_reference_spells": len(gt_spells),
        "predicted_spells_in_reference_regions": len(predictions),
        "limitations": (
            "Unknown/censored regions are excluded; full prediction extents are retained. "
            "Provider-reference spell scores are not challenge spell F1."
        ),
    }
