import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np

from evaluation.possession_metrics import match_spells, possession_spells, spell_overlaps
from possession import AWAY, DEAD, HOME
from possession.cli import load_config
from possession.decoder import build_evidence, decode
from possession.external_guarded_calibration import _save
from possession.external_metrica_validation import _file_sha256
from possession.external_possession_events import score_censored_dead_frames
from possession.external_transfer_check import restart_only_tags
from possession.tagging import infer_restart_ownership, localize_restarts
from possession.tracking import control_features_from_arrays
from possession.trajectory import repair_ball_trajectory


SECTIONS = ("firstHalf", "secondHalf")


def load_dfl_match(metadata_path, positions_path):
    root = ET.parse(metadata_path).getroot()
    general = root.find(".//General")
    environment = root.find(".//Environment")
    if general is None or environment is None:
        raise ValueError("DFL validation needs match information and pitch metadata")
    home_id, away_id = general.get("HomeTeamId"), general.get("GuestTeamId")
    pitch = (float(environment.get("PitchX")), float(environment.get("PitchY")))
    if not home_id or not away_id or home_id == away_id or not np.isfinite(pitch).all():
        raise ValueError("DFL match has invalid teams or pitch")
    with open(positions_path, "rb") as handle:
        return _parse_frame_sets(handle, home_id, away_id, pitch)


def _parse_frame_sets(handle, home_id, away_id, pitch):
    periods = {}
    for _, frameset in ET.iterparse(handle, events=("end",)):
        if frameset.tag != "FrameSet":
            continue
        section = frameset.get("GameSection")
        team_id = frameset.get("TeamId")
        if team_id == "BALL":
            if section not in SECTIONS:
                raise ValueError("DFL positions have unknown section")
            frames = frameset.findall("Frame")
            if not frames:
                raise ValueError("DFL positions contain an empty FrameSet")
            if section not in periods:
                first = int(frames[0].get("N"))
                last = int(frames[-1].get("N"))
                if last < first or len(frames) != last - first + 1:
                    raise ValueError("DFL ball timeline is not contiguous")
                if len(frames) < 2 or abs(
                    (datetime.fromisoformat(frames[1].get("T"))
                     - datetime.fromisoformat(frames[0].get("T"))).total_seconds() - 0.04
                ) > 1e-4:
                    raise ValueError("DFL ball timeline must be sampled at 25 fps")
                periods[section] = {
                    "first_frame": first, "frames": len(frames),
                    "home": {}, "away": {}, "ball": None, "truth": None,
                }
        frameset.clear()
    if set(periods) != set(SECTIONS):
        raise ValueError("DFL validation requires a ball timeline in both halves")
    handle.seek(0)
    for _, frameset in ET.iterparse(handle, events=("end",)):
        if frameset.tag != "FrameSet":
            continue
        section = frameset.get("GameSection")
        team_id = frameset.get("TeamId")
        if section not in SECTIONS or team_id not in (home_id, away_id, "BALL", "referee"):
            raise ValueError("DFL positions have unknown section or team")
        if team_id == "referee":
            frameset.clear()
            continue
        frames = frameset.findall("Frame")
        if not frames:
            raise ValueError("DFL positions contain an empty FrameSet")
        period = periods[section]
        n = period["frames"]
        rows = np.asarray([int(frame.get("N")) - period["first_frame"] for frame in frames])
        if np.any((rows < 0) | (rows >= n)) or len(np.unique(rows)) != len(rows):
            raise ValueError("DFL FrameSet has duplicate or out-of-period frames")
        xy = np.asarray([
            (float(frame.get("X")) + pitch[0] / 2,
             float(frame.get("Y")) + pitch[1] / 2) for frame in frames
        ], dtype=np.float32)
        if not np.isfinite(xy).all():
            raise ValueError("DFL FrameSet has nonfinite coordinates")
        if team_id == "BALL":
            if period["ball"] is not None or len(rows) != n or not np.array_equal(rows, np.arange(n)):
                raise ValueError("DFL ball timeline must be complete and unique within each half")
            status = np.asarray([int(frame.get("BallStatus")) for frame in frames])
            owner = np.asarray([int(frame.get("BallPossession")) for frame in frames])
            if not np.isin(status, (0, 1)).all() or not np.isin(owner, (1, 2)).all():
                raise ValueError("DFL ball status and possession must use documented codes")
            truth = np.where(status == 0, DEAD, np.where(owner == 1, HOME, AWAY)).astype(np.int8)
            period["ball"], period["truth"] = xy, truth
        else:
            person = frameset.get("PersonId")
            team = "home" if team_id == home_id else "away"
            if not person or person in period[team]:
                raise ValueError("DFL player track needs a unique person ID within its half")
            positions = np.full((n, 2), np.nan, dtype=np.float32)
            positions[rows] = xy
            period[team][person] = positions
        frameset.clear()
    if set(periods) != set(SECTIONS):
        raise ValueError("DFL validation requires two complete halves")
    result = []
    for section in SECTIONS:
        period = periods[section]
        if period["ball"] is None or any(not period[team] for team in ("home", "away")):
            raise ValueError("DFL half lacks ball or player positions")
        result.append({
            "section": section, "ball": period["ball"], "truth": period["truth"],
            "home": np.stack([period["home"][key] for key in sorted(period["home"])], axis=1),
            "away": np.stack([period["away"][key] for key in sorted(period["away"])], axis=1),
        })
    return pitch, result


def score_dfl_labels(labels, truth, period_lengths):
    if labels.shape != truth.shape or sum(period_lengths) != len(labels):
        raise ValueError("DFL scoring requires aligned model labels and two complete halves")
    dead = score_censored_dead_frames(labels, truth)
    starts = np.cumsum([0] + list(period_lengths))
    runs = list(zip(starts[:-1], starts[1:]))
    reference = possession_spells(truth, runs)
    predicted = possession_spells(labels, runs)
    overlaps = spell_overlaps(reference, predicted)
    scores = {}
    for iou in (0.3, 0.4, 0.5, 0.6, 0.7, 0.75):
        tp, fp, fn = match_spells(overlaps, len(reference), len(predicted), iou)
        scores[str(iou)] = {
            "matched": tp, "spurious": fp, "missed": fn,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        }
    return {
        "known_frame_accuracy": float(np.mean(labels == truth)),
        "live_team_accuracy": float(np.mean(labels[truth != DEAD] == truth[truth != DEAD])),
        "dead": dead, "reference_live_spells": len(reference),
        "predicted_live_spells": len(predicted), "live_spell_f1": scores,
        "reference_source": "DFL per-frame ball status and possession flags; not challenge truth",
    }


def audit_dfl_spell_structure(labels, truth, period_lengths, ball_xy, pitch):
    n = len(truth)
    if (
        not isinstance(labels, np.ndarray) or labels.shape != (n,)
        or not isinstance(truth, np.ndarray) or truth.shape != (n,)
        or not np.isin(labels, (HOME, AWAY, DEAD)).all()
        or not np.isin(truth, (HOME, AWAY, DEAD)).all()
        or len(period_lengths) != 2 or any(length < 1 for length in period_lengths)
        or sum(period_lengths) != n
        or not isinstance(ball_xy, np.ndarray) or ball_xy.shape != (n, 2)
        or not np.isfinite(ball_xy).all()
        or np.shape(pitch) != (2,) or not np.isfinite(pitch).all()
        or np.any(np.asarray(pitch) <= 0)
    ):
        raise ValueError("DFL spell audit requires aligned two-half labels and observed ball")
    starts = np.cumsum([0] + list(period_lengths))
    runs = list(zip(starts[:-1], starts[1:]))
    counts = {
        "reference_live_spells": len(possession_spells(truth, runs)),
        "predicted_live_spells": len(possession_spells(labels, runs)),
        "reference_dead_spells": 0,
        "predicted_dead_spells": 0,
        "reference_dead_spells_completely_missed": 0,
        "reference_dead_frames_in_completely_missed_spells": 0,
        "reference_same_team_restart_pairs": 0,
        "reference_same_team_restart_pairs_merged": 0,
        "reference_live_starts_after_dead": 0,
        "reference_direct_live_switches": 0,
        "predicted_direct_live_switches": 0,
        "reference_direct_switches_correct_at_boundary": 0,
        "reference_direct_switches_incoming_label_within_1s": 0,
        "reference_direct_switches_incoming_label_within_2s": 0,
        "reference_direct_switches_still_outgoing_for_2s": 0,
        "reference_dead_frames_inside_pitch": 0,
        "reference_dead_frames_outside_pitch": 0,
        "reference_dead_consecutive_stationary_steps": 0,
    }
    for start, stop in runs:
        reference = truth[start:stop]
        predicted = labels[start:stop]
        dead = reference == DEAD
        outside = (
            (ball_xy[start:stop, 0] < 0) | (ball_xy[start:stop, 0] > pitch[0])
            | (ball_xy[start:stop, 1] < 0) | (ball_xy[start:stop, 1] > pitch[1])
        )
        counts["reference_dead_frames_inside_pitch"] += int(np.count_nonzero(dead & ~outside))
        counts["reference_dead_frames_outside_pitch"] += int(np.count_nonzero(dead & outside))
        counts["reference_dead_consecutive_stationary_steps"] += int(np.count_nonzero(
            dead[1:] & dead[:-1]
            & np.all(ball_xy[start + 1:stop] == ball_xy[start:stop - 1], axis=1)
        ))
        for series, key in ((reference, "reference_dead_spells"),
                            (predicted, "predicted_dead_spells")):
            boundaries = np.r_[0, np.flatnonzero(series[1:] != series[:-1]) + 1, len(series)]
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                if series[left] != DEAD:
                    continue
                counts[key] += 1
                if key != "reference_dead_spells":
                    continue
                if not np.any(predicted[left:right] == DEAD):
                    counts["reference_dead_spells_completely_missed"] += 1
                    counts["reference_dead_frames_in_completely_missed_spells"] += int(right - left)
                if left > 0 and right < len(series) and series[left - 1] == series[right]:
                    counts["reference_same_team_restart_pairs"] += 1
                    counts["reference_same_team_restart_pairs_merged"] += int(np.all(
                        predicted[left - 1:right + 1] == series[right]
                    ))
        counts["reference_live_starts_after_dead"] += int(np.count_nonzero(
            (reference[:-1] == DEAD) & np.isin(reference[1:], (HOME, AWAY))
        ))
        true_switches = np.flatnonzero(
            (reference[:-1] != reference[1:])
            & np.isin(reference[:-1], (HOME, AWAY))
            & np.isin(reference[1:], (HOME, AWAY))
        ) + 1
        counts["reference_direct_live_switches"] += len(true_switches)
        counts["predicted_direct_live_switches"] += int(np.count_nonzero(
            (predicted[:-1] != predicted[1:])
            & np.isin(predicted[:-1], (HOME, AWAY))
            & np.isin(predicted[1:], (HOME, AWAY))
        ))
        for row in true_switches:
            incoming, outgoing = reference[row], reference[row - 1]
            counts["reference_direct_switches_correct_at_boundary"] += int(predicted[row] == incoming)
            for seconds in (1, 2):
                counts["reference_direct_switches_incoming_label_within_{}s".format(seconds)] += int(
                    np.any(predicted[row:min(len(predicted), row + 25 * seconds + 1)] == incoming)
                )
            counts["reference_direct_switches_still_outgoing_for_2s"] += int(
                row + 50 < len(predicted) and np.all(predicted[row:row + 51] == outgoing)
            )
    return {
        **counts,
        "provider_flags_are_not_physical_touch_annotations": True,
        "stationary_ball_may_reflect_provider_encoding": True,
    }


def evaluate_dfl_match(metadata_path, positions_path, baseline, candidate, *,
                       spell_structure=False):
    pitch, periods = load_dfl_match(metadata_path, positions_path)
    predictions = {"baseline": [], "exit_bracket": []}
    lengths, diagnostics = [], {"baseline": {}, "exit_bracket": {}}
    for index, period in enumerate(periods):
        ball = period["ball"]
        n = len(ball)
        lengths.append(n)
        trajectory = repair_ball_trajectory(
            ball, 25, max_gap_frames=baseline["max_interpolation_gap_frames"],
            max_speed=baseline["max_ball_speed_metres_per_second"],
            prediction_history_frames=baseline["prediction_history_frames"],
            prediction_horizon_frames=baseline["prediction_horizon_frames"],
            max_acceleration=baseline["max_ball_acceleration_metres_per_second_squared"],
            pitch=pitch, pitch_margin=baseline["prediction_pitch_margin_metres"],
        )
        control = control_features_from_arrays(
            period["home"], period["away"], trajectory.xy, 25, baseline,
        )
        tags = restart_only_tags(
            [], np.full(n, index + 1, dtype=np.int8), 25, baseline, None, 2027,
        )
        tags = infer_restart_ownership(
            localize_restarts(tags, trajectory, 25, baseline),
            trajectory, control, baseline,
        )
        for name, config in (("baseline", baseline), ("exit_bracket", candidate)):
            evidence = build_evidence(control, trajectory, tags, pitch, config, fps=25)
            predictions[name].append(decode(evidence, tags, config))
            if evidence.exit_bracket_diagnostics is not None:
                for key, value in evidence.exit_bracket_diagnostics.items():
                    diagnostics[name][key] = diagnostics[name].get(key, 0) + value
    truth = np.concatenate([period["truth"] for period in periods])
    results = {}
    for name in predictions:
        labels = np.concatenate(predictions[name])
        results[name] = {
            **score_dfl_labels(labels, truth, lengths),
            "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "source_only_exit_bracket": diagnostics[name] or None,
        }
        if spell_structure:
            results[name]["spell_structure"] = audit_dfl_spell_structure(
                labels, truth, lengths,
                np.concatenate([period["ball"] for period in periods]), pitch,
            )
    return {
        "period_lengths": lengths, "pitch_metres": list(pitch),
        "role": "DFL provider flag diagnostic, distinct from Metrica references",
        "models": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score frozen candidate on public DFL per-frame flags")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--positions", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spell-structure", action="store_true")
    args = parser.parse_args(argv)
    baseline = load_config(args.baseline_config)
    candidate = load_config(args.candidate_config)
    report = evaluate_dfl_match(
        args.metadata, args.positions, baseline, candidate,
        spell_structure=args.spell_structure,
    )
    report.update({
        "schema_version": 1,
        "metadata_sha256": _file_sha256(Path(args.metadata)),
        "positions_sha256": _file_sha256(Path(args.positions)),
        "baseline_config_sha256": _file_sha256(Path(args.baseline_config)),
        "candidate_config_sha256": _file_sha256(Path(args.candidate_config)),
        "code_sha256": _file_sha256(Path(__file__)),
        "selected_submission_changed": False,
    })
    _save(args.output, report)
    for name, result in report["models"].items():
        print("{}: D recall={:.4f}, D precision={:.4f}, F1@0.5={:.4f}, F1@0.75={:.4f}".format(
            name, result["dead"]["dead_recall_on_known_frames"],
            result["dead"]["dead_precision_on_known_frames"],
            result["live_spell_f1"]["0.5"]["f1"], result["live_spell_f1"]["0.75"]["f1"],
        ))


if __name__ == "__main__":
    main()
