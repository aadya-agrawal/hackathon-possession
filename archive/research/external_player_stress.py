import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from possession.cli import load_config
from possession.external_guarded_calibration import _evaluate, _prepare, _save
from possession.external_metrica_validation import (
    _file_sha256, _load_events, _load_game, _scoring_events,
    load_validation_protocol,
)
from possession import HOME, AWAY, UNKNOWN


def choose_player_stress_windows(data, settings):
    ball, periods = data["ball"], data["periods"]
    home, away = data["home"], data["away"]
    fps = data["fps"]
    length = max(1, int(round(settings["window_seconds"] * fps)))
    if length < 2 or settings["windows_per_game"] < 1:
        raise ValueError("player stress requires multi-frame windows and a positive sample count")
    if (settings["near_ball_metres"] <= 0 or settings["minimum_players_per_team"] < 1
            or settings["jitter_std_metres"] < 0):
        raise ValueError("player stress requires valid geometry and a nonnegative jitter")
    valid_ball = np.isfinite(ball).all(axis=1)
    home_valid = np.isfinite(home).all(axis=2)
    away_valid = np.isfinite(away).all(axis=2)
    near = []
    for positions, valid in ((home, home_valid), (away, away_valid)):
        distances = np.linalg.norm(positions - ball[:, None, :], axis=2)
        distances[~valid_ball[:, None] | ~valid] = np.inf
        near.append(distances)
    nearest = np.stack([np.min(values, axis=1) for values in near], axis=1)
    team = np.argmin(nearest, axis=1)
    rows = np.arange(len(ball))
    slot = np.where(team == 0, np.argmin(near[0], axis=1), np.argmin(near[1], axis=1))
    for candidate in (home, away):
        if candidate.ndim != 3 or candidate.shape[0] != len(ball) or candidate.shape[2] != 2:
            raise ValueError("player stress requires aligned tracking frames")
    observed = (
        valid_ball & (home_valid.sum(axis=1) >= settings["minimum_players_per_team"])
        & (away_valid.sum(axis=1) >= settings["minimum_players_per_team"])
    )
    cumulative = np.r_[0, np.cumsum(observed, dtype=np.int32)]
    start = rows[:len(ball) - length + 1]
    same_period = periods[start] == periods[start + length - 1]
    valid = (
        same_period & (cumulative[start + length] - cumulative[start] == length)
        & (nearest[start, team[start]] <= settings["near_ball_metres"])
    )
    candidates = start[valid]
    if len(candidates) < settings["windows_per_game"]:
        raise ValueError("insufficient valid observed near-ball player stress windows")
    selected = []
    for target in np.linspace(0, len(candidates) - 1, settings["windows_per_game"], dtype=int):
        index = int(target)
        while index < len(candidates) and selected and candidates[index] < selected[-1] + length + 2:
            index += 1
        if index == len(candidates):
            raise ValueError("player stress cannot place nonoverlapping windows in each sampled cohort")
        selected.append(int(candidates[index]))
    return [
        {"start_row": row, "end_row": row + length, "team": int(team[row]),
         "slot": int(slot[row]), "period": int(periods[row])}
        for row in selected
    ]


def corrupt_player_tracks(data, windows, settings, mode):
    if mode not in settings["modes"]:
        raise ValueError("player stress mode is not frozen by its protocol")
    players = {team: data[team].copy() for team in ("home", "away")}
    rng = np.random.default_rng(settings["seed"])
    used = []
    for item in windows:
        start, stop, team, slot = (
            item["start_row"], item["end_row"], item["team"], item["slot"]
        )
        if (start < 0 or stop > len(data["ball"]) or start >= stop
                or not np.all(data["periods"][start:stop] == item["period"])):
            raise ValueError("player stress window crosses a period or input boundary")
        current = players["home" if team == 0 else "away"]
        other = players["away" if team == 0 else "home"]
        if slot < 0 or slot >= current.shape[1]:
            raise ValueError("player stress refers to an invalid team track")
        occupied = np.isfinite(current[start:stop, slot]).all(axis=1)
        if not occupied[0]:
            raise ValueError("selected near-ball track is missing at the stress anchor")
        if mode == "nearest_dropout":
            current[start:stop, slot] = np.nan
            used.append(item)
        elif mode == "near_player_jitter":
            shifts = rng.normal(0, settings["jitter_std_metres"], (stop - start, 2))
            current[start:stop, slot][occupied] += shifts[occupied]
            used.append(item)
        else:
            free = np.flatnonzero(
                ~np.isfinite(other[start:stop]).any(axis=(0, 2))
            )
            if not len(free):
                continue
            displaced = current[start:stop, slot].copy()
            other[start:stop, free[0]] = displaced
            current[start:stop, slot] = np.nan
            used.append({**item, "receiving_slot": int(free[0])})
    if not used:
        raise ValueError("player stress could not corrupt a single valid track")
    return players, used


def local_stress_effect(original, stressed, truth, periods, windows, radius):
    if original.shape != stressed.shape or truth.shape != original.shape or periods.shape != truth.shape:
        raise ValueError("player stress comparison needs aligned labels and references")
    region = np.zeros(len(original), dtype=bool)
    direct = np.zeros(len(original), dtype=bool)
    for item in windows:
        start, stop = item["start_row"], item["end_row"]
        if (start < 0 or stop > len(region) or start >= stop
                or not np.all(periods[start:stop] == item["period"])):
            raise ValueError("player stress locality window is out of bounds or crosses periods")
        direct[start:stop] = True
        left, right = start, stop
        while left > 0 and left > start - radius and periods[left - 1] == item["period"]:
            left -= 1
        while right < len(region) and right < stop + radius and periods[right] == item["period"]:
            right += 1
        region[left:right] = True
    known_live = region & np.isin(truth, [HOME, AWAY])
    return {
        "direct_frames": int(direct.sum()),
        "nearby_frames": int(region.sum()),
        "direct_label_changes": int(np.count_nonzero(original[direct] != stressed[direct])),
        "nearby_label_changes": int(np.count_nonzero(original[region] != stressed[region])),
        "known_live_nearby_frames": int(known_live.sum()),
        "known_live_accuracy_before": (
            float(np.mean(original[known_live] == truth[known_live])) if known_live.any() else None
        ),
        "known_live_accuracy_after": (
            float(np.mean(stressed[known_live] == truth[known_live])) if known_live.any() else None
        ),
    }


def evaluate_player_stress(root, protocol_path, output, only_game=None,
                           extra_report=None, extra_sha256=None):
    project = Path(__file__).resolve().parents[2]
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol["games"] != {
        "development": ["Sample_Game_1"],
        "previously_inspected_diagnostic": ["Sample_Game_2"],
    } or protocol["modes"] != ["nearest_dropout", "near_player_jitter", "team_swap"]:
        raise ValueError("player stress must preserve the declared games and three corruption modes")
    all_games = [name for names in protocol["games"].values() for name in names]
    if only_game is not None and only_game not in all_games:
        raise ValueError("player stress game is not declared by its protocol")
    source = project / protocol["soft_contact_report"]
    if _file_sha256(source) != protocol["soft_contact_report_sha256"]:
        raise ValueError("player stress source candidate changed")
    soft = json.loads(source.read_text())
    if (extra_report is None) != (extra_sha256 is None):
        raise ValueError("player stress extra candidate requires both report and frozen SHA-256")
    extra = None
    if extra_report is not None:
        if _file_sha256(extra_report) != extra_sha256:
            raise ValueError("player stress extra candidate report changed")
        extra = json.loads(Path(extra_report).read_text())
        if (extra["candidate"] not in (
                "age_confident_near_ball_control", "short_gap_resumed_player_confidence")
                or extra["baseline_sha256"] != protocol["baseline_sha256"]
                or extra["results"][0]["game"] != "Sample_Game_1"
                or extra["values"]["player_reliability_enabled"] is not True
                or extra["values"]["player_reliability_minimum_age_frames"] != 8
                or {key: value for key, value in extra["values"].items()
                    if key not in ("player_reliability_enabled", "player_reliability_minimum_age_frames",
                                   "player_reliability_resume_gap_frames")}
                != soft["values"]):
            raise ValueError("player stress extra candidate must be the frozen age-confidence ablation")
    baseline_path = project / protocol["baseline_config"]
    scenario_path = project / protocol["scenario_protocol"]
    if (_file_sha256(baseline_path) != protocol["baseline_sha256"]
            or _file_sha256(scenario_path) != protocol["scenario_protocol_sha256"]):
        raise ValueError("player stress baseline or scenario policy changed")
    baseline = load_config(baseline_path)
    scenario = json.loads(scenario_path.read_text())
    validation = load_validation_protocol(project / "configs" / "external_transfer_validation_v2.json")
    root = Path(root).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != validation["metrica"]["repository_commit"]:
        raise ValueError("player stress source repository differs from frozen validation")
    if soft["evaluated_games"] != ["Sample_Game_1", "Sample_Game_2"]:
        raise ValueError("player stress requires the authorized frozen soft-contact comparison")
    results = []
    for role, games in protocol["games"].items():
        for game in games:
            if only_game is not None and game != only_game:
                continue
            data = _load_game(root, game)
            settings = {
                **scenario, "source_fps": 25, "development_game": game,
                "live_missing_durations_seconds": [],
                "tag_modes": {key: scenario["tag_modes"][key] for key in ("none", "degraded_restart_tags")},
            }
            events = _scoring_events(
                data, _load_events(data["event_path"], set(validation["metrica"]["event_control_types"]))
            )
            previous = next(item for item in soft["results"] if item["game"] == game)
            inputs = {path.name: _file_sha256(path) for path in data["input_paths"]}
            if inputs != previous["input_sha256"]:
                raise ValueError("player stress game inputs differ from frozen soft-contact study")
            windows = choose_player_stress_windows(data, protocol)
            modes = {}
            original_labels = {}
            for mode in ("original", *protocol["modes"]):
                if mode == "original":
                    arrays, used = None, []
                else:
                    arrays, used = corrupt_player_tracks(data, windows, protocol, mode)
                prepare_config = {
                    **baseline,
                    **({"player_reliability_resume_gap_frames":
                        extra["values"]["player_reliability_resume_gap_frames"]}
                       if extra is not None
                       and "player_reliability_resume_gap_frames" in extra["values"]
                       else {}),
                }
                prepared = _prepare(root, settings, prepare_config, arrays)
                candidates = {}
                tested_configs = [
                    ("deployment", baseline),
                    ("frozen_soft_contact", {**baseline, **soft["values"]}),
                ]
                if extra is not None:
                    tested_configs.append(("frozen_age_confidence", {**baseline, **extra["values"]}))
                for name, config in tested_configs:
                    evaluation, labels = _evaluate(prepared, config, settings, events, include_labels=True)
                    if mode == "original":
                        old = (
                            previous["deployment"] if name == "deployment"
                            else previous["candidate"]["evaluation"] if name == "frozen_soft_contact"
                            else extra["results"][0]["candidate"]["evaluation"] if game == "Sample_Game_1"
                            else None
                        )
                        if old is not None and any(
                                case["label_sha256"] != old["cases"][key]["label_sha256"]
                                for key, case in evaluation["cases"].items()
                        ):
                            raise ValueError("player stress failed to reproduce the frozen original labels")
                        original_labels[name] = labels
                    candidates[name] = evaluation
                    if mode != "original":
                        for case_name, current in labels.items():
                            candidates[name]["cases"][case_name]["local_player_stress"] = local_stress_effect(
                                original_labels[name][case_name], current, prepared["truth"],
                                data["periods"], used,
                                baseline["semi_markov_strong_transition_window_frames"],
                            )
                modes[mode] = {
                    "windows_used": len(used), "selected_windows": used,
                    "models": candidates,
                }
                print("{} {}: windows={} baseline={:.5f} soft={:.5f}{}".format(
                    game, mode, len(used),
                    candidates["deployment"]["objective"], candidates["frozen_soft_contact"]["objective"],
                    " age={:.5f}".format(candidates["frozen_age_confidence"]["objective"])
                    if extra is not None else "",
                ), flush=True)
            results.append({
                "game": game, "role": role, "input_sha256": inputs,
                "window_seconds": protocol["window_seconds"],
                "preselected_windows": windows,
                "modes": modes,
            })
    report = {
        "schema_version": 1, "external_data": True,
        "external_revision": revision,
        "protocol_sha256": _file_sha256(protocol_path),
        "source_candidate_sha256": protocol["soft_contact_report_sha256"],
        "extra_candidate_sha256": extra_sha256,
        "baseline_sha256": protocol["baseline_sha256"],
        "results": results, "parameters_changed": False,
        "evaluated_games": [item["game"] for item in results],
        "incomplete_cross_game_diagnostic": only_game is not None,
        "challenge_data_used_for_selection": False,
        "original_outputs_reproduced": True,
        "limitations": [
            "Synthetic perturbations of external player tracking, not actual tracking annotation errors.",
            "Candidate is diagnostic and already fails development guards; these results cannot select/promote it.",
            "Games 1-2 were inspected; Game 2 is reused diagnostic and not a fresh holdout.",
            "Provider references and synthetic restart tags remain imperfect stand-ins for challenge inputs.",
            "Windows are chosen from automatic player/ball geometry, never provider labels or game outcomes.",
            "Subset evaluation covers clean and recovery-masked cases with both tag modes; it is not the eight-case development objective or release gate.",
            "An age-confidence candidate is diagnostic and already fails one Game1 development guard; Game2 is previously inspected.",
        ],
        "code_sha256": {path.name: _file_sha256(path) for path in (
            Path(__file__), project / "possession" / "tracking.py",
            project / "possession" / "external_guarded_calibration.py",
            project / "possession" / "decoder.py",
        )},
    }
    _save(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose external player-dropout and team-slot corruption.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", default=str(
        Path(__file__).resolve().parents[2] / "configs" / "metrica_player_stress.json"
    ))
    parser.add_argument("--output", required=True)
    parser.add_argument("--game", choices=["Sample_Game_1", "Sample_Game_2"])
    parser.add_argument("--extra-candidate-report")
    parser.add_argument("--extra-candidate-sha256")
    args = parser.parse_args(argv)
    try:
        evaluate_player_stress(args.root, args.protocol, args.output, args.game,
                               args.extra_candidate_report, args.extra_candidate_sha256)
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, "error: {}\n".format(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
