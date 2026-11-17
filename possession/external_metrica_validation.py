import argparse
import csv
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from games import GAME, GAMES
from possession import AWAY, DEAD, HOME, UNKNOWN
from possession.cli import load_config
from possession.decoder import build_evidence, decode
from possession.external_possession_events import (
    recovery_masks,
    recovery_references,
    score_recovery_transitions,
)
from possession.submission import summarize
from possession.tagging import TagFeatures
from possession.tracking import (
    _parse_group, control_features_from_arrays, load_raw_ball,
)
from possession.trajectory import repair_ball_trajectory
from starter.baseline import smooth


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def load_validation_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if "base_protocol" in protocol:
        extension = protocol
        project = Path(__file__).resolve().parents[1]
        base_path = project / extension["base_protocol"]
        if _file_sha256(base_path) != extension["base_protocol_sha256"]:
            raise ValueError("validation base protocol changed")
        protocol = json.loads(base_path.read_text())
        if "base_protocol" in protocol:
            raise ValueError("validation protocols may inherit only one level")
        for key, value in extension.items():
            if key in ("metrica", "policy"):
                protocol[key] = {**protocol[key], **value}
            elif key not in ("base_protocol", "base_protocol_sha256"):
                protocol[key] = value
    parity = protocol.get("runtime_parity")
    if parity is not None:
        if parity["game"] != GAME or parity["fps"] != GAMES[GAME]["fps"]:
            raise ValueError("runtime parity must match the challenge's declared game and fps")
        project = Path(__file__).resolve().parents[1]
        if _file_sha256(project / protocol["frozen_config"]) != parity["config_sha256"]:
            raise ValueError("runtime-parity selected config changed")
        if protocol["frozen_config_source_fps"] != parity["fps"]:
            raise ValueError("deployment validation must use runtime fps, not calibration metadata")
        for candidate in protocol["candidate_comparison"]["candidates"]:
            role = candidate.get("timing_role")
            if role not in ("deployment", "historical_calibration"):
                raise ValueError("runtime-parity candidates require an explicit timing role")
            if role == "deployment" and candidate["source_fps"] != parity["fps"]:
                raise ValueError("deployment candidate source fps differs from runtime")
    return protocol


def config_at_fps(config, source_fps, target_fps):
    if not (
        np.isfinite(source_fps) and source_fps > 0
        and np.isfinite(target_fps) and target_fps > 0
    ):
        raise ValueError("source and target fps must be finite and positive")
    ratio = float(target_fps) / float(source_fps)
    result = dict(config)
    for key, value in config.items():
        if key.endswith("_frames") and isinstance(value, (int, float)):
            minimum = 2 if key in {
                "semi_markov_minimum_live_frames",
                "semi_markov_minimum_dead_frames",
                "hsmm_minimum_live_frames",
                "hsmm_minimum_dead_frames",
            } else int(value > 0)
            result[key] = max(minimum, int(math.floor(value * ratio + 0.5)))
        elif key.endswith("_frames_by_type") and isinstance(value, dict):
            result[key] = {
                name: max(int(frames > 0), int(math.floor(frames * ratio + 0.5)))
                for name, frames in value.items()
            }
    if "restart_ownership_minimum_votes" in config:
        result["restart_ownership_minimum_votes"] = max(
            1, int(math.floor(config["restart_ownership_minimum_votes"] * ratio + 0.5))
        )
    # Transition costs compete with sums of per-frame emissions.
    for key in (
        "semi_markov_live_transition_cost",
        "semi_markov_strong_transition_cost",
        "semi_markov_dead_transition_cost",
        "semi_markov_start_dead_bonus",
        "core_uncertain_live_transition_cost",
        "core_uncertain_strong_transition_cost",
        "touch_motion_weak_transition_cost",
        "phase_turnover_transition_cost",
    ):
        if key in config:
            result[key] = config[key] * ratio
    return result


def _to_metres(xy):
    result = np.asarray(xy, dtype=np.float64).copy()
    result[..., 0] = (result[..., 0] - 0.5) * PITCH_LENGTH
    result[..., 1] = (0.5 - result[..., 1]) * PITCH_WIDTH
    return result


def _load_tracking(path):
    path = Path(path)
    with path.open(newline="") as handle:
        rows = csv.reader(handle)
        next(rows)
        next(rows)
        header = next(rows)
    values = np.genfromtxt(
        path, delimiter=",", skip_header=3, dtype=np.float64
    )
    if values.ndim != 2 or values.shape[1] != len(header):
        raise ValueError("{}: invalid tracking table".format(path))
    frames = values[:, 1].astype(np.int64)
    ball = _to_metres(values[:, -2:])
    player_columns = values[:, 3:-2]
    if player_columns.shape[1] % 2:
        raise ValueError("{}: unmatched player coordinate".format(path))
    players = _to_metres(
        player_columns.reshape(len(values), -1, 2)
    )
    return frames, ball, players


def _load_events(path, allowed_types):
    path = Path(path)
    if path.suffix == ".json":
        metadata = ET.parse(path.with_name(path.name.replace("_events.json", "_metadata.xml"))).getroot()
        teams = list(metadata.iter("Team"))
        if len(teams) != 2:
            raise ValueError("{}: expected two metadata teams".format(path))
        by_id = {team.get("id"): name for team, name in zip(teams, ("home", "away"))}
        by_name = {team.findtext("Name"): name for team, name in zip(teams, ("home", "away"))}
        records = json.loads(path.read_text())["data"]
        events = []
        for record in records:
            kind = record["type"]["name"].strip().upper()
            if kind not in allowed_types:
                continue
            team = by_id.get(record["team"].get("id")) or by_name.get(record["team"].get("name"))
            if team is None:
                raise ValueError("{}: event team is not in metadata".format(path))
            subtypes = record.get("subtypes")
            if subtypes is None:
                subtype = ""
            elif isinstance(subtypes, dict):
                subtype = subtypes["name"].strip()
            elif isinstance(subtypes, list):
                subtype = "-".join(item["name"].strip() for item in subtypes)
            else:
                raise ValueError("{}: invalid event subtypes".format(path))
            events.append({
                "frame": int(record["start"]["frame"]),
                "end_frame": int(record["end"]["frame"]),
                "period": int(record["period"]),
                "team": team, "type": kind, "subtype": subtype,
            })
        return events
    events = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            event_type = row["Type"].strip().upper()
            if event_type not in allowed_types:
                continue
            try:
                frame = int(row["Start Frame"])
                period = int(row["Period"])
                end_frame = int(row["End Frame"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "{}: invalid event frame or period".format(path)
                ) from error
            team = row["Team"].strip().lower()
            if team not in {"home", "away"}:
                raise ValueError("{}: invalid event team".format(path))
            events.append({
                "frame": frame,
                "team": team,
                "type": event_type,
                "period": period,
                "end_frame": end_frame,
                "subtype": row["Subtype"].strip(),
            })
    return events


def _select_windows(ball, length, count=100, periods=None):
    if length < 1 or count < 1:
        raise ValueError("gap length and window count must be positive")
    observed = np.isfinite(ball).all(axis=1)
    candidates = []
    for start in range(1, len(ball) - length - 1):
        stop = start + length
        if (
            observed[start - 1:stop + 1].all()
            and (periods is None or
                 np.all(periods[start - 1:stop + 1] == periods[start]))
            and (not candidates or start >= candidates[-1] + length + 2)
        ):
            candidates.append(start)
    if not candidates:
        return []
    return [
        candidates[int(round(index))]
        for index in np.linspace(
            0, len(candidates) - 1, min(count, len(candidates))
        )
    ]


def _repair_by_period(ball, fps, config, periods=None):
    if periods is None:
        periods = np.ones(len(ball), dtype=np.int8)
    boundaries = np.r_[0, np.flatnonzero(np.diff(periods)) + 1, len(ball)]
    pieces = [
        repair_ball_trajectory(
            ball[start:stop],
            fps,
            max_gap_frames=int(config["max_interpolation_gap_frames"]),
            max_speed=float(config["max_ball_speed_metres_per_second"]),
            prediction_history_frames=int(config["prediction_history_frames"]),
            prediction_horizon_frames=int(config["prediction_horizon_frames"]),
            max_acceleration=float(
                config["max_ball_acceleration_metres_per_second_squared"]
            ),
            pitch=(PITCH_LENGTH, PITCH_WIDTH),
            pitch_margin=float(config["prediction_pitch_margin_metres"]),
        )
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]
    trajectory = pieces[0]
    for field in ("xy", "velocity", "speed", "confidence", "source"):
        setattr(trajectory, field, np.concatenate([
            getattr(piece, field) for piece in pieces
        ]))
    return trajectory


def _gap_metrics(ball, fps, duration, config, periods=None, repair=True):
    length = max(1, int(round(duration * fps)))
    starts = _select_windows(ball, length, periods=periods)
    masked = ball.copy()
    masked_frames = []
    for start in starts:
        masked[start:start + length] = np.nan
        masked_frames.extend(range(start, start + length))
    prediction = (
        _repair_by_period(masked, fps, config, periods).xy if repair else masked
    )
    indexes = np.asarray(masked_frames, dtype=np.int64)
    if indexes.size == 0:
        raise ValueError(
            "no eligible {} second gap windows".format(duration)
        )
    recovered = np.isfinite(prediction[indexes]).all(axis=1)
    errors = np.linalg.norm(
        prediction[indexes][recovered] - ball[indexes][recovered],
        axis=1,
    )
    all_errors = np.linalg.norm(prediction[indexes] - ball[indexes], axis=1)
    window_errors = all_errors.reshape(len(starts), length)
    per_window = [
        {
            "start_row": start,
            "recovery_fraction": float(np.isfinite(error).mean()),
            "within_2m_fraction_of_masked": float((error <= 2.0).mean()),
        } for start, error in zip(starts, window_errors)
    ]
    return {
        "duration_seconds": float(duration),
        "duration_frames": length,
        "actual_duration_seconds": length / fps,
        "windows": len(starts),
        "mask_start_rows": starts,
        "per_window": per_window,
        "masked_frames": int(indexes.size),
        "recovered_frames": int(np.count_nonzero(recovered)),
        "recovery_fraction": float(np.mean(recovered)),
        "abstention_fraction": float(1.0 - np.mean(recovered)),
        "within_0.5m_fraction_of_masked": (
            int(np.count_nonzero(errors <= 0.5)) / int(indexes.size)
        ),
        "within_2m_fraction_of_masked": (
            int(np.count_nonzero(errors <= 2.0)) / int(indexes.size)
        ),
        "median_error_metres": (
            float(np.median(errors)) if errors.size else None
        ),
        "p90_error_metres": (
            float(np.percentile(errors, 90)) if errors.size else None
        ),
    }


def _event_team_metrics(frames, ball, home, away, events):
    row_by_frame = {
        int(frame): index for index, frame in enumerate(frames)
    }
    correct = 0
    evaluated = 0
    by_type = {}
    distances = []
    for event in events:
        row = row_by_frame.get(event["frame"])
        if row is None or not np.isfinite(ball[row]).all():
            continue
        team_distances = {}
        for team, players in (("home", home), ("away", away)):
            valid = np.isfinite(players[row]).all(axis=1)
            if np.any(valid):
                team_distances[team] = float(np.min(np.linalg.norm(
                    players[row][valid] - ball[row], axis=1
                )))
        if len(team_distances) != 2:
            continue
        predicted = min(team_distances, key=team_distances.get)
        is_correct = predicted == event["team"]
        evaluated += 1
        correct += int(is_correct)
        distances.append(team_distances[event["team"]])
        item = by_type.setdefault(event["type"], [0, 0])
        item[0] += int(is_correct)
        item[1] += 1
    return {
        "events": evaluated,
        "correct": correct,
        "accuracy": correct / evaluated if evaluated else 0.0,
        "event_team_distance_median": (
            float(np.median(distances)) if distances else None
        ),
        "by_type": {
            key: {
                "correct": value[0],
                "events": value[1],
                "accuracy": value[0] / value[1],
            }
            for key, value in sorted(by_type.items())
        },
    }


def _load_game(root, game):
    if game == "Sample_Game_3":
        return _load_epts_game(root / "data" / game, game)
    game_root = root / "data" / game
    home_path = game_root / (game + "_RawTrackingData_Home_Team.csv")
    away_path = game_root / (game + "_RawTrackingData_Away_Team.csv")
    home_frames, home_ball, home = _load_tracking(home_path)
    away_frames, away_ball, away = _load_tracking(away_path)
    timeline = np.genfromtxt(
        home_path, delimiter=",", skip_header=3, usecols=(0, 1, 2)
    )
    away_timeline = np.genfromtxt(
        away_path, delimiter=",", skip_header=3, usecols=(0, 1, 2)
    )
    if not np.array_equal(timeline, away_timeline):
        raise ValueError("{}: team frame timelines differ".format(game))
    if (
        not np.isfinite(timeline).all()
        or not np.all(np.diff(home_frames) == 1)
        or not np.all(timeline[:, :2] == np.floor(timeline[:, :2]))
    ):
        raise ValueError("{}: invalid tracking timeline".format(game))
    steps = np.diff(timeline[:, 2])
    if not np.allclose(steps, steps[0]) or steps[0] <= 0:
        raise ValueError("{}: nonuniform frame times".format(game))
    shared = (
        np.isfinite(home_ball).all(axis=1)
        & np.isfinite(away_ball).all(axis=1)
    )
    if not np.allclose(home_ball[shared], away_ball[shared], atol=1e-6):
        raise ValueError("{}: team ball coordinates differ".format(game))
    # Production evidence expects a positive-coordinate pitch, not centred xy.
    for array in (home_ball, home, away):
        array[..., 0] += PITCH_LENGTH / 2
        array[..., 1] = PITCH_WIDTH / 2 - array[..., 1]
    return {
        "frames": home_frames,
        "periods": timeline[:, 0].astype(np.int8),
        "fps": float(1.0 / steps[0]),
        "ball": home_ball,
        "home": home,
        "away": away,
        "event_path": game_root / (game + "_RawEventsData.csv"),
        "input_paths": [home_path, away_path, game_root / (game + "_RawEventsData.csv")],
    }


def _load_epts_game(game_root, game):
    meta_path = game_root / (game + "_metadata.xml")
    raw_path = game_root / (game + "_tracking.txt")
    root = ET.parse(meta_path).getroot()
    fps = float(root.findtext("./Metadata/GlobalConfig/FrameRate"))
    pitch = (float(root.findtext(".//FieldSize/Width")), float(root.findtext(".//FieldSize/Height")))
    if not np.isfinite(fps) or fps <= 0 or pitch != (PITCH_LENGTH, PITCH_WIDTH):
        raise ValueError("external EPTS fps or pitch dimensions are invalid")
    channels = list(root.findall(".//Sensor[@id='position']/Channels/Channel"))
    if len(channels) != 2 or any(channel.findtext("Unit") != "normalized" for channel in channels):
        raise ValueError("external EPTS reader requires normalized x/y channels")
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("external EPTS reader requires two teams")
    players = list(root.iter("Player"))
    slot_by_id = {player.get("id"): slot for slot, player in enumerate(players)}
    if len(slot_by_id) != len(players):
        raise ValueError("duplicate EPTS player identities")
    team_by_slot = np.asarray([
        0 if player.get("teamId") == teams[0].get("id") else 1
        for player in players
    ])
    if any(player.get("teamId") not in {team.get("id") for team in teams} for player in players):
        raise ValueError("EPTS player has an unknown team")
    channel_by_id = {channel.get("id"): channel for channel in root.iter("PlayerChannel")}
    formats = []
    for spec in root.findall("./DataFormatSpecifications/DataFormatSpecification"):
        if spec.get("separator") != ":":
            raise ValueError("unsupported external EPTS frame separator")
        registers = spec.findall("SplitRegister")
        if len(registers) != 2 or registers[0].get("separator") != ";":
            raise ValueError("unsupported external EPTS register layout")
        if (
            registers[1].get("separator") != ","
            or [ref.get("channelId") for ref in registers[1].findall("BallChannelRef")] != ["x", "y"]
        ):
            raise ValueError("EPTS ball register must contain normalized x then y")
        slots = []
        for group in registers[0].findall("SplitRegister"):
            if group.get("separator") != ",":
                raise ValueError("unsupported EPTS player coordinate separator")
            refs = group.findall("PlayerChannelRef")
            if len(refs) != 2:
                raise ValueError("EPTS player groups must contain x and y")
            pair = [channel_by_id[ref.get("playerChannelId")] for ref in refs]
            if (
                [channel.get("channelId") for channel in pair] != ["x", "y"]
                or pair[0].get("playerId") != pair[1].get("playerId")
            ):
                raise ValueError("EPTS player channel pair is inconsistent")
            slots.append(slot_by_id[pair[0].get("playerId")])
        if len(slots) != len(set(slots)):
            raise ValueError("duplicate active EPTS player slots")
        formats.append((int(spec.get("startFrame")), int(spec.get("endFrame")), slots))
    formats.sort()
    if not formats or formats[0][0] != 1 or any(
        left[1] + 1 != right[0] for left, right in zip(formats, formats[1:])
    ):
        raise ValueError("EPTS formats must cover one contiguous frame timeline")
    n_frames = formats[-1][1]
    periods = np.zeros(n_frames, dtype=np.int8)
    parameters = {
        item.findtext("Name"): item.findtext("Value")
        for item in root.findall(".//ProviderGlobalParameters/ProviderParameter")
    }
    for number, name in enumerate(("first_half", "second_half", "first_extra_half", "second_extra_half"), start=1):
        start, stop = parameters.get(name + "_start"), parameters.get(name + "_end")
        if not start and not stop:
            continue
        if not start or not stop:
            raise ValueError("incomplete EPTS period metadata")
        first, last = int(start) - 1, int(stop)
        if first < 0 or last > n_frames or first >= last or np.any(periods[first:last]):
            raise ValueError("invalid EPTS period bounds")
        periods[first:last] = number
    if np.any(periods == 0):
        raise ValueError("EPTS periods do not cover the tracking timeline")
    positions = np.full((n_frames, len(players), 2), np.nan, dtype=np.float32)
    layout = 0
    seen = 0
    with raw_path.open() as handle:
        for seen, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split(":")
            if len(fields) != 3 or int(fields[0]) != seen or seen > n_frames:
                raise ValueError("invalid or misaligned EPTS frame")
            while seen > formats[layout][1]:
                layout += 1
            groups = fields[1].split(";")
            slots = formats[layout][2]
            if len(groups) != len(slots):
                raise ValueError("EPTS active player count differs from metadata format")
            for slot, group in zip(slots, groups):
                point = _parse_group(group)
                if point is not None:
                    positions[seen - 1, slot] = (point[0] * pitch[0], point[1] * pitch[1])
    if seen != n_frames:
        raise ValueError("EPTS tracking frame count differs from metadata")
    ball, _ = load_raw_ball(raw_path, n_frames, pitch)
    event_path = game_root / (game + "_events.json")
    return {
        "frames": np.arange(1, n_frames + 1), "periods": periods, "fps": fps,
        "ball": ball, "home": positions[:, team_by_slot == HOME],
        "away": positions[:, team_by_slot == AWAY],
        "event_path": event_path, "input_paths": [meta_path, raw_path, event_path],
    }


def validate_metrica(root, protocol_path):
    root = Path(root).resolve()
    protocol_path = Path(protocol_path)
    protocol = load_validation_protocol(protocol_path)
    settings = protocol["metrica"]
    config_path = Path(__file__).resolve().parents[1] / protocol[
        "frozen_config"
    ]
    config_bytes = config_path.read_bytes()
    config = load_config(config_path)
    source_fps = protocol.get(
        "frozen_config_source_fps", config["compliance_provenance"]["target_fps"]
    )
    results = []
    for game in settings["games"]["validation"]:
        data = _load_game(root, game)
        events = _load_events(
            data["event_path"], set(settings["event_control_types"])
        )
        fps = data["fps"]
        scaled_config = config_at_fps(config, source_fps, fps)
        gap_results = [
            _gap_metrics(
                data["ball"], fps, item["duration_seconds"],
                scaled_config, data["periods"]
            )
            for item in settings["gap_checks"]
        ]
        event_result = _event_team_metrics(
            data["frames"], data["ball"], data["home"], data["away"], events
        )
        results.append({
            "game": game,
            "frames": len(data["frames"]),
            "fps": fps,
            "effective_interpolation_gap_frames": scaled_config[
                "max_interpolation_gap_frames"
            ],
            "effective_interpolation_gap_seconds": (
                scaled_config["max_interpolation_gap_frames"] / fps
            ),
            "observed_ball_frames": int(np.count_nonzero(
                np.isfinite(data["ball"]).all(axis=1)
            )),
            "gap_checks": gap_results,
            "event_team": event_result,
        })

    checks = []
    for game in results:
        by_duration = {
            item["duration_seconds"]: item
            for item in game["gap_checks"]
        }
        for policy in settings["gap_checks"]:
            result = by_duration[policy["duration_seconds"]]
            if "minimum_recovery_fraction" in policy:
                checks.append({
                    "name": "{}_{}_recovery".format(
                        game["game"], policy["duration_seconds"]
                    ),
                    "passed": (
                        result["recovery_fraction"]
                        >= policy["minimum_recovery_fraction"]
                    ),
                    "actual": result["recovery_fraction"],
                    "minimum": policy["minimum_recovery_fraction"],
                })
            if "minimum_abstention_fraction" in policy:
                checks.append({
                    "name": "{}_{}_abstention".format(
                        game["game"], policy["duration_seconds"]
                    ),
                    "passed": (
                        result["abstention_fraction"]
                        >= policy["minimum_abstention_fraction"]
                    ),
                    "actual": result["abstention_fraction"],
                    "minimum": policy["minimum_abstention_fraction"],
                })
            if result["p90_error_metres"] is not None:
                checks.append({
                    "name": "{}_{}_p90_error".format(
                        game["game"], policy["duration_seconds"]
                    ),
                    "passed": (
                        result["p90_error_metres"]
                        <= policy["maximum_p90_error_metres"]
                    ),
                    "actual": result["p90_error_metres"],
                    "maximum": policy["maximum_p90_error_metres"],
                })
            elif "minimum_recovery_fraction" in policy:
                checks.append({
                    "name": "{}_{}_p90_error".format(
                        game["game"], policy["duration_seconds"]
                    ),
                    "passed": False,
                    "reason": "No recovered frames; error cannot be measured",
                })
        checks.append({
            "name": "{}_event_team_accuracy".format(game["game"]),
            "passed": (
                game["event_team"]["accuracy"]
                >= settings["minimum_event_team_accuracy"]
            ),
            "actual": game["event_team"]["accuracy"],
            "minimum": settings["minimum_event_team_accuracy"],
        })
    return {
        "schema_version": 2,
        "dataset": "Metrica Sports sample-data",
        "external_data": True,
        "repository_commit": settings["repository_commit"],
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": _file_sha256(protocol_path),
        "frozen_config": str(config_path.resolve()),
        "frozen_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "untouched_games": settings["games"]["untouched_test"],
        "results": results,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "limitations": [
            "Event-team accuracy is nearest-team geometry, not decoder accuracy.",
            "No frame-level possession truth or spell F1 is available.",
        ],
    }


def _candidate_labels(data, config, baseline_window=None):
    ball = data["ball"]
    periods = data["periods"]
    fps = data["fps"]
    boundaries = np.r_[0, np.flatnonzero(np.diff(periods)) + 1, len(ball)]
    labels = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        xy = ball[start:stop]
        if baseline_window is None:
            trajectory = _repair_by_period(xy, fps, config)
            xy = trajectory.xy
        control = control_features_from_arrays(
            data["home"][start:stop], data["away"][start:stop], xy,
            fps, config,
        )
        if baseline_window is not None:
            nearest = control.nearest_distances[:, :, 0]
            raw = np.argmin(nearest, axis=1).astype(np.int8)
            raw[~np.isfinite(nearest).any(axis=1)] = UNKNOWN
            letters = smooth(raw, int(round(baseline_window * fps)))
            labels.append(np.where(
                letters == "H", HOME, np.where(letters == "A", AWAY, DEAD)
            ).astype(np.int8))
        else:
            tags = TagFeatures(
                team_prior=np.zeros(len(xy), dtype=np.float32),
                dead_hint=np.zeros(len(xy), dtype=bool),
                pre_restart_window=np.zeros(len(xy), dtype=bool),
                restart_team=np.full(len(xy), UNKNOWN, dtype=np.int8),
            )
            evidence = build_evidence(
                control, trajectory, tags, (PITCH_LENGTH, PITCH_WIDTH), config, fps=fps
            )
            labels.append(decode(evidence, tags, config))
    return np.concatenate(labels)


def _scoring_events(data, events):
    row_by_frame = {int(frame): row for row, frame in enumerate(data["frames"])}
    teams_by_frame = {}
    for event in events:
        teams_by_frame.setdefault(event["frame"], set()).add(event["team"])
    scored = []
    seen = set()
    for event in events:
        row = row_by_frame.get(event["frame"])
        key = (event["frame"], event["team"], event["type"])
        if row is None:
            raise ValueError("event frame is outside the tracking timeline")
        if data["periods"][row] != event["period"]:
            raise ValueError("event period does not match tracking period")
        if (
            key in seen or len(teams_by_frame[event["frame"]]) != 1
            or not np.isfinite(data["ball"][row]).all()
            or not np.isfinite(data["home"][row]).all(axis=1).any()
            or not np.isfinite(data["away"][row]).all(axis=1).any()
        ):
            continue
        seen.add(key)
        scored.append({**event, "row": row})
    if not scored:
        raise ValueError("no independent event anchors available for scoring")
    return scored


def _decoded_event_metrics(labels, events):
    if not events:
        raise ValueError("event scoring requires at least one anchor")
    rows = np.asarray([event["row"] for event in events], dtype=np.int64)
    truth = np.asarray([
        HOME if event["team"] == "home" else AWAY for event in events
    ])
    correct = labels[rows] == truth
    groups = {}
    for key in ("type", "period"):
        groups[key] = {}
        for value in sorted({str(event[key]) for event in events}):
            select = np.asarray([str(event[key]) == value for event in events])
            groups[key][value] = {
                "events": int(select.sum()),
                "accuracy": float(correct[select].mean()),
            }
    return {
        "events": len(events),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "dead_at_live_anchor_fraction": float(np.mean(labels[rows] == DEAD)),
        "by_type": groups["type"],
        "by_period": groups["period"],
    }, correct


def _paired_event_delta(candidate_correct, reference_correct, events, fps):
    difference = candidate_correct.astype(float) - reference_correct.astype(float)
    block_ids = np.asarray([
        (event["period"], int(event["row"] / (60 * fps))) for event in events
    ])
    _, blocks = np.unique(block_ids, axis=0, return_inverse=True)
    totals = np.bincount(blocks, weights=difference)
    counts = np.bincount(blocks)
    rng = np.random.default_rng(2027)
    draws = rng.integers(0, len(counts), size=(2000, len(counts)))
    deltas = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "delta_percentage_points": float(difference.mean() * 100),
        "ci95_percentage_points": [
            float(value * 100) for value in np.percentile(deltas, [2.5, 97.5])
        ],
        "gained_events": int(np.count_nonzero(difference > 0)),
        "lost_events": int(np.count_nonzero(difference < 0)),
        "minute_blocks": len(counts),
    }


def _paired_gap_delta(candidate, reference):
    if candidate["mask_start_rows"] != reference["mask_start_rows"]:
        raise ValueError("paired gap comparisons require identical masks")
    result = {}
    rng = np.random.default_rng(2027)
    n_windows = len(candidate["per_window"])
    draws = rng.integers(0, n_windows, size=(2000, n_windows))
    for metric in ("recovery_fraction", "within_2m_fraction_of_masked"):
        differences = np.asarray([
            first[metric] - second[metric]
            for first, second in zip(candidate["per_window"], reference["per_window"])
        ])
        deltas = differences[draws].mean(axis=1)
        result[metric] = {
            "delta_percentage_points": float(differences.mean() * 100),
            "ci95_percentage_points": [
                float(value * 100) for value in np.percentile(deltas, [2.5, 97.5])
            ],
        }
    return result


def _period_spell_diagnostics(labels, fps, periods):
    boundaries = np.r_[0, np.flatnonzero(np.diff(periods)) + 1, len(labels)]
    pieces = [
        summarize(labels[start:stop], fps)
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]
    return {
        "possessions": sum(piece["possessions"] for piece in pieces),
        "shorter_than_one_second": sum(
            piece["shorter_than_one_second"] for piece in pieces
        ),
        "counts": {
            label: sum(piece["counts"][label] for piece in pieces)
            for label in ("H", "A", "D")
        },
    }


def _comparison_summary(results, candidates):
    summary = {}
    for candidate in candidates:
        name = candidate["name"]
        scored = [
            game["candidates"][name]["modes"]["clean"]["event_anchors"]
            for game in results
        ]
        n_events = sum(item["events"] for item in scored)
        masked_correct = sum(
            game["candidates"][name]["modes"]["masked"]["event_anchors"]["correct"]
            for game in results
        )
        spells = [
            game["candidates"][name]["modes"]["clean"]["spell_diagnostics"]
            for game in results
        ]
        summary[name] = {
            "events": n_events,
            "clean_event_accuracy": sum(item["correct"] for item in scored) / n_events,
            "masked_event_accuracy": masked_correct / n_events,
            "live_spells": sum(item["possessions"] for item in spells),
            "spells_under_one_second": sum(
                item["shorter_than_one_second"] for item in spells
            ),
        }
    return summary


def compare_metrica_candidates(root, protocol_path):
    root = Path(root).resolve()
    protocol_path = Path(protocol_path)
    protocol = load_validation_protocol(protocol_path)
    settings = protocol["metrica"]
    candidates = protocol["candidate_comparison"]["candidates"]
    reference = protocol["candidate_comparison"]["reference"]
    if reference not in {item["name"] for item in candidates}:
        raise ValueError("comparison reference must name a candidate")
    project = Path(__file__).resolve().parents[1]
    results = []
    candidate_configs = {}
    for candidate in candidates:
        path = project / candidate["config"]
        candidate_configs[candidate["name"]] = {
            "source_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_config": candidate["config"],
            "source_fps": candidate["source_fps"],
        }
    for game in settings["games"]["validation"]:
        print("Comparing candidates on {}".format(game), flush=True)
        data = _load_game(root, game)
        events = _scoring_events(data, _load_events(
            data["event_path"], set(settings["event_control_types"])
        ))
        masked = data["ball"].copy()
        duration = protocol["candidate_comparison"]["event_mask_seconds"]
        length = int(round(duration * data["fps"]))
        mask_starts = _select_windows(
            masked, length, periods=data["periods"]
        )
        for start in mask_starts:
            masked[start:start + length] = np.nan
        game_results = {}
        correctness = {}
        for candidate in candidates:
            name = candidate["name"]
            config = load_config(project / candidate["config"])
            config = config_at_fps(config, candidate["source_fps"], data["fps"])
            config["video_ball_observations_path"] = None
            config.update(candidate.get("overrides", {}))
            window = candidate.get("baseline_half_window_seconds")
            candidate_configs[name]["effective_config_at_metrica_fps"] = config
            print("  {}".format(name), flush=True)
            gap_results = [
                _gap_metrics(
                    data["ball"], data["fps"], policy["duration_seconds"],
                    config, data["periods"], repair=window is None
                ) for policy in settings["gap_checks"]
            ]
            modes = {}
            for mode, ball in (("clean", data["ball"]), ("masked", masked)):
                labels = _candidate_labels({**data, "ball": ball}, config, window)
                event_metrics, correct = _decoded_event_metrics(labels, events)
                modes[mode] = {
                    "event_anchors": event_metrics,
                    "spell_diagnostics": _period_spell_diagnostics(
                        labels, data["fps"], data["periods"]
                    ),
                    "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                }
                masked_events = [
                    event for event in events
                    if not np.isfinite(masked[event["row"]]).all()
                ]
                modes[mode]["masked_event_anchors"] = (
                    _decoded_event_metrics(labels, masked_events)[0]
                    if masked_events else {"events": 0, "accuracy": None}
                )
                correctness.setdefault(mode, {})[name] = correct
            game_results[name] = {"gaps": gap_results, "modes": modes}
        paired = {
            mode: {
                name: _paired_event_delta(
                    correct, values[reference], events, data["fps"]
                )
                for name, correct in values.items()
            } for mode, values in correctness.items()
        }
        results.append({
            "game": game,
            "fps": data["fps"],
            "frames": len(data["ball"]),
            "event_anchors": len(events),
            "mask_start_rows": mask_starts,
            "candidates": game_results,
            "paired_delta_vs_reference": paired,
            "paired_gap_delta_vs_reference": {
                name: [
                    _paired_gap_delta(first, second)
                    for first, second in zip(
                        values["gaps"], game_results[reference]["gaps"]
                    )
                ] for name, values in game_results.items()
            },
            "input_sha256": {
                path.name: _file_sha256(path)
                for path in (
                    root / "data" / game / (game + "_RawTrackingData_Home_Team.csv"),
                    root / "data" / game / (game + "_RawTrackingData_Away_Team.csv"),
                    data["event_path"],
                )
            },
        })
    return {
        "schema_version": 1,
        "dataset": "Metrica Sports sample-data",
        "protocol_sha256": _file_sha256(protocol_path),
        "reference": reference,
        "candidate_configs": candidate_configs,
        "summary": _comparison_summary(results, candidates),
        "code_sha256": {
            path.name: _file_sha256(path) for path in (
                Path(__file__), project / "possession" / "tracking.py",
                project / "possession" / "decoder.py",
                project / "possession" / "trajectory.py",
                project / "starter" / "baseline.py",
            )
        },
        "results": results,
        "limitations": [
            "Only Games 1-2; no match-generalization confidence interval.",
            "CIs resample paired one-minute blocks within each match (seed 2027).",
            "Gap CIs resample paired masked windows, not independent matches.",
            "Events are scoring-only; Smart Tagging/restart priors are unavailable.",
            "Anchor accuracy and spell counts are not possession spell F1.",
            "Legacy configs are diagnostic; challenge-specific video fusion is excluded.",
            "Decoder switches are architecture ablations, not historical checkpoints.",
            "No candidate promotion or challenge submission change is performed.",
        ],
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_metrica_recoveries(root, protocol_path):
    root = Path(root).resolve()
    protocol_path = Path(protocol_path)
    protocol = load_validation_protocol(protocol_path)
    settings = protocol["recovery_analysis"]
    candidates = protocol["candidate_comparison"]["candidates"]
    project = Path(__file__).resolve().parents[1]
    results = []
    for game in protocol["metrica"]["games"]["validation"]:
        print("Analyzing recovery transitions on {}".format(game), flush=True)
        data = _load_game(root, game)
        events = _load_events(data["event_path"], {
            "PASS", "RECOVERY", "BALL LOST", "BALL OUT", "SET PIECE",
            "FAULT RECEIVED", "SHOT",
        })
        references, counts = recovery_references(
            events, data["frames"], data["periods"], data["fps"],
            settings["loss_pair_tolerance_seconds"],
        )
        if not references:
            raise ValueError("{}: no paired recovery references".format(game))
        masked, mask_starts, length = recovery_masks(
            data["ball"], data["periods"], references, data["fps"],
            settings["mask_seconds"],
        )
        if not mask_starts:
            raise ValueError("{}: no eligible recovery-centered masks".format(game))
        targeted = [
            reference for reference in references
            if not np.isfinite(masked[reference["row"]]).all()
        ]
        targeted_rows = np.asarray([reference["row"] for reference in targeted])
        targeted_teams = np.asarray([reference["incoming"] for reference in targeted])
        correctness = {}
        game_results = {}
        for candidate in candidates:
            name = candidate["name"]
            print("  {}".format(name), flush=True)
            path = project / candidate["config"]
            config = config_at_fps(
                load_config(path), candidate["source_fps"], data["fps"]
            )
            config["video_ball_observations_path"] = None
            config.update(candidate.get("overrides", {}))
            modes = {}
            for mode, ball in (("clean", data["ball"]), ("recovery_masked", masked)):
                labels = _candidate_labels(
                    {**data, "ball": ball}, config,
                    candidate.get("baseline_half_window_seconds"),
                )
                modes[mode] = score_recovery_transitions(
                    labels, data["periods"], references, data["fps"], settings
                )
                modes[mode]["label_sha256"] = hashlib.sha256(
                    labels.tobytes()
                ).hexdigest()
                modes[mode]["masked_reference_subset"] = score_recovery_transitions(
                    labels, data["periods"], targeted, data["fps"], settings
                )
                correctness.setdefault(mode, {})[name] = (
                    labels[targeted_rows] == targeted_teams
                )
            game_results[name] = {
                "source_config_sha256": _file_sha256(path),
                "effective_config": config,
                "modes": modes,
            }
        results.append({
            "game": game,
            "fps": data["fps"],
            "reference_counts": counts,
            "references": references,
            "mask_start_rows": mask_starts,
            "mask_duration_frames": length,
            "input_sha256": {
                path.name: _file_sha256(path)
                for path in (data["event_path"],
                             root / "data" / game / (game + "_RawTrackingData_Home_Team.csv"),
                             root / "data" / game / (game + "_RawTrackingData_Away_Team.csv"))
            },
            "candidates": game_results,
            "masked_subset_paired_delta_vs_external_semi_markov": {
                mode: {
                    name: _paired_event_delta(
                        correct, values["external_semi_markov"], targeted, data["fps"]
                    )
                    for name, correct in values.items()
                }
                for mode, values in correctness.items()
            },
        })
    return {
        "schema_version": 1,
        "dataset": "Metrica Sports sample-data",
        "protocol_sha256": _file_sha256(protocol_path),
        "code_sha256": {
            path.name: _file_sha256(path) for path in (
                Path(__file__), project / "possession" / "external_possession_events.py",
                project / "possession" / "decoder.py", project / "possession" / "tracking.py",
                project / "possession" / "trajectory.py", project / "starter" / "baseline.py",
            )
        },
        "results": results,
        "limitations": [
            "Provider loss/recovery event references, not challenge-equivalent spell ground truth.",
            "Loss end and recovery start bracket boundary uncertainty; unmatched cases stay excluded.",
            "One-to-one nearest same-direction matching; unrelated predicted boundaries are not false positives.",
            "Post-recovery support is censored by known subsequent events, not continuous possession truth.",
            "Recovery-centered masks use external annotations for evaluation design only.",
            "Events never enter candidate inference; no candidate is promoted or retuned.",
            "Only Metrica Games 1-2; Game 3 and SoccerNet test are not evaluated.",
            "Paired confidence intervals resample within-match one-minute blocks, not matches.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate frozen possession geometry on Metrica data."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--protocol",
        default=str(
            Path(__file__).resolve().parents[1]
            / "configs" / "external_transfer_validation_v2.json"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--compare-candidates", action="store_true",
        help="Compare protocol-defined candidates with identical external masks.",
    )
    parser.add_argument(
        "--recovery-analysis", action="store_true",
        help="Analyze event-derived recovery timing and recovery-centered missing-ball masks.",
    )
    args = parser.parse_args(argv)
    if args.compare_candidates and args.recovery_analysis:
        parser.error("choose either --compare-candidates or --recovery-analysis")
    try:
        report = (
            analyze_metrica_recoveries(args.root, args.protocol)
            if args.recovery_analysis
            else (
                compare_metrica_candidates(args.root, args.protocol)
                if args.compare_candidates
                else validate_metrica(args.root, args.protocol)
            )
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (IOError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, "error: {}\n".format(error))
    if args.compare_candidates or args.recovery_analysis:
        print("Saved external analysis to {}".format(args.output))
        return 0
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
