from dataclasses import dataclass

import numpy as np

from possession import AWAY, HOME, UNKNOWN
from possession.tracking import contact_motion_features


UNSEEN = 0
CONTACT = 1
FLIGHT = 2
RECEPTION = 3
TURNOVER = 4
STOPPAGE = 5


@dataclass
class ControlPhase:
    kind: np.ndarray
    team: np.ndarray
    diagnostics: dict
    anchor_rows: np.ndarray = None
    incipient_contact: np.ndarray = None


def sustained_player_contacts(source, strong_control, slots, minimum):
    n = len(source)
    if (strong_control.shape != (n, 2) or not isinstance(slots, np.ndarray)
            or slots.shape != (n, 2) or not np.issubdtype(slots.dtype, np.integer)
            or not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 2):
        raise ValueError("sustained contact needs aligned player slots and a minimum of two frames")
    run = np.zeros(2, dtype=np.int32)
    previous = np.full(2, -1, dtype=np.int32)
    supported = np.zeros((n, 2), dtype=bool)
    for frame in range(n):
        eligible = (source[frame] == 1) & strong_control[frame] & (slots[frame] >= 0)
        run = np.where(eligible, np.where(slots[frame] == previous, run + 1, 1), 0)
        previous = np.where(eligible, slots[frame], -1)
        supported[frame] = run >= minimum
    return supported


def _physics_supported_short_gaps(trajectory, control_features, tags, fps, config):
    source, xy = trajectory.source, trajectory.xy
    nearest = control_features.nearest_distances[:, :, 0]
    bridge = np.zeros(len(source), dtype=bool)
    window = config["phase_contact_minimum_frames"]
    maximum = config["phase_bridge_max_frames"]
    if not maximum:
        return bridge
    boundaries = np.r_[0, np.flatnonzero(source[1:] != source[:-1]) + 1, len(source)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if source[start] != 2 or end - start > maximum:
            continue
        left, right = start - 1, end
        if (left - window + 1 < 0 or right + window > len(source)
                or not np.all(source[left - window + 1:left + 1] == 1)
                or not np.all(source[right:right + window] == 1)
                or np.any(tags.dead_hint[left:right + 1])
                or np.isin(tags.restart_team[left:right + 1], (HOME, AWAY)).any()
                or not np.all(nearest[start:end] > config["phase_contact_distance_metres"])):
            continue
        points = xy[[left - window + 1, left, right, right + window - 1]]
        if not np.isfinite(points).all():
            continue
        before = (points[1] - points[0]) * fps / (window - 1)
        across = (points[2] - points[1]) * fps / (right - left)
        after = (points[3] - points[2]) * fps / (window - 1)
        speeds = np.linalg.norm(np.stack((before, across, after)), axis=1)
        tolerance = (
            config["max_ball_acceleration_metres_per_second_squared"]
            * (right - left + window - 1) / (2 * fps)
        )
        if (not np.isfinite(speeds).all()
                or np.min(speeds) < config["phase_minimum_ball_speed_mps"]
                or np.max(speeds) > config["max_ball_speed_metres_per_second"]
                or np.dot(before, across) <= 0 or np.dot(after, across) <= 0
                or np.linalg.norm(before - across) > tolerance
                or np.linalg.norm(after - across) > tolerance):
            continue
        bridge[start:end] = True
    return bridge


def _corroborate_long_receptions(trajectory, control_features, tags, nearest,
                                 kind, team, anchors, config):
    n = len(kind)
    if (
        not isinstance(trajectory.xy, np.ndarray) or trajectory.xy.shape != (n, 2)
        or not isinstance(trajectory.velocity, np.ndarray) or trajectory.velocity.shape != (n, 2)
        or not isinstance(control_features.nearest_player_velocity, np.ndarray)
        or control_features.nearest_player_velocity.shape != (n, 2, 2)
    ):
        raise ValueError("long reception bridging needs aligned ball and same-track player motion")
    maximum = config["phase_observed_reception_bridge_max_frames"]
    minimum = config["phase_contact_minimum_frames"]
    radius = config["phase_contact_distance_metres"]
    speed_limit = config["strong_control_relative_speed_metres_per_second"]
    relative = np.linalg.norm(
        trajectory.velocity[:, None, :] - control_features.nearest_player_velocity,
        axis=2,
    )
    bridged = extended = 0
    for frame in range(1, n):
        if kind[frame] != CONTACT or (
            kind[frame - 1] == CONTACT and team[frame - 1] == team[frame]
        ):
            continue
        start = max(0, frame - maximum - minimum)
        flights = np.flatnonzero(kind[start:frame] == FLIGHT)
        if not flights.size:
            continue
        last_flight = start + int(flights[-1])
        anchor = int(anchors[last_flight])
        if (
            anchor < 0 or frame - anchor <= config["phase_flight_max_frames"] + minimum
            or frame - anchor > maximum + minimum
            or kind[anchor] not in (CONTACT, RECEPTION)
            or team[anchor] != team[frame]
            or np.isin(kind[last_flight + 1:frame], (CONTACT, RECEPTION, TURNOVER, STOPPAGE)).any()
            or not np.all(trajectory.source[anchor:frame + 1] == 1)
            or np.any(tags.dead_hint[anchor:frame + 1])
            or np.isin(tags.restart_team[anchor:frame + 1], (HOME, AWAY)).any()
        ):
            continue
        between = nearest[anchor + 1:frame]
        if (
            not np.isfinite(between).all()
            or not np.all(between[:, AWAY if team[frame] == HOME else HOME] > radius)
        ):
            continue
        same_player_motion = relative[frame - minimum + 1:frame + 1, team[frame]]
        if not np.any(np.isfinite(same_player_motion) & (same_player_motion <= speed_limit)):
            continue
        far = np.all(between > radius, axis=1)
        flight_speed = trajectory.speed[anchor + 1:frame][far]
        displacement = trajectory.xy[frame] - trajectory.xy[anchor]
        if (
            not flight_speed.size or not np.isfinite(flight_speed).all()
            or not np.all(flight_speed >= config["phase_minimum_ball_speed_mps"])
            or not np.isfinite(displacement).all()
            or np.linalg.norm(displacement) <= radius
        ):
            continue
        velocity = trajectory.velocity[anchor + 1:frame][far]
        if not np.isfinite(velocity).all() or not np.all(velocity @ displacement > 0):
            continue
        fill = np.flatnonzero(far & (kind[anchor + 1:frame] == UNSEEN)) + anchor + 1
        kind[fill] = FLIGHT
        team[fill] = team[anchor]
        anchors[fill] = anchor
        kind[frame] = RECEPTION
        bridged += 1
        extended += len(fill)
    return bridged, extended


def infer_control_phase(trajectory, control_features, tags, strong_control, config, fps=None):
    source, speed = trajectory.source, trajectory.speed
    n = len(source)
    distances = control_features.nearest_distances
    if (not isinstance(distances, np.ndarray) or distances.ndim != 3
            or distances.shape[:2] != (n, 2) or distances.shape[2] < 1
            or speed.shape != (n,)
            or tags.dead_hint.shape != (n,) or tags.restart_team.shape != (n,)):
        raise ValueError("phase inference needs aligned ball, player and restart inputs")
    nearest = distances[:, :, 0]
    contact = sustained_player_contacts(
        source, strong_control & (nearest <= config["phase_contact_distance_metres"]),
        control_features.nearest_player_slot,
        config["phase_contact_minimum_frames"],
    )
    motion_mode = config.get("phase_motion_continuity_enabled", False)
    bridges = np.zeros(n, dtype=bool)
    if motion_mode:
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("motion-supported contacts require positive physical fps")
        disturbance = contact_motion_features(
            trajectory.xy, source, nearest, fps,
            config["phase_contact_distance_metres"], 1,
            config["phase_contact_minimum_frames"],
            config["max_ball_speed_metres_per_second"],
        )["contact_motion_disturbance"]
        supported = np.isfinite(disturbance) & (
            disturbance >= config["phase_contact_motion_minimum"]
        )
        recent = np.zeros_like(contact)
        for lag in range(config["phase_contact_minimum_frames"]):
            recent[lag:] |= supported[:n - lag]
        contact &= recent
        bridges = _physics_supported_short_gaps(
            trajectory, control_features, tags, fps, config,
        )
    incipient = None
    if config.get("phase_early_opponent_veto_enabled", False):
        incipient = sustained_player_contacts(
            source,
            strong_control & (nearest <= config["phase_contact_distance_metres"]),
            control_features.nearest_player_slot,
            max(2, config["phase_contact_minimum_frames"] - 1),
        )
        incipient[
            tags.dead_hint | np.isin(tags.restart_team, (HOME, AWAY))
        ] = False
    kind = np.full(n, UNSEEN, dtype=np.int8)
    team = np.full(n, UNKNOWN, dtype=np.int8)
    anchors = np.full(n, -1, dtype=np.int32)
    owner, last_contact, was_flight = UNKNOWN, -1, False
    for frame in range(n):
        if tags.dead_hint[frame] or tags.restart_team[frame] in (HOME, AWAY):
            kind[frame] = STOPPAGE
            owner, last_contact, was_flight = UNKNOWN, -1, False
            continue
        if source[frame] != 1 and not bridges[frame]:
            owner, last_contact, was_flight = UNKNOWN, -1, False
            continue
        if contact[frame].all():
            owner, last_contact, was_flight = UNKNOWN, -1, False
            continue
        if source[frame] == 1 and contact[frame].any():
            current = int(np.argmax(contact[frame]))
            kind[frame] = (
                RECEPTION if current == owner and was_flight
                else TURNOVER if owner in (HOME, AWAY) and current != owner
                else CONTACT
            )
            team[frame] = current
            owner, last_contact, was_flight = current, frame, False
            continue
        if (
            owner in (HOME, AWAY)
            and frame - last_contact <= config["phase_flight_max_frames"]
            and np.isfinite(speed[frame])
            and speed[frame] >= config["phase_minimum_ball_speed_mps"]
            and np.isfinite(nearest[frame, owner])
            and nearest[frame, owner] > config["phase_contact_distance_metres"]
            and np.isfinite(nearest[frame, AWAY if owner == HOME else HOME])
            and nearest[frame, AWAY if owner == HOME else HOME] > config["phase_contact_distance_metres"]
        ):
            kind[frame] = FLIGHT
            team[frame] = owner
            anchors[frame] = last_contact
            was_flight = True
        elif frame - last_contact > config["phase_flight_max_frames"]:
            owner, last_contact, was_flight = UNKNOWN, -1, False
    bridged = extended = 0
    if config.get("phase_observed_reception_bridge_enabled", False):
        bridged, extended = _corroborate_long_receptions(
            trajectory, control_features, tags, nearest, kind, team, anchors, config,
        )
    diagnostics = {
        name: int(np.count_nonzero(kind == code))
        for name, code in (
            ("unseen", UNSEEN), ("contact", CONTACT), ("flight", FLIGHT),
            ("reception", RECEPTION), ("turnover", TURNOVER), ("stoppage", STOPPAGE),
        )
    }
    if motion_mode:
        diagnostics["motion_supported_contact_frames"] = int(np.count_nonzero(contact))
        diagnostics["physics_qualified_gap_frames"] = int(np.count_nonzero(bridges))
        diagnostics["physics_bridge_frames"] = int(np.count_nonzero(
            bridges & (kind == FLIGHT)
        ))
    if config.get("phase_observed_reception_bridge_enabled", False):
        diagnostics["observed_long_receptions"] = bridged
        diagnostics["observed_long_flight_frames"] = extended
    return ControlPhase(
        kind=kind, team=team, diagnostics=diagnostics, anchor_rows=anchors,
        incipient_contact=incipient,
    )
