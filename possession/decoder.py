from dataclasses import dataclass, replace

import numpy as np

from possession.tracking import TEAM_X_RANKS, contact_motion_features
from possession.control_phase import (
    CONTACT, FLIGHT, RECEPTION, STOPPAGE, TURNOVER, UNSEEN,
    ControlPhase, infer_control_phase,
)

from possession import AWAY, DEAD, HOME, UNKNOWN

# Fixed-label value leaving a frame unconstrained (UNKNOWN there means live with a free team).
FREE = -2


CORE_CALIBRATION_KEYS = (
    "closeness_scale_metres", "nearby_two_weight", "nearby_four_weight",
    "semi_markov_control_margin_weight", "semi_markov_live_transition_cost",
    "semi_markov_strong_transition_cost", "semi_markov_strong_transition_window_frames",
    "semi_markov_strong_transition_minimum_frames", "strong_control_distance_metres",
    "strong_control_distance_margin_metres",
)
CORE_RELIABILITY_KEYS = (
    "core_reliability_mode", "core_uncertain_closeness_scale_metres",
    "core_uncertain_nearby_two_weight", "core_uncertain_nearby_four_weight",
    "core_uncertain_live_transition_cost", "core_uncertain_strong_transition_cost",
    "core_uncertain_strong_transition_window_frames",
    "core_uncertain_strong_transition_minimum_frames",
    "core_uncertain_strong_control_distance_metres",
    "core_uncertain_strong_control_distance_margin_metres",
)


@dataclass
class Evidence:
    control: np.ndarray
    dead: np.ndarray
    strong_control: np.ndarray = None
    handover_control: np.ndarray = None
    observed_handover_support: np.ndarray = None
    gap_handover_control: np.ndarray = None
    gap_handover_context: np.ndarray = None
    reliable_observation: np.ndarray = None
    uncertain_strong_control: np.ndarray = None
    touch_motion_disturbance: np.ndarray = None
    owner_flight_prior: np.ndarray = None
    owner_flight_diagnostics: dict = None
    owner_flight_anchor_rows: np.ndarray = None
    player_reliability_diagnostics: dict = None
    control_phase: ControlPhase = None
    exit_bracket_diagnostics: dict = None
    restart_dead_diagnostics: dict = None
    contact_carry: np.ndarray = None
    contact_carry_support: np.ndarray = None
    contact_carry_diagnostics: dict = None
    broadcast_stoppage_diagnostics: dict = None
    ballfree_penalty: np.ndarray = None
    ballfree_support: np.ndarray = None
    forced_dead: np.ndarray = None


def validate_observed_handover_config(config):
    enabled = config.get("semi_markov_observed_handover_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("semi_markov_observed_handover_enabled must be boolean")
    if not enabled:
        return
    if (config.get("decoder_mode") != "semi_markov"
            or any(config.get(key, False) for key in (
                "semi_markov_local_handover_enabled",
                "semi_markov_gap_handover_enabled",
                "semi_markov_touch_motion_gate_enabled",
                "semi_markov_owner_flight_enabled",
                "semi_markov_phase_enabled",
            ))):
        raise ValueError("observed handover needs standalone semi-Markov decoding")
    window = config.get("observed_handover_motion_window_frames")
    minimum = config.get("observed_handover_motion_minimum")
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("observed handover motion window must be a positive frame count")
    if (not isinstance(minimum, (int, float)) or isinstance(minimum, bool)
            or not np.isfinite(minimum) or not 0 < minimum <= 1):
        raise ValueError("observed handover motion minimum must be in (0, 1]")


def validate_control_phase_config(config):
    enabled = config.get("semi_markov_phase_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("semi_markov_phase_enabled must be boolean")
    veto = config.get("phase_early_opponent_veto_enabled", False)
    if not isinstance(veto, bool) or (veto and not enabled):
        raise ValueError("early opponent veto requires enabled phases and a boolean flag")
    preserve_switch = config.get("phase_preserve_supported_switch_enabled", False)
    if not isinstance(preserve_switch, bool) or (preserve_switch and not enabled):
        raise ValueError("supported switch preservation requires enabled phases and a boolean flag")
    motion_mode = config.get("phase_motion_continuity_enabled", False)
    if not isinstance(motion_mode, bool) or (motion_mode and not enabled):
        raise ValueError("motion continuity requires enabled phases and a boolean flag")
    if not enabled:
        return
    if config.get("decoder_mode") != "semi_markov":
        raise ValueError("control phases require semi-Markov decoding")
    if config.get("semi_markov_owner_flight_enabled", False):
        raise ValueError("control phases cannot combine with owner-flight continuity")
    if not isinstance(config.get("phase_confirm_decoded_owner", False), bool):
        raise ValueError("phase_confirm_decoded_owner must be boolean")
    phase_mode = config.get("phase_decode_mode", "global")
    if phase_mode not in ("global", "bounded"):
        raise ValueError("phase_decode_mode must be global or bounded")
    if phase_mode == "bounded" and config.get("phase_confirm_decoded_owner", False):
        raise ValueError("bounded phases already require agreement with the decoded contact owner")
    if veto and phase_mode != "bounded":
        raise ValueError("early opponent veto requires bounded phase decoding")
    if preserve_switch and phase_mode != "bounded":
        raise ValueError("supported switch preservation requires bounded phase decoding")
    if motion_mode:
        if phase_mode != "bounded" or veto or preserve_switch:
            raise ValueError("motion continuity requires bounded phases without other phase safeguards")
        minimum = config.get("phase_contact_motion_minimum")
        maximum = config.get("phase_bridge_max_frames")
        if (not isinstance(minimum, (int, float)) or isinstance(minimum, bool)
                or not np.isfinite(minimum) or not 0 < minimum <= 1):
            raise ValueError("phase contact motion minimum must be within (0, 1]")
        if (not isinstance(maximum, int) or isinstance(maximum, bool)
                or not 0 <= maximum <= config.get("phase_flight_max_frames", 0)):
            raise ValueError("phase bridge maximum must be within the flight horizon")
        for key in ("max_ball_speed_metres_per_second",
                    "max_ball_acceleration_metres_per_second_squared"):
            value = config.get(key)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError("{} must be finite and positive".format(key))
    bridge_enabled = config.get("phase_observed_reception_bridge_enabled", False)
    if not isinstance(bridge_enabled, bool):
        raise ValueError("phase_observed_reception_bridge_enabled must be boolean")
    if bridge_enabled:
        if motion_mode:
            raise ValueError("motion continuity cannot combine with long reception bridging")
        maximum = config.get("phase_observed_reception_bridge_max_frames")
        if phase_mode != "bounded":
            raise ValueError("observed reception bridging requires bounded decoding")
        if (
            not isinstance(maximum, int) or isinstance(maximum, bool)
            or maximum <= config.get("phase_flight_max_frames", 0)
        ):
            raise ValueError("observed reception bridge needs a longer positive frame horizon")
        motion = config.get("strong_control_relative_speed_metres_per_second")
        if (
            not isinstance(motion, (int, float)) or isinstance(motion, bool)
            or not np.isfinite(motion) or motion <= 0
        ):
            raise ValueError("observed reception bridge needs finite positive same-track motion limit")
    required = (
        "phase_contact_minimum_frames", "phase_flight_max_frames",
        "phase_minimum_ball_speed_mps", "phase_contact_distance_metres",
        "phase_flight_weight", "phase_turnover_transition_cost",
    )
    missing = sorted(set(required) - set(config))
    if missing:
        raise ValueError("control phase configuration is missing {}".format(missing))
    for key in required[:2]:
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("{} must be a positive integer".format(key))
    if config["phase_contact_minimum_frames"] < 2:
        raise ValueError("phase contact needs at least two frames")
    for key in required[2:]:
        value = config[key]
        if (
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not np.isfinite(value) or value < 0
        ):
            raise ValueError("{} must be finite and nonnegative".format(key))
    if (config["phase_minimum_ball_speed_mps"] == 0
            or config["phase_contact_distance_metres"] == 0
            or config["phase_turnover_transition_cost"] > config["semi_markov_live_transition_cost"]):
        raise ValueError("control phase speed/distance/cost is invalid")


def _validated_control_phase(evidence):
    phase = evidence.control_phase
    n = len(evidence.dead)
    if (
        not isinstance(phase, ControlPhase)
        or not isinstance(phase.kind, np.ndarray) or phase.kind.shape != (n,)
        or not isinstance(phase.team, np.ndarray) or phase.team.shape != (n,)
        or not isinstance(phase.anchor_rows, np.ndarray) or phase.anchor_rows.shape != (n,)
        or not np.issubdtype(phase.kind.dtype, np.integer)
        or not np.issubdtype(phase.team.dtype, np.integer)
        or not np.issubdtype(phase.anchor_rows.dtype, np.integer)
        or not np.isin(phase.kind, (UNSEEN, CONTACT, FLIGHT, RECEPTION, TURNOVER, STOPPAGE)).all()
        or np.any(np.isin(phase.kind, (CONTACT, FLIGHT, RECEPTION, TURNOVER))
                  & ~np.isin(phase.team, (HOME, AWAY)))
        or np.any(np.isin(phase.kind, (UNSEEN, STOPPAGE)) & (phase.team != UNKNOWN))
        or np.any((phase.kind == FLIGHT) & (
            (phase.anchor_rows < 0) | (phase.anchor_rows >= np.arange(n))
        ))
        or np.any((phase.kind != FLIGHT) & (phase.anchor_rows != -1))
    ):
        raise ValueError("phase decoding requires aligned, valid control phase and team arrays")
    return phase


def validate_player_reliability_config(config):
    enabled = config.get("player_reliability_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("player_reliability_enabled must be boolean")
    if not enabled:
        return
    minimum = config.get("player_reliability_minimum_age_frames")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 2:
        raise ValueError("player reliability requires a minimum age of at least two frames")
    if config.get("decoder_mode") != "semi_markov":
        raise ValueError("player reliability requires semi-Markov decoding")
    resume = config.get("player_reliability_resume_gap_frames", 0)
    if not isinstance(resume, int) or isinstance(resume, bool) or resume < 0:
        raise ValueError("player reliability resume gap must be a nonnegative integer")


def validate_owner_flight_config(config):
    enabled = config.get("semi_markov_owner_flight_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("semi_markov_owner_flight_enabled must be boolean")
    if not enabled:
        return
    if config.get("decoder_mode") != "semi_markov":
        raise ValueError("owner-flight continuity requires semi-Markov decoding")
    required = ("owner_flight_horizon_frames", "owner_flight_weight",
                "owner_flight_minimum_ball_speed_mps", "owner_flight_touch_motion_minimum",
                "touch_motion_window_frames")
    missing = sorted(set(required) - set(config))
    if missing:
        raise ValueError("owner-flight configuration is missing {}".format(missing))
    horizon = config["owner_flight_horizon_frames"]
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("owner-flight horizon must be a positive integer")
    motion_window = config["touch_motion_window_frames"]
    if not isinstance(motion_window, int) or isinstance(motion_window, bool) or motion_window < 1:
        raise ValueError("owner-flight motion window must be a positive integer")
    for key in required[1:-1]:
        value = config[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError("{} must be finite and positive".format(key))
    if config["owner_flight_touch_motion_minimum"] > 1:
        raise ValueError("owner-flight touch motion must not exceed one")
    if not isinstance(config.get("owner_flight_reset_on_gap", False), bool):
        raise ValueError("owner_flight_reset_on_gap must be boolean")
    if not isinstance(config.get("owner_flight_confirm_decoded_owner", False), bool):
        raise ValueError("owner_flight_confirm_decoded_owner must be boolean")
    if not isinstance(config.get("owner_flight_require_sustained_player", False), bool):
        raise ValueError("owner_flight_require_sustained_player must be boolean")
    if not isinstance(config.get("owner_flight_require_same_team_reception", False), bool):
        raise ValueError("owner_flight_require_same_team_reception must be boolean")
    if (config.get("owner_flight_require_same_team_reception", False)
            and not config.get("owner_flight_reset_on_gap", False)):
        raise ValueError("same-team reception requires gap reset")
    if config.get("owner_flight_require_sustained_player", False):
        minimum = config.get("owner_flight_contact_minimum_frames")
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 2:
            raise ValueError("owner-flight player contact requires at least two consecutive frames")
    if config.get("semi_markov_local_handover_enabled", False) or config.get("semi_markov_gap_handover_enabled", False):
        raise ValueError("owner-flight continuity cannot combine with local/gap handover experiments")


def owner_flight_evidence(trajectory, control_features, tags, strong_control, fps, config):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("owner-flight evidence requires positive physical fps")
    n = len(trajectory.source)
    nearest = control_features.nearest_distances[:, :, 0]
    if (nearest.shape != (n, 2) or strong_control.shape != (n, 2)
            or trajectory.xy.shape != (n, 2) or trajectory.speed.shape != (n,)
            or tags.dead_hint.shape != (n,) or tags.restart_team.shape != (n,)):
        raise ValueError("owner-flight evidence requires aligned tracking and restart arrays")
    disturbance = contact_motion_features(
        trajectory.xy, trajectory.source, nearest, fps,
        config["strong_control_distance_metres"], 1,
        config["touch_motion_window_frames"], config["max_ball_speed_metres_per_second"],
    )["contact_motion_disturbance"]
    touch = (
        (trajectory.source == 1)[:, None] & strong_control
        & np.isfinite(disturbance)
        & (disturbance >= config["owner_flight_touch_motion_minimum"])
    )
    if config.get("owner_flight_require_sustained_player", False):
        slots = control_features.nearest_player_slot
        if (not isinstance(slots, np.ndarray) or slots.shape != (n, 2)):
            raise ValueError("sustained owner contact requires stable nearest player slot IDs")
        run = np.zeros(2, dtype=np.int32)
        previous = np.full(2, -1, dtype=np.int32)
        sustained = np.zeros_like(touch)
        minimum = config["owner_flight_contact_minimum_frames"]
        for frame in range(n):
            eligible = (trajectory.source[frame] == 1) & strong_control[frame] & (slots[frame] >= 0)
            run = np.where(eligible, np.where(slots[frame] == previous, run + 1, 1), 0)
            previous = np.where(eligible, slots[frame], -1)
            sustained[frame] = run >= minimum
        touch &= sustained
    prior = np.zeros((n, 2), dtype=np.float32)
    anchor_rows = np.full(n, -1, dtype=np.int64)
    owner, last_contact = UNKNOWN, -1
    ambiguous, anchors, resets, gap_resets = 0, 0, 0, 0
    for frame in range(n):
        if tags.dead_hint[frame] or tags.restart_team[frame] in (HOME, AWAY):
            owner, last_contact = UNKNOWN, -1
            resets += 1
            continue
        if config.get("owner_flight_reset_on_gap", False) and trajectory.source[frame] != 1:
            if owner != UNKNOWN:
                gap_resets += 1
            owner, last_contact = UNKNOWN, -1
            continue
        if touch[frame].all():
            owner, last_contact = UNKNOWN, -1
            ambiguous += 1
            continue
        if touch[frame].any():
            owner = int(np.argmax(touch[frame]))
            last_contact = frame
            anchors += 1
            continue
        if owner == UNKNOWN or frame - last_contact > config["owner_flight_horizon_frames"]:
            owner = UNKNOWN
            continue
        if (
            trajectory.source[frame] == 1
            and np.isfinite(trajectory.speed[frame])
            and trajectory.speed[frame] >= config["owner_flight_minimum_ball_speed_mps"]
            and nearest[frame, owner] > config["strong_control_distance_metres"]
            and np.isfinite(nearest[frame, owner])
        ):
            prior[frame, owner] = config["owner_flight_weight"]
            anchor_rows[frame] = last_contact
    rejected = 0
    if config.get("owner_flight_require_same_team_reception", False):
        next_team, next_row = UNKNOWN, -1
        for frame in range(n - 1, -1, -1):
            if (tags.dead_hint[frame] or tags.restart_team[frame] in (HOME, AWAY)
                    or trajectory.source[frame] != 1 or touch[frame].all()):
                next_team, next_row = UNKNOWN, -1
            elif touch[frame].any():
                next_team, next_row = int(np.argmax(touch[frame])), frame
            elif prior[frame].any():
                held = int(np.argmax(prior[frame]))
                if next_team != held or next_row - frame > config["owner_flight_horizon_frames"]:
                    prior[frame, held] = 0
                    anchor_rows[frame] = -1
                    rejected += 1
    return prior, {
        "trusted_contact_frames": anchors,
        "ambiguous_contact_frames": ambiguous,
        "restart_or_dead_reset_frames": resets,
        "unobserved_gap_resets": gap_resets,
        "unconfirmed_flight_prior_frames": rejected,
        "owner_flight_prior_frames": int(np.count_nonzero(prior)),
        "owner_flight_observed_frames": int(np.count_nonzero(trajectory.source == 1)),
    }, anchor_rows


def validate_touch_motion_config(config):
    enabled = config.get("semi_markov_touch_motion_gate_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("semi_markov_touch_motion_gate_enabled must be boolean")
    if not enabled:
        return
    if config.get("decoder_mode") != "semi_markov":
        raise ValueError("touch-motion gating requires semi-Markov decoding")
    mode = config.get("touch_motion_mode", "gate")
    if mode not in ("gate", "soften", "confidence"):
        raise ValueError("touch_motion_mode must be gate, soften or confidence")
    if config.get("semi_markov_local_handover_enabled", False) or config.get("semi_markov_gap_handover_enabled", False):
        raise ValueError("touch-motion gating cannot be combined with local/gap handover experiments")
    required = ("touch_motion_horizon_frames", "touch_motion_window_frames", "touch_motion_minimum")
    missing = sorted(set(required) - set(config))
    if missing:
        raise ValueError("touch-motion configuration is missing {}".format(missing))
    for key in required[:2]:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError("{} must be a positive integer".format(key))
    minimum = config["touch_motion_minimum"]
    if (not isinstance(minimum, (int, float)) or isinstance(minimum, bool)
            or not np.isfinite(minimum) or not 0 <= minimum <= 1):
        raise ValueError("touch_motion_minimum must be finite and within [0, 1]")
    if mode == "confidence" and minimum <= 0:
        raise ValueError("touch-motion confidence scale must be positive")
    if mode in ("soften", "confidence"):
        cost = config.get("touch_motion_weak_transition_cost")
        if (not isinstance(cost, (int, float)) or isinstance(cost, bool) or not np.isfinite(cost)
                or not config["semi_markov_strong_transition_cost"] <= cost
                <= config["semi_markov_live_transition_cost"]):
            raise ValueError("touch-motion weak cost must lie between strong and regular live costs")


def _touch_motion_evidence(evidence):
    observed, motion = evidence.reliable_observation, evidence.touch_motion_disturbance
    n = len(evidence.dead)
    if (not isinstance(observed, np.ndarray) or not isinstance(motion, np.ndarray) or observed.shape != (n,)
            or motion.shape != (n, 2) or observed.dtype != np.bool_):
        raise ValueError("touch-motion gating requires aligned observation and motion evidence")
    if np.isinf(motion).any() or np.any(np.isfinite(motion) & ((motion < 0) | (motion > 1))):
        raise ValueError("touch-motion evidence must be within [0, 1] or unknown")
    return observed, motion


def touch_motion_summary(evidence, config):
    observed, motion = _touch_motion_evidence(evidence)
    available = observed[:, None] & np.isfinite(motion)
    count = int(2 * np.count_nonzero(observed))
    known = int(np.count_nonzero(available))
    return {
        "observed_team_frames": count,
        "available_team_frames": known,
        "unknown_team_frames": count - known,
        "low_motion_team_frames": int(np.count_nonzero(
            available & (motion < config["touch_motion_minimum"])
        )),
    }


def validate_core_reliability_config(config):
    mode = config.get("core_reliability_mode", "none")
    if mode not in ("none", "geometry", "transitions", "both"):
        raise ValueError("core reliability mode must be none, geometry, transitions, or both")
    if mode == "none":
        return
    if mode in ("transitions", "both") and config.get("decoder_mode") != "semi_markov":
        raise ValueError("transition reliability fallback requires semi-Markov decoding")
    if config.get("semi_markov_local_handover_enabled", False) or config.get("semi_markov_gap_handover_enabled", False):
        raise ValueError("core reliability cannot be combined with local/gap handover experiments")
    required = set()
    if mode in ("geometry", "both"):
        required.update(CORE_RELIABILITY_KEYS[1:4])
    if mode in ("transitions", "both"):
        required.update(CORE_RELIABILITY_KEYS[4:])
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("core reliability configuration is missing {}".format(missing))
    for key in required:
        value = config[key]
        if (
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not np.isfinite(value) or value < 0
        ):
            raise ValueError("{} must be finite and nonnegative".format(key))
    if mode in ("geometry", "both") and config["core_uncertain_closeness_scale_metres"] <= 0:
        raise ValueError("uncertain closeness scale must be positive")
    if mode in ("transitions", "both"):
        window = config["core_uncertain_strong_transition_window_frames"]
        minimum = config["core_uncertain_strong_transition_minimum_frames"]
        if not isinstance(window, int) or not isinstance(minimum, int) or not 1 <= minimum <= window:
            raise ValueError("uncertain support requires 1 <= minimum <= window frames")
        if config["core_uncertain_strong_transition_cost"] > config["core_uncertain_live_transition_cost"]:
            raise ValueError("uncertain strong cost must not exceed live transition cost")
        if config["core_uncertain_strong_control_distance_metres"] <= 0:
            raise ValueError("uncertain strong-control distance must be positive")


def validate_local_handover_config(config):
    enabled = config.get("semi_markov_local_handover_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("semi_markov_local_handover_enabled must be boolean")
    if not enabled:
        return
    required = {
        "semi_markov_local_handover_window_frames",
        "semi_markov_local_handover_minimum_frames",
        "semi_markov_local_handover_distance_metres",
        "semi_markov_local_handover_distance_margin_metres",
        "semi_markov_local_handover_minimum_confidence",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("local handover configuration is missing {}".format(missing))
    window = config["semi_markov_local_handover_window_frames"]
    minimum = config["semi_markov_local_handover_minimum_frames"]
    if (
        not isinstance(window, int) or not isinstance(minimum, int)
        or not 1 <= minimum <= window
    ):
        raise ValueError("local handover requires 1 <= minimum frames <= window frames")
    radius = config["semi_markov_local_handover_distance_metres"]
    margin = config["semi_markov_local_handover_distance_margin_metres"]
    confidence = config["semi_markov_local_handover_minimum_confidence"]
    if (
        not np.isfinite([radius, margin, confidence]).all()
        or radius <= 0 or margin < 0 or not 0 <= confidence <= 1
    ):
        raise ValueError("local handover distance, margin, or confidence is invalid")


def validate_tracking_stoppage_config(config):
    enabled = config.get("tracking_stoppage_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("tracking_stoppage_enabled must be boolean")
    if not enabled:
        return
    required = {
        "tracking_stoppage_minimum_missing_frames",
        "tracking_stoppage_motion_window_frames",
        "tracking_stoppage_minimum_slow_fraction",
        "tracking_stoppage_maximum_median_speed_mps",
        "tracking_stoppage_minimum_stationary_fraction",
        "tracking_stoppage_dead_weight",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("tracking stoppage configuration is missing {}".format(missing))
    for key in (
        "tracking_stoppage_minimum_missing_frames",
        "tracking_stoppage_motion_window_frames",
    ):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError("{} must be a positive frame count".format(key))
    for key in (
        "tracking_stoppage_minimum_slow_fraction",
        "tracking_stoppage_minimum_stationary_fraction",
        "tracking_stoppage_dead_weight",
    ):
        if not np.isfinite(config[key]) or not 0 < config[key] <= 1:
            raise ValueError("{} must be in (0, 1]".format(key))
    speed = config["tracking_stoppage_maximum_median_speed_mps"]
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("tracking stoppage speed must be finite and positive")
    corroborated = config.get("tracking_stoppage_require_corroboration", False)
    if not isinstance(corroborated, bool):
        raise ValueError("tracking_stoppage_require_corroboration must be boolean")
    if corroborated:
        for key in (
            "tracking_stoppage_corroboration_lookback_frames",
            "tracking_stoppage_continuity_max_gap_frames",
            "tracking_stoppage_exit_minimum_observations",
        ):
            value = config.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("{} must be a positive integer".format(key))
        if config["tracking_stoppage_exit_minimum_observations"] < 2:
            raise ValueError("exit corroboration requires at least two observations")


def validate_tracking_exit_bracket_config(config):
    enabled = config.get("tracking_exit_bracket_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("tracking_exit_bracket_enabled must be boolean")
    if not enabled:
        return
    minimum = config.get("tracking_exit_bracket_minimum_frames")
    maximum = config.get("tracking_exit_bracket_maximum_frames")
    speed = config.get("tracking_exit_bracket_minimum_outward_speed_mps")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 2:
        raise ValueError("tracking exit bracket needs at least two frames")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= minimum:
        raise ValueError("tracking exit bracket needs a finite maximum duration above its minimum")
    if (
        not isinstance(speed, (int, float)) or isinstance(speed, bool)
        or not np.isfinite(speed) or speed <= 0
    ):
        raise ValueError("tracking exit bracket needs positive finite outward ball speed")
    weight = config.get("outside_pitch_dead_weight")
    margin = config.get("pitch_margin_metres")
    if (
        not isinstance(weight, (int, float)) or isinstance(weight, bool)
        or not np.isfinite(weight) or weight <= 0
        or not isinstance(margin, (int, float)) or isinstance(margin, bool)
        or not np.isfinite(margin) or margin <= 0
    ):
        raise ValueError("tracking exit bracket requires positive outside-pitch dead weight")


def tracking_exit_brackets(trajectory, pitch, fps, config):
    validate_tracking_exit_bracket_config(config)
    source, xy = trajectory.source, trajectory.xy
    n = len(source)
    if (
        not config.get("tracking_exit_bracket_enabled", False)
        or not isinstance(xy, np.ndarray) or xy.shape != (n, 2)
        or not np.isfinite(fps) or fps <= 0
        or not np.isfinite(pitch).all() or np.any(np.asarray(pitch) <= 0)
    ):
        raise ValueError("tracking exit brackets require enabled, aligned physical tracking")
    if np.any((source == 1) & ~np.isfinite(xy).all(axis=1)):
        raise ValueError("observed ball coordinates must be finite for tracking exit brackets")
    distances = np.stack((
        xy[:, 0], pitch[0] - xy[:, 0],
        xy[:, 1], pitch[1] - xy[:, 1],
    ), axis=1)
    observed = source == 1
    outside = np.min(distances, axis=1) < 0
    exits = np.flatnonzero(
        observed[1:] & observed[:-1] & ~outside[:-1] & outside[1:]
    ) + 1
    inside_rows = np.flatnonzero(observed & ~outside)
    bracket = np.zeros(n, dtype=bool)
    accepted = 0
    for row in exits:
        edge = int(np.argmin(distances[row]))
        direction = -1 if edge in (0, 2) else 1
        outward_speed = direction * (xy[row, edge // 2] - xy[row - 1, edge // 2]) * fps
        if outward_speed < config["tracking_exit_bracket_minimum_outward_speed_mps"]:
            continue
        next_index = np.searchsorted(inside_rows, row + 1)
        if next_index == len(inside_rows):
            continue
        stop = int(inside_rows[next_index])
        if not (
            config["tracking_exit_bracket_minimum_frames"]
            <= stop - row <= config["tracking_exit_bracket_maximum_frames"]
        ):
            continue
        if not np.any(
            observed[row:stop]
            & (np.min(distances[row:stop], axis=1) < -config["pitch_margin_metres"])
        ):
            continue
        bracket[row:stop] = True
        accepted += 1
    return bracket, {
        "observed_border_crossings": int(len(exits)),
        "qualified_exit_to_return_intervals": accepted,
        "bracketed_frames": int(np.count_nonzero(bracket)),
    }


def validate_restart_dead_config(config):
    kickoff = config.get("tracking_kickoff_dead_enabled", False)
    restart = config.get("tracking_restart_extension_enabled", False)
    if not all(isinstance(flag, bool) for flag in (kickoff, restart)):
        raise ValueError("kickoff and restart dead-ball flags must be boolean")
    integers, positives = [], []
    if kickoff:
        integers += [
            ("tracking_kickoff_formation_minimum_frames", 1),
            ("tracking_kickoff_minimum_players", 2),
            ("tracking_kickoff_goal_lookback_frames", 1),
            ("tracking_kickoff_period_start_maximum_frames", 0),
            ("tracking_kickoff_maximum_wrong_side_players", 0),
            ("tracking_kickoff_merge_gap_frames", 0),
            ("tracking_kickoff_corroboration_frames", 1),
        ]
        positives += [
            "tracking_kickoff_halfway_tolerance_metres",
            "tracking_kickoff_centre_radius_metres",
            "tracking_kickoff_goal_mouth_half_width_metres",
        ]
    if restart:
        integers += [("tracking_restart_maximum_frames", 1), ("tracking_restart_bridge_frames", 0)]
        positives += [
            "tracking_restart_touchline_distance_metres",
            "tracking_restart_goal_area_margin_metres",
        ]
    for key, minimum in integers:
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError("{} must be an integer of at least {}".format(key, minimum))
    for key in positives:
        value = config.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not np.isfinite(value) or value <= 0):
            raise ValueError("{} must be finite and positive".format(key))
    if (kickoff or restart) and (
        not isinstance(config.get("outside_pitch_dead_weight"), (int, float))
        or not np.isfinite(config["outside_pitch_dead_weight"])
        or config["outside_pitch_dead_weight"] <= 0
    ):
        raise ValueError("restart dead-ball cues require positive outside-pitch dead weight")
    if kickoff and config["tracking_kickoff_maximum_wrong_side_players"] >= TEAM_X_RANKS:
        raise ValueError("kickoff wrong-side tolerance exceeds tracked x order statistics")
    pre_kickoff = config.get("pre_kickoff_forced_dead_enabled", False)
    if not isinstance(pre_kickoff, bool):
        raise ValueError("pre_kickoff_forced_dead_enabled must be boolean")
    if pre_kickoff and (
        not kickoff or config.get("decoder_mode") != "semi_markov"
        or (config.get("semi_markov_phase_enabled", False)
            and config.get("phase_decode_mode", "global") == "bounded")
    ):
        raise ValueError("pre-kickoff forced dead needs kickoff detection and global semi-Markov decoding")


def _episodes(mask, minimum):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= minimum]


def kickoff_dead_intervals(trajectory, control_features, tag_features, pitch, config,
                           period_starts=(), return_kickoffs=False):
    # Before every kickoff the ball is out of play: half starts and post-goal restarts.
    # period_starts are only for periods known to include pre-kickoff frames.
    length, width = pitch
    n = len(trajectory.source)
    extremes = control_features.team_x_extremes
    counts = control_features.player_counts
    wrong_side = config["tracking_kickoff_maximum_wrong_side_players"]
    if (not isinstance(extremes, np.ndarray) or extremes.ndim != 4
            or extremes.shape[:3] != (n, 2, 2) or extremes.shape[3] <= wrong_side
            or counts.shape != (n, 2)):
        raise ValueError("kickoff detection needs aligned team x order statistics and player counts")
    observed = trajectory.source == 1
    xy = trajectory.xy
    middle, tolerance = length / 2, config["tracking_kickoff_halfway_tolerance_metres"]
    lowest, highest = extremes[:, :, 0, wrong_side], extremes[:, :, 1, wrong_side]
    with np.errstate(invalid="ignore"):
        home_left = (highest[:, HOME] <= middle + tolerance) & (lowest[:, AWAY] >= middle - tolerance)
        away_left = (highest[:, AWAY] <= middle + tolerance) & (lowest[:, HOME] >= middle - tolerance)
    formation = (
        (counts.sum(axis=1) >= config["tracking_kickoff_minimum_players"])
        & (home_left | away_left)
    )
    merge = config["tracking_kickoff_merge_gap_frames"]
    merged = []
    for first, end in _episodes(formation, 1):
        if merged and first - merged[-1][1] <= merge:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((first, end))
    centre_distance = np.linalg.norm(xy - np.array([middle, width / 2]), axis=1)
    centre_ball = observed & (centre_distance <= config["tracking_kickoff_centre_radius_metres"])
    window = config["tracking_kickoff_corroboration_frames"]
    types = tuple(tag_features.restart_types or ())
    tagged = np.asarray([
        event[0] for event, kind in zip(tag_features.restart_events, types)
        if "kick off" in str(kind).lower()
    ], dtype=np.int64)
    goal_line = observed & ((xy[:, 0] < 0) | (xy[:, 0] > length)) & (
        np.abs(xy[:, 1] - width / 2) <= config["tracking_kickoff_goal_mouth_half_width_metres"]
    )
    goal_entries = np.flatnonzero(goal_line[1:] & observed[:-1] & ~goal_line[:-1]) + 1
    starts = np.asarray(sorted(int(start) for start in period_starts), dtype=np.int64)
    dead = np.zeros(n, dtype=bool)
    kickoffs = []
    used_periods = set()
    diagnostics = {"kickoff_formations": 0, "kickoffs_tag_corroborated": 0,
                   "kickoffs_ball_corroborated": 0, "kickoffs_from_period_start": 0,
                   "kickoffs_after_goal_mouth_exit": 0, "kickoff_dead_frames": 0}
    for first, kickoff in merged:
        if kickoff - first < config["tracking_kickoff_formation_minimum_frames"]:
            continue
        by_tag = bool(np.any((tagged >= first - window) & (tagged <= kickoff + window)))
        by_ball = bool(centre_ball[max(0, kickoff - window):min(n, kickoff + window)].any())
        if not (by_tag or by_ball):
            continue
        diagnostics["kickoffs_tag_corroborated"] += int(by_tag)
        diagnostics["kickoffs_ball_corroborated"] += int(by_ball and not by_tag)
        start = first
        period = int(np.searchsorted(starts, first, side="right")) - 1
        if (period >= 0 and period not in used_periods
                and first - starts[period] <= config["tracking_kickoff_period_start_maximum_frames"]):
            start = int(starts[period])
            diagnostics["kickoffs_from_period_start"] += 1
        else:
            earlier = goal_entries[(goal_entries < first)
                                   & (goal_entries >= first - config["tracking_kickoff_goal_lookback_frames"])]
            if period >= 0:
                earlier = earlier[earlier >= starts[period]]
            if earlier.size:
                start = int(earlier[-1])
                diagnostics["kickoffs_after_goal_mouth_exit"] += 1
        if period >= 0:
            used_periods.add(period)
        dead[start:kickoff] = True
        kickoffs.append(int(kickoff))
        diagnostics["kickoff_formations"] += 1
    diagnostics["kickoff_dead_frames"] = int(np.count_nonzero(dead))
    if return_kickoffs:
        return dead, diagnostics, kickoffs
    return dead, diagnostics


def pre_kickoff_forced_dead(n, kickoffs, period_blocks):
    # A graded block opening before its first detected kickoff is dead until that kickoff.
    forced = np.zeros(n, dtype=bool)
    block_kickoffs = []
    for start, stop in period_blocks:
        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= n:
            raise ValueError("pre-kickoff blocks must be non-empty and inside the timeline")
        inside = [kickoff for kickoff in kickoffs if start <= kickoff < stop]
        first = min(inside) if inside else None
        if first is not None:
            forced[start:first] = True
        block_kickoffs.append([start, first])
    return forced, {
        "pre_kickoff_block_kickoffs": block_kickoffs,
        "pre_kickoff_forced_dead_frames": int(np.count_nonzero(forced)),
    }


# Only these restarts follow the ball leaving the pitch, so the stoppage begins at that exit.
EXIT_RESTART_TYPES = {"throw-in": "touchline", "corner kick": "goal_line", "goal kick": "goal_line"}


def restart_anchored_dead_intervals(trajectory, tag_features, pitch, config):
    # Walk back from each tagged exit restart while the ball is unseen or still at the boundary.
    length, width = pitch
    n = len(trajectory.source)
    observed = trajectory.source == 1
    xy = trajectory.xy
    outside = observed & ((xy[:, 0] < 0) | (xy[:, 0] > length) | (xy[:, 1] < 0) | (xy[:, 1] > width))
    touch = config["tracking_restart_touchline_distance_metres"]
    margin = config["tracking_restart_goal_area_margin_metres"]
    goal_depth = np.minimum(xy[:, 0], length - xy[:, 0])
    zones = {
        "touchline": outside | (observed & (np.minimum(xy[:, 1], width - xy[:, 1]) <= touch)),
        "goal_line": outside | (observed & (
            (goal_depth <= touch)
            | ((goal_depth <= 5.5 + margin) & (np.abs(xy[:, 1] - width / 2) <= 9.16 + margin))
        )),
    }
    maximum = config["tracking_restart_maximum_frames"]
    dead = np.zeros(n, dtype=bool)
    diagnostics = {"exit_restart_tags": 0, "restart_anchored_stoppages": 0,
                   "restart_anchor_abstentions": 0, "restart_anchored_frames": 0}
    types = tuple(tag_features.restart_types or ())
    for event, kind in zip(tag_features.restart_events, types):
        zone = EXIT_RESTART_TYPES.get(str(kind).lower())
        anchor = int(event[0])
        if zone is None or not 0 < anchor <= n:
            continue
        diagnostics["exit_restart_tags"] += 1
        lower = max(0, anchor - maximum)
        live = np.flatnonzero(observed[lower:anchor] & ~zones[zone][lower:anchor])
        if not live.size:
            diagnostics["restart_anchor_abstentions"] += 1
            continue
        last_live = lower + int(live[-1])
        seen = last_live + 1 + np.flatnonzero(zones[zone][last_live + 1:anchor])
        if not seen.size:
            diagnostics["restart_anchor_abstentions"] += 1
            continue
        # Start at the first sighting beyond the line, else the first placed-ball sighting.
        beyond = seen[outside[seen]]
        start = int(beyond[0]) if beyond.size else int(seen[0])
        if start - last_live - 1 <= config["tracking_restart_bridge_frames"]:
            start = last_live + 1
        dead[start:anchor] = True
        diagnostics["restart_anchored_stoppages"] += 1
    diagnostics["restart_anchored_frames"] = int(np.count_nonzero(dead))
    return dead, diagnostics


def validate_contact_carry_config(config):
    enabled = config.get("semi_markov_contact_carry_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("semi_markov_contact_carry_enabled must be boolean")
    if not enabled:
        return
    if not isinstance(config.get("contact_carry_include_video_ball", False), bool):
        raise ValueError("contact_carry_include_video_ball must be boolean")
    for key, minimum in (("contact_carry_window_frames", 1), ("contact_carry_maximum_frames", 1),
                         ("contact_carry_support_frames", 0)):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError("{} must be an integer of at least {}".format(key, minimum))
    for key in ("contact_carry_distance_metres", "contact_carry_turn_minimum",
                "contact_carry_minimum_ball_speed_mps", "contact_carry_weight"):
        value = config.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not np.isfinite(value) or value <= 0):
            raise ValueError("{} must be finite and positive".format(key))
    if config["contact_carry_turn_minimum"] > 2:
        raise ValueError("contact_carry_turn_minimum is one minus a cosine, at most 2")


def contact_carry_evidence(control_features, trajectory, fps, config):
    # A player within touching distance as the observed ball sharply changes direction is a touch;
    # the touching team keeps the ball until the next detected touch (bounded carry).
    validate_contact_carry_config(config)
    n = len(trajectory.source)
    window = config["contact_carry_window_frames"]
    carry = np.zeros((n, 2), dtype=np.float32)
    support = np.zeros((n, 2), dtype=bool)
    diagnostics = {"contact_carry_touches": 0, "contact_carry_frames": 0}
    if n <= 2 * window + 1:
        return carry, support, diagnostics
    observed = trajectory.source == 1
    if config.get("contact_carry_include_video_ball", False):
        # Opt-in: detector-supported video ball fills (source 4) may also reveal touches.
        observed = observed | (trajectory.source == 4)
    seen = np.r_[0, np.cumsum(observed)]
    rows = np.arange(window, n - window)
    fully_seen = seen[rows + window + 1] - seen[rows - window] == 2 * window + 1
    velocity = np.zeros((n, 2))
    velocity[1:] = np.diff(np.where(observed[:, None], trajectory.xy, 0.0), axis=0) * fps
    total = np.vstack((np.zeros((1, 2)), np.cumsum(velocity, axis=0)))
    before = (total[rows] - total[rows - window]) / window
    after = (total[rows + window + 1] - total[rows + 1]) / window
    speed_before = np.linalg.norm(before, axis=1)
    speed_after = np.linalg.norm(after, axis=1)
    moving = np.minimum(speed_before, speed_after) >= config["contact_carry_minimum_ball_speed_mps"]
    cosine = np.einsum("ij,ij->i", before, after) / np.maximum(speed_before * speed_after, 1e-9)
    turned = fully_seen & moving & (1 - cosine >= config["contact_carry_turn_minimum"])
    nearest = control_features.nearest_distances[:, :, 0]
    distances = np.where(np.isfinite(nearest), nearest, np.inf)
    closest = np.lib.stride_tricks.sliding_window_view(distances, 2 * window + 1, axis=0).min(axis=2)
    close = closest <= config["contact_carry_distance_metres"]
    touch = np.zeros((n, 2), dtype=bool)
    for team, other in ((HOME, AWAY), (AWAY, HOME)):
        touch[rows, team] = turned & close[:, team] & (
            ~close[:, other] | (closest[:, team] < closest[:, other])
        )
    events = sorted(
        (start, team) for team in (HOME, AWAY) for start, _ in _episodes(touch[:, team], 1)
    )
    diagnostics["contact_carry_touches"] = len(events)
    reach = config["contact_carry_support_frames"]
    unseen = np.flatnonzero(~observed)
    for index, (start, team) in enumerate(events):
        stop = start + config["contact_carry_maximum_frames"]
        if index + 1 < len(events):
            stop = min(stop, events[index + 1][0])
        # A touch is only detectable with `window` observed frames after it, so the carry ends
        # where a later touch could no longer be seen.
        next_unseen = np.searchsorted(unseen, start)
        if next_unseen < len(unseen):
            stop = min(stop, int(unseen[next_unseen]) - window)
        carry[start:min(stop, n), team] = config["contact_carry_weight"]
        support[max(0, start - reach):start + reach + 1, team] = True
    diagnostics["contact_carry_frames"] = int(np.count_nonzero(carry.any(axis=1)))
    return carry, support, diagnostics


def video_team_evidence(carry, support, video_team, config):
    # The team seen at the ball in the video counts like a detected touch on that frame.
    if carry is None or support is None:
        raise ValueError("video team evidence requires contact carry")
    frames = np.asarray(video_team[0], dtype=np.int64)
    teams = np.asarray(video_team[1], dtype=np.int64)
    if (frames.ndim != 1 or frames.shape != teams.shape or np.any(frames < 0)
            or np.any(frames >= len(carry)) or not np.isin(teams, (HOME, AWAY)).all()):
        raise ValueError("video team evidence needs in-range frames and home/away teams")
    carry, support = carry.copy(), support.copy()
    carry[frames, teams] = config["contact_carry_weight"]
    carry[frames, 1 - teams] = 0.0
    reach = config["contact_carry_support_frames"]
    for offset in range(-reach, reach + 1):
        rows = np.clip(frames + offset, 0, len(support) - 1)
        support[rows, teams] = True
    return carry, support


def video_team_pass_evidence(penalty, support, video_team, n, config):
    """Seen-at-the-ball team as team-choice evidence for the pass where live/dead is already fixed."""
    if penalty is None:
        penalty = np.zeros((n, 3), dtype=np.float32)
        support = np.zeros((n, 2), dtype=bool)
    carry, seen_support = video_team_evidence(
        np.zeros((n, 2), dtype=np.float32), np.zeros((n, 2), dtype=bool), video_team, config)
    seen = carry.any(axis=1)
    penalty = penalty.copy()
    penalty[seen, HOME] = carry[seen, AWAY]
    penalty[seen, AWAY] = carry[seen, HOME]
    return penalty, support | seen_support


def validate_gap_handover_config(config):
    enabled = config.get("semi_markov_gap_handover_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("semi_markov_gap_handover_enabled must be boolean")
    if not enabled:
        return
    if config.get("semi_markov_local_handover_enabled", False):
        raise ValueError("choose either global local-handover or gap-handover mode")
    for key in (
        "semi_markov_gap_handover_maximum_gap_frames",
        "semi_markov_gap_handover_reappearance_frames",
        "semi_markov_gap_handover_window_frames",
        "semi_markov_gap_handover_minimum_frames",
    ):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("{} must be a positive frame count".format(key))
    if config["semi_markov_gap_handover_minimum_frames"] > config["semi_markov_gap_handover_window_frames"]:
        raise ValueError("gap handover minimum frames exceed its window")
    for key in (
        "semi_markov_gap_handover_distance_metres",
        "semi_markov_gap_handover_distance_margin_metres",
        "semi_markov_gap_handover_minimum_confidence",
    ):
        value = config.get(key)
        if (
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not np.isfinite(value) or value <= 0
        ):
            raise ValueError("{} must be finite and positive".format(key))
    if config["semi_markov_gap_handover_minimum_confidence"] > 1:
        raise ValueError("gap handover confidence must not exceed one")
    if config.get("semi_markov_gap_handover_mode") not in ("augment", "contact_guarded"):
        raise ValueError("gap handover mode must be augment or contact_guarded")


def _anchor_teams(nearest, nearest_margin, observed, radius, margin):
    trusted = observed[:, None] & (nearest <= radius) & (nearest_margin >= margin)
    teams = np.full(len(nearest), UNKNOWN, dtype=np.int8)
    unique = trusted.sum(axis=1) == 1
    teams[unique] = np.argmax(trusted[unique], axis=1)
    return teams


def _corroborated_stoppage(trajectory, tags, pitch, anchors, config):
    rows = np.arange(len(trajectory.source))
    length, width = pitch
    margin = config["pitch_margin_metres"]
    outside = (trajectory.source == 1) & np.isfinite(trajectory.xy).all(axis=1) & (
        (trajectory.xy[:, 0] < -margin) | (trajectory.xy[:, 0] > length + margin)
        | (trajectory.xy[:, 1] < -margin) | (trajectory.xy[:, 1] > width + margin)
    )
    last_inside = np.maximum.accumulate(np.where(~outside, rows, -1))
    confirmed = outside & (
        rows - last_inside >= config["tracking_stoppage_exit_minimum_observations"]
    )
    last_exit = np.maximum.accumulate(np.where(confirmed, rows, -1))
    last_in_pitch = np.maximum.accumulate(np.where(
        (trajectory.source == 1) & np.isfinite(trajectory.xy).all(axis=1) & ~outside,
        rows, -1,
    ))
    exit_context = (last_exit >= 0) & (last_exit > last_in_pitch) & (
        rows - last_exit <= config["tracking_stoppage_corroboration_lookback_frames"]
    )
    restart_frames = (
        np.asarray(tags.restart_frames)
        if tags.restart_frames is not None
        else np.asarray([event[2] for event in tags.restart_events])
    )
    if np.any((restart_frames < 0) | (restart_frames > len(rows))):
        raise ValueError("restart corroboration frame is outside the trajectory")
    restart_present = np.zeros(len(rows), dtype=bool)
    restart_present[restart_frames[restart_frames < len(rows)].astype(np.int64)] = True
    last_restart = np.maximum.accumulate(np.where(restart_present, rows, -1))
    exit_context &= last_exit > last_restart
    corroboration = exit_context | tags.pre_restart_window
    observed = np.flatnonzero(trajectory.source == 1)
    for left, right in zip(observed[:-1], observed[1:]):
        gap = right - left - 1
        if (
            gap < 1 or gap > config["tracking_stoppage_continuity_max_gap_frames"]
            or anchors[left] == UNKNOWN or anchors[left] != anchors[right]
        ):
            continue
        if exit_context[left:right + 1].any() or np.any(
            (restart_frames >= left + 1) & (restart_frames <= right)
        ):
            continue
        corroboration[left + 1:right] = False
    return corroboration


def _gap_handover_evidence(trajectory, control, nearest, nearest_margin, config):
    observed = trajectory.source == 1
    anchors = _anchor_teams(
        nearest, nearest_margin, observed,
        config["semi_markov_gap_handover_distance_metres"],
        config["semi_markov_gap_handover_distance_margin_metres"],
    )
    context = np.zeros((len(observed), 2), dtype=bool)
    indexes = np.flatnonzero(observed)
    for left, right in zip(indexes[:-1], indexes[1:]):
        gap = right - left - 1
        if (
            gap < 1 or gap > config["semi_markov_gap_handover_maximum_gap_frames"]
            or anchors[left] == UNKNOWN or anchors[right] == UNKNOWN
            or anchors[left] == anchors[right]
        ):
            continue
        stop = min(len(observed), right + config["semi_markov_gap_handover_reappearance_frames"])
        context[left + 1:stop, anchors[right]] = True
    contact = (
        context & np.isin(trajectory.source, [1, 2])[:, None]
        & (trajectory.confidence[:, None] >= config["semi_markov_gap_handover_minimum_confidence"])
        & (nearest <= config["semi_markov_gap_handover_distance_metres"])
        & (nearest_margin >= config["semi_markov_gap_handover_distance_margin_metres"])
        & (control >= config["minimum_control_score"])
        & ((control - control[:, ::-1]) >= config["control_margin"])
    )
    return contact, context


def _tracking_stoppage_hint(control_features, trajectory, config):
    missing = trajectory.source == 0
    rows = np.arange(len(missing))
    last_seen = np.maximum.accumulate(np.where(~missing, rows, -1))
    elapsed = np.where(missing, rows - last_seen, 0)
    slow = (
        np.isfinite(control_features.median_player_speed)
        & np.isfinite(control_features.stationary_player_fraction)
        & (
            control_features.median_player_speed
            <= config["tracking_stoppage_maximum_median_speed_mps"]
        )
        & (
            control_features.stationary_player_fraction
            >= config["tracking_stoppage_minimum_stationary_fraction"]
        )
    )
    window = config["tracking_stoppage_motion_window_frames"]
    cumulative = np.r_[0, np.cumsum(slow, dtype=np.int32)]
    ends = rows + 1
    starts = np.maximum(0, ends - window)
    stable = (
        (ends >= window)
        & ((cumulative[ends] - cumulative[starts])
           >= config["tracking_stoppage_minimum_slow_fraction"] * window)
    )
    return missing & slow & stable & (
        elapsed >= config["tracking_stoppage_minimum_missing_frames"]
    )


def _restart_stoppage_hint(control_features, trajectory, tag_features, config):
    missing = trajectory.source == 0
    slow = (
        control_features.median_player_speed
        <= float(config["stoppage_median_speed_metres_per_second"])
    )
    stationary = (
        control_features.stationary_player_fraction
        >= float(config["stoppage_stationary_player_fraction"])
    )
    support = tag_features.pre_restart_window & (
        (missing & (slow | stationary)) | (slow & stationary)
    )
    expanded = tag_features.dead_hint.copy()
    window = tag_features.pre_restart_window
    boundaries = np.flatnonzero(window[1:] != window[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(window)]
    tolerance = int(config["stoppage_gap_tolerance_frames"])

    for start, end in zip(starts, ends):
        if not window[start]:
            continue
        direct = np.flatnonzero(tag_features.dead_hint[start:end])
        if direct.size == 0:
            continue
        direct_start = start + int(direct[0])
        extension_start = direct_start
        gap = 0
        for frame in range(direct_start - 1, start - 1, -1):
            if support[frame]:
                gap = 0
            else:
                gap += 1
                if gap > tolerance:
                    break
            extension_start = frame
        expanded[extension_start:direct_start] = True
    return expanded


def build_evidence(control_features, trajectory, tag_features, pitch, config, fps=None,
                   period_starts=None, broadcast_view=None, ballfree_home_probability=None,
                   video_team=None, period_blocks=None):
    validate_local_handover_config(config)
    validate_tracking_stoppage_config(config)
    validate_tracking_exit_bracket_config(config)
    validate_restart_dead_config(config)
    validate_gap_handover_config(config)
    validate_core_reliability_config(config)
    validate_touch_motion_config(config)
    validate_observed_handover_config(config)
    validate_owner_flight_config(config)
    validate_control_phase_config(config)
    validate_player_reliability_config(config)
    reliability_mode = config.get("core_reliability_mode", "none")
    reliable_observation = trajectory.source == 1
    nearest = control_features.nearest_distances[:, :, 0]
    closeness = np.zeros_like(nearest, dtype=np.float32)
    finite = np.isfinite(nearest)
    closeness[finite] = np.exp(
        -nearest[finite] / float(config["closeness_scale_metres"])
    )
    closeness *= trajectory.confidence[:, None]
    player_quality = None
    if config.get("player_reliability_enabled", False):
        age = control_features.nearest_player_age_frames
        if (not isinstance(age, np.ndarray) or age.shape != nearest.shape
                or not np.issubdtype(age.dtype, np.integer) or np.any(age < 0)
                or np.any(np.isfinite(nearest) & (age == 0))):
            raise ValueError("player reliability needs aligned nearest-player track ages")
        observed_near = (
            (trajectory.source == 1)[:, None]
            & (nearest <= config["distance_thresholds_metres"][1])
        )
        confidence = np.minimum(age / config["player_reliability_minimum_age_frames"], 1.)
        closeness *= np.where(observed_near, confidence, 1)
        player_quality = {
            "observed_near_ball_team_frames": int(np.count_nonzero(observed_near)),
            "young_near_ball_team_frames": int(np.count_nonzero(
                observed_near & (age < config["player_reliability_minimum_age_frames"])
            )),
            "age_threshold_frames": config["player_reliability_minimum_age_frames"],
        }

    counts = control_features.nearby_counts
    support = (
        float(config["nearby_two_weight"]) * counts[:, :, 1]
        + float(config["nearby_four_weight"]) * counts[:, :, 2]
    )
    support *= trajectory.confidence[:, None]
    if reliability_mode in ("geometry", "both"):
        uncertain_closeness = np.zeros_like(nearest, dtype=np.float32)
        uncertain_closeness[finite] = np.exp(
            -nearest[finite] / float(config["core_uncertain_closeness_scale_metres"])
        )
        uncertain_closeness *= trajectory.confidence[:, None]
        uncertain_support = (
            float(config["core_uncertain_nearby_two_weight"]) * counts[:, :, 1]
            + float(config["core_uncertain_nearby_four_weight"]) * counts[:, :, 2]
        )
        uncertain_support *= trajectory.confidence[:, None]
        closeness[~reliable_observation] = uncertain_closeness[~reliable_observation]
        support[~reliable_observation] = uncertain_support[~reliable_observation]

    evidence = closeness + support
    relative_velocity = (
        trajectory.velocity[:, None, :]
        - control_features.nearest_player_velocity
    )
    relative_speed = np.linalg.norm(relative_velocity, axis=2)
    motion_coherence = np.zeros_like(relative_speed, dtype=np.float32)
    finite_motion = np.isfinite(relative_speed)
    motion_coherence[finite_motion] = np.exp(
        -relative_speed[finite_motion]
        / float(config["motion_coherence_scale_metres_per_second"])
    )
    motion_coherence *= (
        trajectory.confidence[:, None]
        * (nearest <= float(config["motion_coherence_radius_metres"]))
    )
    evidence += float(config["motion_coherence_weight"]) * motion_coherence

    tag_weight = float(config["tag_weight"])
    evidence[:, HOME] += tag_weight * np.maximum(tag_features.team_prior, 0.0)
    evidence[:, AWAY] += tag_weight * np.maximum(-tag_features.team_prior, 0.0)

    restart_weight = float(config["restart_weight"])
    for team in (HOME, AWAY):
        evidence[:, team] += restart_weight * (tag_features.restart_team == team)

    observed = trajectory.source == 1
    opposing_nearest = nearest[:, ::-1]
    nearest_margin = np.full_like(nearest, -np.inf)
    np.subtract(
        opposing_nearest,
        nearest,
        out=nearest_margin,
        where=np.isfinite(opposing_nearest) & np.isfinite(nearest),
    )
    density_advantage = counts[:, :, 1] > counts[:, ::-1, 1]
    if config["strong_control_motion_gate_enabled"]:
        motion_gate = (
            relative_speed
            <= float(config["strong_control_relative_speed_metres_per_second"])
        )
    else:
        motion_gate = np.ones_like(relative_speed, dtype=bool)
    strong_control = (
        observed[:, None]
        & (nearest <= float(config["strong_control_distance_metres"]))
        & (
            density_advantage
            | (
                nearest_margin
                >= float(config["strong_control_distance_margin_metres"])
            )
        )
        & motion_gate
    )
    if player_quality is not None:
        strong_control &= (
            control_features.nearest_player_age_frames
            >= config["player_reliability_minimum_age_frames"]
        )
    uncertain_strong_control = None
    if reliability_mode in ("transitions", "both"):
        uncertain_strong_control = (
            observed[:, None]
            & (nearest <= config["core_uncertain_strong_control_distance_metres"])
            & (
                density_advantage
                | (nearest_margin >= config["core_uncertain_strong_control_distance_margin_metres"])
            )
            & motion_gate
        )

    dead = np.zeros(len(trajectory.xy), dtype=np.float32)
    stoppage_hint = _restart_stoppage_hint(
        control_features, trajectory, tag_features, config
    )
    dead += float(config["set_piece_dead_weight"]) * stoppage_hint
    total_players = np.sum(control_features.player_counts, axis=1)
    dead += float(config["low_player_dead_weight"]) * (
        total_players < int(config["low_player_count"])
    )

    missing = trajectory.source == 0
    dead += float(config["missing_ball_dead_weight"]) * (
        missing & (total_players < int(config["missing_ball_player_count"]))
    )
    length, width = pitch
    finite_ball = np.isfinite(trajectory.xy).all(axis=1)
    outside = finite_ball & (
        (trajectory.xy[:, 0] < -float(config["pitch_margin_metres"]))
        | (trajectory.xy[:, 0] > length + float(config["pitch_margin_metres"]))
        | (trajectory.xy[:, 1] < -float(config["pitch_margin_metres"]))
        | (trajectory.xy[:, 1] > width + float(config["pitch_margin_metres"]))
    )
    dead += float(config["outside_pitch_dead_weight"]) * outside
    exit_bracket_diagnostics = None
    if config.get("tracking_exit_bracket_enabled", False):
        if fps is None:
            raise ValueError("tracking exit brackets require explicit physical fps")
        exit_bracket, exit_bracket_diagnostics = tracking_exit_brackets(
            trajectory, pitch, fps, config,
        )
        dead = np.maximum(
            dead, float(config["outside_pitch_dead_weight"]) * exit_bracket,
        ).astype(np.float32)
    restart_dead_diagnostics = None
    forced_dead = None
    if any(config.get(key, False) for key in (
        "tracking_kickoff_dead_enabled", "tracking_restart_extension_enabled",
    )):
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("restart dead-ball cues require positive physical fps")
        restart_dead = np.zeros(len(dead), dtype=bool)
        restart_dead_diagnostics = {}
        if config.get("tracking_kickoff_dead_enabled", False):
            kickoff, quality, kickoffs = kickoff_dead_intervals(
                trajectory, control_features, tag_features, pitch, config,
                () if period_starts is None else period_starts, return_kickoffs=True,
            )
            restart_dead |= kickoff
            restart_dead_diagnostics.update(quality)
            if config.get("pre_kickoff_forced_dead_enabled", False) and period_blocks is not None:
                forced_dead, quality = pre_kickoff_forced_dead(len(dead), kickoffs, period_blocks)
                restart_dead_diagnostics.update(quality)
        if config.get("tracking_restart_extension_enabled", False):
            extension, quality = restart_anchored_dead_intervals(
                trajectory, tag_features, pitch, config,
            )
            restart_dead |= extension
            restart_dead_diagnostics.update(quality)
        dead = np.maximum(
            dead, float(config["outside_pitch_dead_weight"]) * restart_dead,
        ).astype(np.float32)
    broadcast_stoppage_diagnostics = None
    if config.get("broadcast_stoppage_enabled", False) and broadcast_view is not None:
        from possession.broadcast import broadcast_stoppage_intervals, validate_broadcast_config
        validate_broadcast_config(config)
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("broadcast stoppages require positive physical fps")
        cutaway, broadcast_stoppage_diagnostics = broadcast_stoppage_intervals(
            broadcast_view, trajectory, tag_features, fps, config,
        )
        dead = np.maximum(dead, config["broadcast_stoppage_dead_weight"] * cutaway).astype(np.float32)
    if config.get("tracking_stoppage_enabled", False):
        tracking_hint = _tracking_stoppage_hint(control_features, trajectory, config)
        if config.get("tracking_stoppage_require_corroboration", False):
            anchors = _anchor_teams(
                nearest, nearest_margin, observed,
                config["strong_control_distance_metres"],
                config["strong_control_distance_margin_metres"],
            )
            tracking_hint &= _corroborated_stoppage(
                trajectory, tag_features, pitch, anchors, config
            )
        dead = np.maximum(
            dead,
            config["tracking_stoppage_dead_weight"] * tracking_hint,
        ).astype(np.float32)
    handover_control = None
    if config.get("semi_markov_local_handover_enabled", False):
        handover_control = (
            np.isin(trajectory.source, [1, 2])[:, None]
            & (
                trajectory.confidence[:, None]
                >= config["semi_markov_local_handover_minimum_confidence"]
            )
            & (nearest <= config["semi_markov_local_handover_distance_metres"])
            & (
                nearest_margin
                >= config["semi_markov_local_handover_distance_margin_metres"]
            )
            & (evidence >= config["minimum_control_score"])
            & ((evidence - evidence[:, ::-1]) >= config["control_margin"])
        )
    gap_contact, gap_context = None, None
    if config.get("semi_markov_gap_handover_enabled", False):
        gap_contact, gap_context = _gap_handover_evidence(
            trajectory, evidence, nearest, nearest_margin, config
        )
    touch_motion = None
    touch_enabled = config.get("semi_markov_touch_motion_gate_enabled", False)
    if touch_enabled:
        if fps is None:
            raise ValueError("touch-motion evidence requires explicit runtime fps")
        touch_motion = contact_motion_features(
            trajectory.xy, trajectory.source, nearest, fps,
            config["strong_control_distance_metres"], config["touch_motion_horizon_frames"],
            config["touch_motion_window_frames"], config["max_ball_speed_metres_per_second"],
        )["contact_motion_disturbance"]
    observed_handover = None
    if config.get("semi_markov_observed_handover_enabled", False):
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("observed handover requires positive physical fps")
        disturbance = contact_motion_features(
            trajectory.xy, trajectory.source, nearest, fps,
            config["strong_control_distance_metres"], 1,
            config["observed_handover_motion_window_frames"],
            config["max_ball_speed_metres_per_second"],
        )["contact_motion_disturbance"]
        observed_handover = (
            strong_control
            & (nearest_margin >= config["strong_control_distance_margin_metres"])
            & np.isfinite(disturbance)
            & (disturbance >= config["observed_handover_motion_minimum"])
        )
    flight_prior, flight_diagnostics, flight_anchors = None, None, None
    if config.get("semi_markov_owner_flight_enabled", False):
        if fps is None:
            raise ValueError("owner-flight evidence requires explicit runtime fps")
        flight_prior, flight_diagnostics, flight_anchors = owner_flight_evidence(
            trajectory, control_features, tag_features, strong_control, fps, config,
        )
    phase = None
    if config.get("semi_markov_phase_enabled", False):
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("control phases require explicit positive physical fps")
        phase = infer_control_phase(
            trajectory, control_features, tag_features, strong_control, config, fps=fps,
        )
    contact_carry = contact_carry_support = contact_carry_diagnostics = None
    if config.get("semi_markov_contact_carry_enabled", False):
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("contact carry requires positive physical fps")
        contact_carry, contact_carry_support, contact_carry_diagnostics = contact_carry_evidence(
            control_features, trajectory, fps, config,
        )
    video_team_mode = config.get("video_team_mode", "carry")
    if video_team_mode not in ("carry", "team_pass"):
        raise ValueError("video_team_mode must be carry or team_pass")
    if video_team is not None and video_team_mode == "carry":
        contact_carry, contact_carry_support = video_team_evidence(
            contact_carry, contact_carry_support, video_team, config)
        contact_carry_diagnostics["video_team_frames"] = int(len(video_team[0]))
    ballfree_penalty = ballfree_support = None
    if config.get("semi_markov_ballfree_enabled", False) and ballfree_home_probability is not None:
        probability = np.asarray(ballfree_home_probability, dtype=np.float32)
        if probability.shape != (len(trajectory.source),) or not np.isfinite(probability).all():
            raise ValueError("ball-free probabilities must align with the trajectory")
        # Teams reorganise a second or two after a turnover, so read the players slightly ahead.
        ahead = int(config.get("ballfree_lookahead_frames", 0))
        if ahead > 0:
            probability = np.r_[probability[ahead:], np.repeat(probability[-1:], ahead)]
        # Only inside long unobserved stretches: penalize the team the players disfavour.
        unseen = np.zeros(len(probability), dtype=bool)
        # Fused broadcast balls fill parts of long gaps; the players still speak there.
        unseen_codes = (0, 4) if config.get("broadcast_ball_fusion_enabled", False) else (0,)
        for start, stop in _episodes(np.isin(trajectory.source, unseen_codes),
                                     config.get("ballfree_minimum_unseen_frames", 1)):
            unseen[start:stop] = True
        lean = 2 * probability - 1
        ballfree_penalty = np.zeros((len(probability), 3), dtype=np.float32)
        ballfree_penalty[unseen, HOME] = config["ballfree_weight"] * np.maximum(0, -lean[unseen])
        ballfree_penalty[unseen, AWAY] = config["ballfree_weight"] * np.maximum(0, lean[unseen])
        # A confident lean also supports switching team, so a switch never costs more than a dead detour.
        confident = config.get("ballfree_support_lean", np.inf)
        ballfree_support = np.zeros((len(probability), 2), dtype=bool)
        ballfree_support[:, HOME] = unseen & (lean >= confident)
        ballfree_support[:, AWAY] = unseen & (lean <= -confident)
        # Optional weaker reading where the ball is tracked: noisy ball positions make proximity unreliable.
        seen_weight = float(config.get("ballfree_seen_weight", 0.0))
        if not np.isfinite(seen_weight) or seen_weight < 0:
            raise ValueError("ballfree_seen_weight must be finite and nonnegative")
        if seen_weight > 0:
            seen = ~unseen
            # Only where proximity is weak: no tracked player within the gate distance of the ball.
            gate = config.get("ballfree_seen_min_distance_metres")
            if gate is not None:
                nearest = np.min(control_features.nearest_distances[:, :, 0], axis=1)
                seen &= ~(nearest <= float(gate))
            # Optionally skip balls in flight (passes), where possession stays with the passer.
            max_speed = config.get("ballfree_seen_max_ball_speed_mps")
            if max_speed is not None:
                seen &= trajectory.speed <= float(max_speed)
            ballfree_penalty[seen, HOME] = seen_weight * np.maximum(0, -lean[seen])
            ballfree_penalty[seen, AWAY] = seen_weight * np.maximum(0, lean[seen])
    if video_team is not None and video_team_mode == "team_pass":
        ballfree_penalty, ballfree_support = video_team_pass_evidence(
            ballfree_penalty, ballfree_support, video_team, len(trajectory.source), config)
    return Evidence(
        control=evidence, dead=dead, strong_control=strong_control,
        handover_control=handover_control,
        observed_handover_support=observed_handover,
        gap_handover_control=gap_contact, gap_handover_context=gap_context,
        reliable_observation=reliable_observation if reliability_mode != "none" or touch_enabled else None,
        uncertain_strong_control=uncertain_strong_control,
        touch_motion_disturbance=touch_motion,
        owner_flight_prior=flight_prior,
        owner_flight_diagnostics=flight_diagnostics,
        owner_flight_anchor_rows=flight_anchors,
        player_reliability_diagnostics=player_quality,
        control_phase=phase,
        exit_bracket_diagnostics=exit_bracket_diagnostics,
        restart_dead_diagnostics=restart_dead_diagnostics,
        forced_dead=forced_dead,
        contact_carry=contact_carry,
        contact_carry_support=contact_carry_support,
        contact_carry_diagnostics=contact_carry_diagnostics,
        broadcast_stoppage_diagnostics=broadcast_stoppage_diagnostics,
        ballfree_penalty=ballfree_penalty,
        ballfree_support=ballfree_support,
    )


def _control_candidate(frame_scores, restart_team, config):
    best = int(np.argmax(frame_scores))
    other = AWAY if best == HOME else HOME
    margin = frame_scores[best] - frame_scores[other]
    if (
        frame_scores[best] >= float(config["minimum_control_score"])
        and margin >= float(config["control_margin"])
    ):
        return best
    if restart_team in (HOME, AWAY):
        return int(restart_team)
    return UNKNOWN


def _decode_state_machine(evidence, tag_features, config):
    n_frames = len(evidence.dead)
    labels = np.full(n_frames, DEAD, dtype=np.int8)
    state = DEAD
    candidate = UNKNOWN
    candidate_start = 0
    candidate_count = 0
    candidate_strong_count = 0

    for frame in range(n_frames):
        dead_candidate = evidence.dead[frame] >= float(config["dead_threshold"])
        control_candidate = _control_candidate(
            evidence.control[frame], tag_features.restart_team[frame], config
        )

        if state == DEAD:
            desired = DEAD if dead_candidate else control_candidate
            required = int(config["live_confirm_frames"])
            strong_required = required
        elif dead_candidate:
            desired = DEAD
            required = int(config["dead_confirm_frames"])
            strong_required = required
        elif control_candidate in (HOME, AWAY) and control_candidate != state:
            desired = control_candidate
            required = int(config["transition_confirm_frames"])
            strong_required = int(config["strong_transition_confirm_frames"])
        else:
            desired = state
            required = 1
            strong_required = required

        strong_frame = (
            evidence.strong_control is not None
            and desired in (HOME, AWAY)
            and bool(evidence.strong_control[frame, desired])
        )

        if desired == state or desired == UNKNOWN:
            candidate = UNKNOWN
            candidate_count = 0
            candidate_strong_count = 0
            labels[frame] = state
            continue

        if desired != candidate:
            candidate = desired
            candidate_start = frame
            candidate_count = 1
            candidate_strong_count = 1 if strong_frame else 0
        else:
            candidate_count += 1
            candidate_strong_count = (
                candidate_strong_count + 1 if strong_frame else 0
            )

        labels[frame] = state
        if (
            candidate_count >= required
            or candidate_strong_count >= strong_required
        ):
            state = candidate
            labels[candidate_start:frame + 1] = state
            candidate = UNKNOWN
            candidate_count = 0
            candidate_strong_count = 0

    return labels


def _semi_markov_emissions(evidence, config):
    control = evidence.control.astype(np.float32, copy=False)
    dead = evidence.dead.astype(np.float32, copy=False)
    minimum_control = float(config["minimum_control_score"])
    margin_weight = float(
        config["semi_markov_control_margin_weight"]
    )
    dead_suppression = float(
        config["semi_markov_live_dead_suppression_weight"]
    )
    dead_control_weight = float(
        config["semi_markov_dead_control_weight"]
    )
    emissions = np.empty((len(dead), 3), dtype=np.float32)
    emissions[:, HOME] = (
        control[:, HOME]
        - minimum_control
        + margin_weight * (control[:, HOME] - control[:, AWAY])
        - dead_suppression * dead
    )
    emissions[:, AWAY] = (
        control[:, AWAY]
        - minimum_control
        + margin_weight * (control[:, AWAY] - control[:, HOME])
        - dead_suppression * dead
    )
    emissions[:, DEAD] = (
        dead
        - float(config["semi_markov_dead_emission_threshold"])
        - dead_control_weight * np.max(control, axis=1)
    )
    if config.get("semi_markov_owner_flight_enabled", False):
        validate_owner_flight_config(config)
        prior = evidence.owner_flight_prior
        if (prior is None or prior.shape != (len(dead), 2) or not np.isfinite(prior).all()
                or np.any(prior < 0)):
            raise ValueError("owner-flight decoding requires finite, aligned team priors")
        emissions[:, HOME:AWAY + 1] += prior
    if config.get("semi_markov_phase_enabled", False):
        validate_control_phase_config(config)
        phase = _validated_control_phase(evidence)
        flying = phase.kind == FLIGHT
        rows = np.flatnonzero(flying)
        emissions[rows, phase.team[rows]] += config["phase_flight_weight"]
    if config.get("semi_markov_contact_carry_enabled", False):
        carry = evidence.contact_carry
        if not isinstance(carry, np.ndarray) or carry.shape != (len(dead), 2):
            raise ValueError("contact carry decoding requires aligned team bonuses")
        emissions[:, HOME:AWAY + 1] += carry
    if evidence.ballfree_penalty is not None:
        emissions -= evidence.ballfree_penalty
    return emissions


def _base_strong_transition_support(evidence, config):
    n_frames = len(evidence.dead)
    support = np.zeros((n_frames, 2), dtype=bool)
    if config.get("semi_markov_local_handover_enabled", False):
        validate_local_handover_config(config)
        contact = evidence.handover_control
        if contact is None or contact.shape != (n_frames, 2):
            raise ValueError("local handover requires aligned contact evidence")
        window = config["semi_markov_local_handover_window_frames"]
        minimum = config["semi_markov_local_handover_minimum_frames"]
        cumulative = np.vstack((
            np.zeros((1, 2), dtype=np.int32),
            np.cumsum(contact, axis=0, dtype=np.int32),
        ))
        ends = np.arange(n_frames) + 1
        starts = np.maximum(0, ends - window)
        return contact & ((cumulative[ends] - cumulative[starts]) >= minimum)
    if evidence.strong_control is None:
        return support

    window = int(config["semi_markov_strong_transition_window_frames"])
    minimum = int(config["semi_markov_strong_transition_minimum_frames"])
    if window < 1 or minimum < 1 or minimum > window:
        raise ValueError(
            "semi-Markov strong transition support must satisfy "
            "1 <= minimum frames <= window frames"
        )

    strong = evidence.strong_control.astype(np.int32, copy=False)
    cumulative = np.vstack((
        np.zeros((1, 2), dtype=np.int32),
        np.cumsum(strong, axis=0, dtype=np.int32),
    ))
    ends = np.minimum(np.arange(n_frames) + window, n_frames)
    counts = cumulative[ends] - cumulative[np.arange(n_frames)]
    support[:] = counts >= minimum
    return support


def _semi_markov_strong_transition_support(evidence, config):
    validate_touch_motion_config(config)
    validate_observed_handover_config(config)
    support = _base_strong_transition_support(evidence, config)
    if config.get("core_reliability_mode", "none") in ("transitions", "both"):
        validate_core_reliability_config(config)
        reliable = evidence.reliable_observation
        strong = evidence.uncertain_strong_control
        if (
            reliable is None or strong is None
            or reliable.shape != (len(support),) or strong.shape != support.shape
        ):
            raise ValueError("core reliability requires aligned observation and fallback-support arrays")
        cumulative = np.vstack((
            np.zeros((1, 2), dtype=np.int32),
            np.cumsum(strong, axis=0, dtype=np.int32),
        ))
        ends = np.minimum(
            np.arange(len(support)) + config["core_uncertain_strong_transition_window_frames"],
            len(support),
        )
        fallback = (
            cumulative[ends] - cumulative[np.arange(len(support))]
            >= config["core_uncertain_strong_transition_minimum_frames"]
        )
        support = np.where(reliable[:, None], support, fallback)
    if (config.get("semi_markov_touch_motion_gate_enabled", False)
            and config.get("touch_motion_mode", "gate") == "gate"):
        validate_touch_motion_config(config)
        observed, motion = _touch_motion_evidence(evidence)
        support &= (
            ~observed[:, None] | ~np.isfinite(motion)
            | (motion >= config["touch_motion_minimum"])
        )
    if config.get("semi_markov_observed_handover_enabled", False):
        handover = evidence.observed_handover_support
        if (not isinstance(handover, np.ndarray) or handover.shape != support.shape
                or handover.dtype != bool):
            raise ValueError("observed handover needs aligned boolean contact evidence")
        support |= handover
    if config.get("semi_markov_contact_carry_enabled", False):
        touched = evidence.contact_carry_support
        if not isinstance(touched, np.ndarray) or touched.shape != support.shape:
            raise ValueError("contact carry needs aligned boolean touch support")
        support |= touched
    if evidence.ballfree_support is not None:
        support |= evidence.ballfree_support
    if not config.get("semi_markov_gap_handover_enabled", False):
        return support
    validate_gap_handover_config(config)
    contact, context = evidence.gap_handover_control, evidence.gap_handover_context
    if (
        contact is None or context is None
        or contact.shape != support.shape or context.shape != support.shape
    ):
        raise ValueError("gap handover requires aligned contact and context evidence")
    cumulative = np.vstack((
        np.zeros((1, 2), dtype=np.int32),
        np.cumsum(contact, axis=0, dtype=np.int32),
    ))
    ends = np.arange(len(support)) + 1
    starts = np.maximum(0, ends - config["semi_markov_gap_handover_window_frames"])
    gap_support = contact & (
        cumulative[ends] - cumulative[starts]
        >= config["semi_markov_gap_handover_minimum_frames"]
    )
    if config["semi_markov_gap_handover_mode"] == "augment":
        return support | gap_support
    return np.where(context, gap_support, support)


def _float32_difference(cost):
    # Whether numpy evaluates float32 score - cost in float32 (else float64), as in the reference loop.
    return (np.float32(0) - cost).dtype == np.float32


def _semi_markov_forward(emissions, previous, backpointers, offsets, mature, durations, fixed_labels,
                         frame_costs):
    """Forward pass of decode_semi_markov for frames 1.., with identical float32/float64 rounding.

    frame_costs[frame][state] holds ((cost, float32 difference) per source state). Fills
    backpointers in place and returns the final scores, or None when inputs are not finite (the
    reference loop then runs instead).
    """
    from array import array

    n_frames, n_expanded = len(emissions), len(previous)
    if (emissions.dtype != np.float32 or previous.dtype != np.float32
            or not np.isfinite(emissions).all() or frame_costs is None):
        return None
    states = (HOME, AWAY, DEAD)
    sources = [[int(mature[other]) for other in states if other != state] for state in states]
    firsts = [int(offsets[state]) for state in states]
    lasts = [int(mature[state]) for state in states]
    key_slots = np.asarray(firsts + lasts)
    chain = np.ones(n_expanded, dtype=bool)
    chain[key_slots] = False
    # Every expanded state except a block's first takes its predecessor's score; firsts and lasts
    # are overwritten below.
    predecessor = np.maximum(np.arange(n_expanded) - 1, 0)
    expanded_emissions = emissions[:, np.repeat(np.arange(3), durations)]
    skipped = np.zeros((n_frames, 3), dtype=bool)
    if fixed_labels is not None:
        for state in states:
            skipped[:, state] = ((fixed_labels != state) & (fixed_labels != FREE)
                                 & ((fixed_labels != UNKNOWN) | (state == DEAD)))
    skipped_frames = skipped.any(axis=1).tolist()
    rounded = array("f", [0.0])
    rows = emissions.tolist()
    best_sources = np.empty((n_frames, 3), dtype=np.int16)
    last_sources = np.empty((n_frames, 3), dtype=np.int16)
    scores = previous.tolist()
    current = previous
    for frame in range(1, n_frames):
        current = previous[predecessor]
        current += expanded_emissions[frame]
        emission = rows[frame]
        costs = frame_costs[frame]
        values = []
        for state in states:
            (cost0, single0), (cost1, single1) = costs[state]
            source0, source1 = sources[state]
            score0 = scores[source0] - cost0
            if single0:
                rounded[0] = score0
                score0 = rounded[0]
            score1 = scores[source1] - cost1
            if single1:
                rounded[0] = score1
                score1 = rounded[0]
            if score0 >= score1:
                rounded[0] = score0 + emission[state]
                best_sources[frame, state] = source0
            else:
                rounded[0] = score1 + emission[state]
                best_sources[frame, state] = source1
            values.append(rounded[0])
        for state in states:
            last = lasts[state]
            if scores[last] >= scores[last - 1]:
                rounded[0] = scores[last] + emission[state]
                last_sources[frame, state] = last
            else:
                rounded[0] = scores[last - 1] + emission[state]
                last_sources[frame, state] = last - 1
            values.append(rounded[0])
        current[key_slots] = values
        if skipped_frames[frame]:
            for state in states:
                if skipped[frame, state]:
                    current[firsts[state]:lasts[state] + 1] = -np.inf
        scores = current.tolist()
        previous = current
    for state in states:
        live = ~skipped[1:, state]
        block = slice(firsts[state], lasts[state] + 1)
        pointers = np.where(chain[block], np.arange(firsts[state], lasts[state] + 1) - 1, -1).astype(np.int16)
        pointers = np.broadcast_to(pointers, (n_frames - 1, len(pointers))).copy()
        pointers[:, 0] = best_sources[1:, state]
        pointers[:, -1] = last_sources[1:, state]
        pointers[~live] = -1
        backpointers[1:, block] = pointers
    return current


def _semi_markov_frame_costs(n_frames, transition_cost, uncertain_transition_cost, reliability_transitions,
                             reliable_observation, strong_live_cost, config, touch_costs, phase,
                             strong_transition_support):
    """Per frame and state, the (cost, float32 difference) of each source state, exactly as the
    reference loop selects them; None when a cost is not a finite number."""
    unreliable = (np.zeros(n_frames, dtype=bool) if not reliability_transitions
                  else ~np.asarray([bool(value) for value in reliable_observation[:n_frames]]))
    # Config values are read only when the reference loop (frames 1..) would read them.
    uncertain_strong = float(config["core_uncertain_strong_transition_cost"]) if unreliable[1:].any() else None
    turnover = np.zeros((n_frames, 2), dtype=bool)
    turnover_cost = None
    if phase is not None:
        for state in (HOME, AWAY):
            turnover[:, state] = (phase.kind == TURNOVER) & (phase.team == state)
        if turnover[1:].any():
            turnover_cost = config["phase_turnover_transition_cost"]
            if (isinstance(turnover_cost, bool) or not isinstance(turnover_cost, (int, float))
                    or not np.isfinite(turnover_cost)):
                return None
    support = np.asarray(strong_transition_support[:n_frames, :2], dtype=bool) | turnover
    matrices = [transition_cost] + ([uncertain_transition_cost] if uncertain_transition_cost is not None else [])
    if not all(np.isfinite(matrix).all() for matrix in matrices) or not np.isfinite(strong_live_cost) or (
            uncertain_strong is not None and not np.isfinite(uncertain_strong)):
        return None
    strong_single = _float32_difference(strong_live_cost)
    uncertain_single = uncertain_strong is not None and _float32_difference(uncertain_strong)
    turnover_single = turnover_cost is not None and _float32_difference(turnover_cost)
    touch_single = _float32_difference(np.float64(0))
    touch_finite = np.zeros((n_frames, 2), dtype=bool) if touch_costs is None else np.isfinite(touch_costs[:n_frames, :2])
    states = (HOME, AWAY, DEAD)

    entries = [[[(float(matrix[source, state]), _float32_difference(matrix[source, state])) for state in states]
                for source in states] for matrix in matrices]
    tables = {}

    def costs_for(unsure, overrides):
        # overrides[state]: the (cost, float32 difference) used for live-to-live transitions into state.
        key = (unsure, tuple(None if item is None else (float(item[0]).hex(), item[1]) for item in overrides))
        if key not in tables:
            tables[key] = [[overrides[state] if state != DEAD and source != DEAD and overrides[state] is not None
                            else entries[unsure][source][state]
                            for source in states if source != state] for state in states]
        return tables[key]

    plain = [costs_for(index, (None, None, None)) for index in range(len(matrices))]
    frame_costs = [plain[1] if flag else plain[0] for flag in unreliable.tolist()]
    support_list, turnover_list, touch_list = support.tolist(), turnover.tolist(), touch_finite.tolist()
    # Frame 0 has no transitions.
    for frame in (np.flatnonzero(support[1:].any(axis=1)) + 1).tolist():
        unsure = 1 if unreliable[frame] else 0
        supported = (uncertain_strong, uncertain_single) if unsure else (strong_live_cost, strong_single)
        overrides = [None, None, None]
        for state in (HOME, AWAY):
            if not support_list[frame][state]:
                continue
            if turnover_list[frame][state]:
                overrides[state] = (float(turnover_cost), turnover_single)
            elif touch_list[frame][state]:
                overrides[state] = (float(touch_costs[frame, state]), touch_single)
            else:
                overrides[state] = supported
        frame_costs[frame] = costs_for(unsure, tuple(overrides))
    return frame_costs


def decode_semi_markov(evidence, config, fixed_labels=None):
    validate_core_reliability_config(config)
    validate_touch_motion_config(config)
    validate_control_phase_config(config)
    emissions = _semi_markov_emissions(evidence, config)
    n_frames = len(emissions)
    if fixed_labels is not None and (
        not isinstance(fixed_labels, np.ndarray) or fixed_labels.shape != (n_frames,)
        or not np.issubdtype(fixed_labels.dtype, np.integer)
        or not np.isin(fixed_labels, (HOME, AWAY, DEAD, UNKNOWN, FREE)).all()
    ):
        raise ValueError("bounded decoding requires aligned H/A/D, free-live or free labels")
    if n_frames == 0:
        return np.empty(0, dtype=np.int8)

    minimum_live = int(config["semi_markov_minimum_live_frames"])
    minimum_dead = int(config["semi_markov_minimum_dead_frames"])
    if minimum_live < 2 or minimum_dead < 2:
        raise ValueError("semi-Markov minimum durations must be at least 2")
    durations = (minimum_live, minimum_live, minimum_dead)
    offsets = np.cumsum((0,) + durations[:-1]).astype(np.int16)
    mature = np.asarray(
        [offsets[state] + durations[state] - 1 for state in range(3)],
        dtype=np.int16,
    )
    expanded_states = int(sum(durations))
    state_of_expanded = np.empty(expanded_states, dtype=np.int8)
    for state in (HOME, AWAY, DEAD):
        state_of_expanded[
            offsets[state]:offsets[state] + durations[state]
        ] = state

    live_cost = float(config["semi_markov_live_transition_cost"])
    strong_live_cost = float(
        config["semi_markov_strong_transition_cost"]
    )
    if strong_live_cost < 0.0 or strong_live_cost > live_cost:
        raise ValueError(
            "semi-Markov strong transition cost must be between zero "
            "and the live transition cost"
        )
    dead_cost = float(config["semi_markov_dead_transition_cost"])
    strong_transition_support = _semi_markov_strong_transition_support(
        evidence, config
    )
    phase = _validated_control_phase(evidence) if config.get("semi_markov_phase_enabled", False) else None
    touch_costs = None
    if (config.get("semi_markov_touch_motion_gate_enabled", False)
            and config.get("touch_motion_mode", "gate") in ("soften", "confidence")):
        validate_touch_motion_config(config)
        observed, motion = _touch_motion_evidence(evidence)
        available = observed[:, None] & np.isfinite(motion)
        touch_costs = np.full(motion.shape, np.nan)
        weak_cost = config["touch_motion_weak_transition_cost"]
        if config["touch_motion_mode"] == "confidence":
            confidence = np.full(motion.shape, np.nan)
            np.divide(motion, motion + config["touch_motion_minimum"],
                      out=confidence, where=available)
            touch_costs[available] = (
                weak_cost - (weak_cost - strong_live_cost) * confidence[available]
            )
        else:
            touch_costs[available & (motion < config["touch_motion_minimum"])] = weak_cost
    transition_cost = np.asarray([
        [0.0, live_cost, dead_cost],
        [live_cost, 0.0, dead_cost],
        [dead_cost, dead_cost, 0.0],
    ], dtype=np.float32)
    reliability_transitions = config.get("core_reliability_mode", "none") in ("transitions", "both")
    uncertain_transition_cost = None
    if reliability_transitions:
        uncertain_transition_cost = transition_cost.copy()
        uncertain_transition_cost[HOME, AWAY] = config["core_uncertain_live_transition_cost"]
        uncertain_transition_cost[AWAY, HOME] = config["core_uncertain_live_transition_cost"]
    previous = np.full(expanded_states, -np.inf, dtype=np.float32)
    backpointers = np.full(
        (n_frames, expanded_states), -1, dtype=np.int16
    )
    for state in (HOME, AWAY, DEAD):
        if (fixed_labels is not None and fixed_labels[0] not in (state, FREE)
                and (fixed_labels[0] != UNKNOWN or state == DEAD)):
            continue
        start_score = emissions[0, state]
        if state == DEAD:
            start_score += float(config["semi_markov_start_dead_bonus"])
        previous[offsets[state]] = start_score
    duration_predecessors = np.arange(expanded_states, dtype=np.int16) - 1

    # Faster equivalent forward pass; the loop below is the reference and runs only when it declines.
    forward = _semi_markov_forward(
        emissions, previous, backpointers, offsets, mature, durations, fixed_labels,
        _semi_markov_frame_costs(
            n_frames, transition_cost, uncertain_transition_cost, reliability_transitions,
            evidence.reliable_observation if reliability_transitions else None, strong_live_cost, config,
            touch_costs, phase, strong_transition_support,
        ),
    )
    if forward is not None:
        previous = forward
    for frame in range(1 if forward is None else n_frames, n_frames):
        if reliability_transitions and not evidence.reliable_observation[frame]:
            frame_transition_cost = uncertain_transition_cost
            frame_strong_cost = float(config["core_uncertain_strong_transition_cost"])
        else:
            frame_transition_cost = transition_cost
            frame_strong_cost = strong_live_cost
        current = np.full(expanded_states, -np.inf, dtype=np.float32)
        for state in (HOME, AWAY, DEAD):
            if (fixed_labels is not None and fixed_labels[frame] not in (state, FREE)
                    and (fixed_labels[frame] != UNKNOWN or state == DEAD)):
                continue
            first = int(offsets[state])
            last = int(mature[state])
            sources = [int(mature[other]) for other in (HOME, AWAY, DEAD)
                       if other != state]
            source_states = [other for other in (HOME, AWAY, DEAD)
                             if other != state]
            supported_cost = frame_strong_cost
            if (touch_costs is not None and state in (HOME, AWAY)
                    and np.isfinite(touch_costs[frame, state])):
                supported_cost = touch_costs[frame, state]
            scores = [
                previous[source]
                - (
                    (config["phase_turnover_transition_cost"]
                     if (phase is not None and phase.kind[frame] == TURNOVER
                         and phase.team[frame] == state)
                     else supported_cost)
                    if (
                        state in (HOME, AWAY)
                        and source_state in (HOME, AWAY)
                        and (strong_transition_support[frame, state]
                             or (phase is not None and phase.kind[frame] == TURNOVER
                                 and phase.team[frame] == state))
                    )
                    else frame_transition_cost[source_state, state]
                )
                for source, source_state in zip(sources, source_states)
            ]
            best = int(np.argmax(scores))
            current[first] = scores[best] + emissions[frame, state]
            backpointers[frame, first] = sources[best]

            current[first + 1:last] = previous[first:last - 1] + emissions[frame, state]
            backpointers[frame, first + 1:last] = duration_predecessors[first + 1:last]

            if previous[last] >= previous[last - 1]:
                source = last
            else:
                source = last - 1
            current[last] = previous[source] + emissions[frame, state]
            backpointers[frame, last] = source
        previous = current

    if fixed_labels is not None and not np.isfinite(previous[mature]).any():
        raise ValueError("bounded decoding has no duration-valid path")
    expanded = int(mature[int(np.argmax(previous[mature]))])
    labels = np.empty(n_frames, dtype=np.int8)
    for frame in range(n_frames - 1, -1, -1):
        labels[frame] = state_of_expanded[expanded]
        if frame:
            expanded = int(backpointers[frame, expanded])
    return labels


def _decode_bounded_control_phases(evidence, config):
    phase = _validated_control_phase(evidence)
    motion_mode = config.get("phase_motion_continuity_enabled", False)
    if motion_mode and (
        not isinstance(evidence.strong_control, np.ndarray)
        or evidence.strong_control.shape != (len(phase.kind), 2)
        or evidence.strong_control.dtype != bool
    ):
        raise ValueError("motion continuity needs aligned observed strong-control evidence")
    veto = config.get("phase_early_opponent_veto_enabled", False)
    preserve_switch = config.get("phase_preserve_supported_switch_enabled", False)
    if veto and (
        not isinstance(phase.incipient_contact, np.ndarray)
        or phase.incipient_contact.shape != (len(phase.kind), 2)
        or phase.incipient_contact.dtype != bool
    ):
        raise ValueError("early opponent veto needs aligned same-player contact evidence")
    baseline = decode_semi_markov(
        replace(evidence, control_phase=None),
        {**config, "semi_markov_phase_enabled": False,
         "phase_early_opponent_veto_enabled": False,
         "phase_preserve_supported_switch_enabled": False,
         "phase_motion_continuity_enabled": False},
    )
    editable = np.zeros(len(baseline), dtype=bool)
    vetoed = np.zeros(len(baseline), dtype=bool) if veto else None
    supported_switches = (
        _base_strong_transition_support(evidence, config) if motion_mode else None
    )
    anchor, owner = -1, UNKNOWN
    bracketed = accepted = 0
    for row, kind in enumerate(phase.kind):
        if kind == FLIGHT:
            anchor = int(phase.anchor_rows[row])
            owner = int(phase.team[row])
        elif kind in (RECEPTION, TURNOVER):
            horizon = (
                config["phase_observed_reception_bridge_max_frames"]
                if config.get("phase_observed_reception_bridge_enabled", False)
                and kind == RECEPTION
                else config["phase_flight_max_frames"]
            )
            if (
                anchor >= 0 and row - anchor <= (
                    horizon
                    + config["phase_contact_minimum_frames"]
                )
                and phase.team[anchor] == owner
            ):
                bracketed += 1
                if baseline[anchor] == owner and baseline[row] == phase.team[row]:
                    edit_end = row
                    if motion_mode:
                        flight = phase.kind[anchor + 1:row] == FLIGHT
                        first_nonflight = np.flatnonzero(~flight)
                        edit_end = anchor + 1 + int(
                            first_nonflight[0] if first_nonflight.size else len(flight)
                        )
                        receiver = phase.team[row]
                        opponent = AWAY if receiver == HOME else HOME
                        if (
                            edit_end == anchor + 1
                            or row - edit_end >= config["phase_contact_minimum_frames"]
                            or not np.all(phase.team[anchor + 1:edit_end] == owner)
                            or not np.all(phase.anchor_rows[anchor + 1:edit_end] == anchor)
                            or not np.all(phase.kind[edit_end:row] == UNSEEN)
                            or np.any(baseline[anchor + 1:row] == DEAD)
                            or not np.all(evidence.strong_control[edit_end:row, receiver])
                            or np.any(evidence.strong_control[edit_end:row, opponent])
                            or (kind == TURNOVER and not np.all(
                                baseline[row:row + config["semi_markov_minimum_live_frames"]]
                                == receiver
                            ))
                        ):
                            anchor, owner = -1, UNKNOWN
                            continue
                        switches = np.flatnonzero(
                            baseline[anchor + 1:edit_end] != baseline[anchor:edit_end - 1]
                        ) + anchor + 1
                        if any(
                            baseline[frame] in (HOME, AWAY)
                            and baseline[frame] != owner
                            and supported_switches[frame, baseline[frame]]
                            for frame in switches
                        ):
                            anchor, owner = -1, UNKNOWN
                            continue
                    if veto:
                        opponent = AWAY if owner == HOME else HOME
                        opposing_contact = phase.incipient_contact[
                            anchor + 1:row, opponent
                        ]
                        editable[anchor + 1:row] = ~opposing_contact
                        vetoed[anchor + 1:row] |= opposing_contact
                    else:
                        editable[anchor + 1:edit_end] = True
                    accepted += 1
            anchor, owner = -1, UNKNOWN
        elif kind in (CONTACT, STOPPAGE):
            anchor, owner = -1, UNKNOWN
    editable &= baseline != DEAD
    if veto:
        vetoed &= baseline != DEAD
    protected = 0
    if preserve_switch:
        strong = evidence.strong_control
        if (
            not isinstance(strong, np.ndarray)
            or strong.shape != (len(baseline), 2)
            or strong.dtype != bool
        ):
            raise ValueError("supported switch preservation needs aligned observed strong control")
        support = _base_strong_transition_support(evidence, config)
        switches = np.flatnonzero(
            (baseline[1:] != baseline[:-1])
            & np.isin(baseline[:-1], (HOME, AWAY))
            & np.isin(baseline[1:], (HOME, AWAY))
        ) + 1
        for frame in switches:
            if support[frame, baseline[frame]] and np.any(
                editable[frame:frame + config["semi_markov_minimum_live_frames"]]
            ):
                editable[frame:frame + config["semi_markov_minimum_live_frames"]] = False
                protected += 1
    phase.diagnostics.update({
        "bounded_source_brackets": bracketed,
        "bounded_baseline_agreeing_brackets": accepted,
        "bounded_editable_live_frames": int(np.count_nonzero(editable)),
        "bounded_changed_live_frames": 0,
        **({"bounded_supported_switches_preserved": protected}
           if preserve_switch else {}),
        **({"incipient_opponent_veto_frames": int(np.count_nonzero(vetoed))}
           if veto else {}),
    })
    if not editable.any():
        return baseline
    fixed = baseline.copy()
    fixed[editable] = phase.team[editable] if motion_mode else UNKNOWN
    labels = decode_semi_markov(evidence, config, fixed_labels=fixed)
    if np.any(labels[~editable] != baseline[~editable]):
        raise ValueError("bounded phase decoding changed a fixed baseline frame")
    phase.diagnostics["bounded_changed_live_frames"] = int(np.count_nonzero(labels != baseline))
    return labels


def _hsmm_duration_scores(durations, state, config):
    mode_key = (
        "hsmm_dead_duration_mode_frames"
        if state == DEAD
        else "hsmm_live_duration_mode_frames"
    )
    mode = float(config[mode_key])
    sigma = float(config["hsmm_duration_log_sigma"])
    weight = float(config["hsmm_duration_weight"])
    if mode <= 0.0 or sigma <= 0.0 or weight < 0.0:
        raise ValueError(
            "HSMM duration mode and sigma must be positive and weight "
            "must be nonnegative"
        )
    log_ratio = np.log(durations.astype(np.float32) / mode)
    return -0.5 * weight * (log_ratio / sigma) ** 2


def _hsmm_candidate_frames(proposal, radius):
    if radius < 0:
        raise ValueError("HSMM boundary search radius must be nonnegative")
    n_frames = len(proposal)
    candidates = np.zeros(n_frames + 1, dtype=bool)
    candidates[[0, n_frames]] = True
    boundaries = np.flatnonzero(proposal[1:] != proposal[:-1]) + 1
    for boundary in boundaries:
        start = max(1, int(boundary) - radius)
        end = min(n_frames, int(boundary) + radius + 1)
        candidates[start:end] = True
    return np.flatnonzero(candidates)


def _hsmm_boundary_scores(emissions, config):
    n_frames = len(emissions)
    window = int(config["hsmm_boundary_evidence_window_frames"])
    weight = float(config["hsmm_boundary_evidence_weight"])
    maximum = float(config["hsmm_boundary_evidence_maximum"])
    if window < 1 or weight < 0.0 or maximum <= 0.0:
        raise ValueError(
            "HSMM boundary window and maximum must be positive and "
            "weight must be nonnegative"
        )

    margins = np.empty_like(emissions)
    margins[:, HOME] = emissions[:, HOME] - emissions[:, AWAY]
    margins[:, AWAY] = emissions[:, AWAY] - emissions[:, HOME]
    margins[:, DEAD] = (
        emissions[:, DEAD] - np.max(emissions[:, :2], axis=1)
    )
    cumulative = np.vstack((
        np.zeros((1, 3), dtype=np.float32),
        np.cumsum(margins, axis=0, dtype=np.float32),
    ))
    frames = np.arange(n_frames)
    pre_starts = np.maximum(0, frames - window)
    post_ends = np.minimum(n_frames, frames + window)
    pre_lengths = np.maximum(1, frames - pre_starts)
    post_lengths = np.maximum(1, post_ends - frames)
    pre = (
        cumulative[frames] - cumulative[pre_starts]
    ) / pre_lengths[:, None]
    post = (
        cumulative[post_ends] - cumulative[frames]
    ) / post_lengths[:, None]
    return weight * np.clip(post - pre, 0.0, maximum)


def decode_hsmm(evidence, config):
    emissions = _semi_markov_emissions(evidence, config)
    n_frames = len(emissions)
    if n_frames == 0:
        return np.empty(0, dtype=np.int8)

    minimum_live = int(config["hsmm_minimum_live_frames"])
    minimum_dead = int(config["hsmm_minimum_dead_frames"])
    maximum = int(config["hsmm_maximum_segment_frames"])
    if minimum_live < 2 or minimum_dead < 2:
        raise ValueError("HSMM minimum durations must be at least 2")
    if maximum < max(minimum_live, minimum_dead):
        raise ValueError(
            "HSMM maximum segment duration must cover minimum durations"
        )

    proposal = decode_semi_markov(evidence, config)
    candidates = _hsmm_candidate_frames(
        proposal, int(config["hsmm_boundary_search_radius_frames"])
    )
    n_candidates = len(candidates)
    prefix = np.vstack((
        np.zeros((1, 3), dtype=np.float32),
        np.cumsum(emissions, axis=0, dtype=np.float32),
    ))
    scores = np.full((n_candidates, 3), -np.inf, dtype=np.float32)
    segment_start = np.full((n_candidates, 3), -1, dtype=np.int32)
    previous_state = np.full((n_candidates, 3), -1, dtype=np.int8)
    strong_support = _semi_markov_strong_transition_support(
        evidence, config
    )
    boundary_scores = _hsmm_boundary_scores(emissions, config)
    live_cost = float(config["semi_markov_live_transition_cost"])
    strong_live_cost = float(
        config["semi_markov_strong_transition_cost"]
    )
    dead_cost = float(config["semi_markov_dead_transition_cost"])
    start_dead_bonus = float(config["semi_markov_start_dead_bonus"])
    minimums = (minimum_live, minimum_live, minimum_dead)

    for end_index in range(1, n_candidates):
        end = int(candidates[end_index])
        durations = end - candidates[:end_index]
        within_maximum = durations <= maximum
        for state in (HOME, AWAY, DEAD):
            valid = within_maximum & (durations >= minimums[state])
            start_indices = np.flatnonzero(valid)
            if start_indices.size == 0:
                continue
            starts = candidates[start_indices]
            entry = np.full((len(start_indices), 3), -np.inf, dtype=np.float32)
            initial = starts == 0
            if np.any(initial):
                entry[initial, state] = (
                    start_dead_bonus if state == DEAD else 0.0
                )

            noninitial = ~initial
            if np.any(noninitial):
                prior = scores[start_indices[noninitial]].copy()
                prior[:, state] = -np.inf
                transition = np.full_like(prior, dead_cost)
                if state in (HOME, AWAY):
                    other = AWAY if state == HOME else HOME
                    transition[:, other] = np.where(
                        strong_support[starts[noninitial], state],
                        strong_live_cost,
                        live_cost,
                    )
                entry[noninitial] = prior - transition

            best_sources = np.argmax(entry, axis=1)
            best_entry = entry[np.arange(len(entry)), best_sources]
            segment_scores = (
                best_entry
                + prefix[end, state]
                - prefix[starts, state]
                + _hsmm_duration_scores(durations[start_indices], state, config)
            )
            segment_scores[~initial] += boundary_scores[
                starts[~initial], state
            ]
            best = int(np.argmax(segment_scores))
            scores[end_index, state] = segment_scores[best]
            segment_start[end_index, state] = start_indices[best]
            previous_state[end_index, state] = best_sources[best]

    end_index = n_candidates - 1
    state = int(np.argmax(scores[end_index]))
    if not np.isfinite(scores[end_index, state]):
        raise ValueError(
            "HSMM candidate boundaries cannot form a complete sequence"
        )
    labels = np.empty(n_frames, dtype=np.int8)
    while end_index > 0:
        start_index = int(segment_start[end_index, state])
        start = int(candidates[start_index])
        end = int(candidates[end_index])
        labels[start:end] = state
        state = int(previous_state[end_index, state])
        end_index = start_index
    return labels


def _forced_dead_labels(evidence):
    # Forced dead frames constrain the live/dead pass; every other frame stays free.
    forced = evidence.forced_dead
    if forced is None:
        return None
    if not isinstance(forced, np.ndarray) or forced.dtype != bool or forced.shape != evidence.dead.shape:
        raise ValueError("forced dead frames must be an aligned boolean mask")
    if not forced.any():
        return None
    return np.where(forced, DEAD, FREE).astype(np.int8)


def decode(evidence, tag_features, config):
    validate_touch_motion_config(config)
    validate_owner_flight_config(config)
    validate_control_phase_config(config)
    mode = config["decoder_mode"]
    forced = _forced_dead_labels(evidence)
    if forced is not None and (mode != "semi_markov" or (
            config.get("semi_markov_phase_enabled", False) and config.get("phase_decode_mode") == "bounded")):
        raise ValueError("forced dead frames need global semi-Markov decoding")
    if mode == "state_machine":
        return _decode_state_machine(evidence, tag_features, config)
    if mode == "semi_markov":
        if config.get("semi_markov_phase_enabled", False) and config.get("phase_decode_mode") == "bounded":
            return _decode_bounded_control_phases(evidence, config)
        if config.get("semi_markov_phase_enabled", False) and config.get("phase_confirm_decoded_owner", False):
            phase = _validated_control_phase(evidence)
            initial = decode_semi_markov(
                replace(evidence, control_phase=None),
                {**config, "semi_markov_phase_enabled": False}, fixed_labels=forced,
            )
            flight_rows = np.flatnonzero(phase.kind == FLIGHT)
            invalid = flight_rows[initial[phase.anchor_rows[flight_rows]] != phase.team[flight_rows]]
            if invalid.size:
                kinds = phase.kind.copy()
                teams = phase.team.copy()
                anchors = phase.anchor_rows.copy()
                kinds[invalid] = UNSEEN
                teams[invalid] = UNKNOWN
                anchors[invalid] = -1
                evidence = replace(evidence, control_phase=replace(
                    phase, kind=kinds, team=teams, anchor_rows=anchors,
                ))
        if config.get("owner_flight_confirm_decoded_owner", False):
            priors, anchors = evidence.owner_flight_prior, evidence.owner_flight_anchor_rows
            n = len(evidence.dead)
            if (not isinstance(priors, np.ndarray) or priors.shape != (n, 2)
                    or not isinstance(anchors, np.ndarray) or anchors.shape != (n,)
                    or evidence.owner_flight_diagnostics is None):
                raise ValueError("decoded-owner confirmation needs aligned owner priors and contact anchors")
            used = np.any(priors > 0, axis=1)
            frames = np.arange(n)
            if (np.any((anchors[used] < 0) | (anchors[used] > frames[used]))
                    or np.any(anchors[~used] != -1)):
                raise ValueError("decoded-owner confirmation needs valid prior contact rows")
            first = decode_semi_markov(
                replace(evidence, owner_flight_prior=np.zeros_like(priors)), config,
                fixed_labels=forced,
            )
            owner = np.argmax(priors, axis=1)
            disagree = used.copy()
            disagree[used] = first[anchors[used]] != owner[used]
            confirmed = priors.copy()
            confirmed[disagree] = 0
            evidence.owner_flight_diagnostics.update({
                "first_pass_prior_frames_rejected": int(np.count_nonzero(disagree)),
                "first_pass_prior_frames_retained": int(np.count_nonzero(used & ~disagree)),
            })
            evidence = replace(evidence, owner_flight_prior=confirmed)
        if evidence.ballfree_penalty is not None:
            # Ball-free evidence only chooses the team: live/dead comes from a pass without it.
            first = decode_semi_markov(
                replace(evidence, ballfree_penalty=None, ballfree_support=None), config,
                fixed_labels=forced,
            )
            fixed = np.where(first == DEAD, DEAD, UNKNOWN).astype(np.int8)
            return decode_semi_markov(evidence, config, fixed_labels=fixed)
        return decode_semi_markov(evidence, config, fixed_labels=forced)
    if mode == "hsmm":
        return decode_hsmm(evidence, config)
    raise ValueError("unknown decoder_mode {!r}".format(mode))
