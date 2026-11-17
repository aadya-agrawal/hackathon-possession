"""Ball-free possession evidence: which team has the ball, from player tracking alone.

Football cues used (all antisymmetric between teams, so swapping teams flips the answer):
collective movement toward a goal (the attacking team pushes forward and the defending team
retreats, so both move toward the defending team's goal), team spread (the team in possession
stretches the pitch), and pressure (opponents converge on the ball carrier).
"""
import numpy as np

FEATURES = (
    "movement_1s", "movement_3s", "spread_difference", "pressure_difference", "territory",
)


def attack_directions(home, away, period_bounds, pitch):
    """Per-frame +1 when home attacks toward x = length, else -1, from each half's goalkeepers."""
    direction = np.zeros(len(home))
    for start, stop in period_bounds:
        home_min = np.nanmedian(np.nanmin(home[start:stop, :, 0], axis=1))
        home_max = np.nanmedian(np.nanmax(home[start:stop, :, 0], axis=1))
        away_min = np.nanmedian(np.nanmin(away[start:stop, :, 0], axis=1))
        away_max = np.nanmedian(np.nanmax(away[start:stop, :, 0], axis=1))
        # The deepest player of each team sits nearest its own goal line.
        home_left = (home_min - 0) + (pitch[0] - away_max) < (pitch[0] - home_max) + (away_min - 0)
        direction[start:stop] = 1.0 if home_left else -1.0
    return direction


def _centred_mean(values, window):
    finite = np.isfinite(values)
    total = np.r_[0., np.cumsum(np.where(finite, values, 0.))]
    count = np.r_[0, np.cumsum(finite)]
    rows = np.arange(len(values))
    lower = np.clip(rows - window // 2, 0, len(values))
    upper = np.clip(rows + window - window // 2, 0, len(values))
    seen = count[upper] - count[lower]
    out = np.full(len(values), np.nan)
    np.divide(total[upper] - total[lower], seen, out=out, where=seen > 0)
    return out


def _pressure(team, opponents, radius):
    # Most opponents within `radius` of any one player of `team`.
    distance = np.linalg.norm(team[:, :, None, :] - opponents[:, None, :, :], axis=3)
    close = np.where(np.isfinite(distance), distance <= radius, False).sum(axis=2)
    return close.max(axis=1).astype(float)


def ballfree_features(home, away, direction, fps, pressure_radius=3.0):
    """(frames, len(FEATURES)) features, positive values favouring home possession."""
    players = np.concatenate([home, away], axis=1)
    centroid_x = np.nanmean(players[:, :, 0], axis=1)
    velocity = np.zeros(len(players))
    velocity[1:] = np.diff(centroid_x) * fps
    velocity[np.abs(velocity) > 10] = np.nan
    second = int(round(fps))
    spread = lambda team: np.nanstd(team[:, :, 0], axis=1) + np.nanstd(team[:, :, 1], axis=1)
    features = np.column_stack((
        _centred_mean(velocity * direction, second),
        _centred_mean(velocity * direction, 3 * second),
        _centred_mean(spread(home) - spread(away), second),
        _centred_mean(_pressure(home, away, pressure_radius) - _pressure(away, home, pressure_radius), second),
        _centred_mean((centroid_x - np.nanmean(centroid_x)) * direction / 10.0, second),
    ))
    return np.nan_to_num(features)


def home_probability(features, coefficients):
    return 1.0 / (1.0 + np.exp(-features @ np.asarray(coefficients, dtype=float)))


def ballfree_home_probability(players_xy, team_of_track, pitch, fps, period_bounds, coefficients):
    """Per-frame probability that home has the ball; attack directions are found per period."""
    home = players_xy[:, team_of_track == 0]
    away = players_xy[:, team_of_track == 1]
    direction = attack_directions(home, away, period_bounds, pitch)
    return home_probability(ballfree_features(home, away, direction, fps), coefficients)


def _sequence_probability(players_xy, team_of_track, pitch, fps, period_bounds, paths, ball_xy=None):
    from pathlib import Path
    from possession.ballfree_seq import sequence_home_probability
    # A list of weight files is an ensemble: probabilities are averaged in list order.
    paths = [paths] if isinstance(paths, str) else list(paths)
    root = Path(__file__).resolve().parents[1]
    probabilities = [
        sequence_home_probability(players_xy, team_of_track, pitch, fps, period_bounds,
                                  str(Path(path) if Path(path).is_absolute() else root / path), ball_xy)
        for path in paths
    ]
    return np.mean(probabilities, axis=0) if len(probabilities) > 1 else probabilities[0]


def ballfree_gap_frames(source, config):
    """Frames the decoder treats as long ball gaps (the ballfree_weight penalty path)."""
    from possession.decoder import _episodes
    gaps = np.zeros(len(source), dtype=bool)
    codes = (0, 4) if config.get("broadcast_ball_fusion_enabled", False) else (0,)
    for start, stop in _episodes(np.isin(source, codes), config.get("ballfree_minimum_unseen_frames", 1)):
        gaps[start:stop] = True
    return gaps


def configured_home_probability(players_xy, team_of_track, pitch, fps, period_bounds, config, trajectory=None):
    """P(home) from the configured model: logistic features, the players-only sequence CNN, or the
    ball-aware sequence CNN ("sequence_ball", which reads the repaired ball trajectory)."""
    model = config.get("ballfree_model", "logistic")
    if model == "sequence_ball":
        if trajectory is None:
            raise ValueError("the sequence_ball model needs the ball trajectory")
        ball = np.where(np.isfinite(trajectory.xy).all(axis=1, keepdims=True), trajectory.xy, np.nan)
        probability = _sequence_probability(players_xy, team_of_track, pitch, fps, period_bounds,
                                            config["ballfree_ball_weights_path"], ball)
        if config.get("ballfree_ball_scope", "all") == "gaps":
            # Ball-aware only inside long gaps; the players-only model keeps the tracked-frame path.
            players_only = _sequence_probability(players_xy, team_of_track, pitch, fps, period_bounds,
                                                 config["ballfree_weights_path"])
            probability = np.where(ballfree_gap_frames(trajectory.source, config), probability, players_only)
        return probability
    if model == "sequence":
        return _sequence_probability(players_xy, team_of_track, pitch, fps, period_bounds,
                                     config["ballfree_weights_path"])
    return ballfree_home_probability(players_xy, team_of_track, pitch, fps, period_bounds,
                                     config["ballfree_coefficients"])


def validate_ballfree_config(config):
    enabled = config.get("semi_markov_ballfree_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("semi_markov_ballfree_enabled must be boolean")
    if not enabled:
        return
    model = config.get("ballfree_model", "logistic")
    if model not in ("logistic", "sequence", "sequence_ball"):
        raise ValueError("ballfree_model must be logistic, sequence or sequence_ball")
    is_paths = lambda paths: isinstance(paths, str) or (  # noqa: E731
        isinstance(paths, list) and paths and all(isinstance(path, str) for path in paths))
    scope = config.get("ballfree_ball_scope", "all")
    if model == "sequence_ball" and scope not in ("all", "gaps"):
        raise ValueError("ballfree_ball_scope must be all or gaps")
    if model == "sequence_ball" and not is_paths(config.get("ballfree_ball_weights_path")):
        raise ValueError("sequence_ball model needs ballfree_ball_weights_path (a path or list of paths)")
    if (model == "sequence" or (model == "sequence_ball" and scope == "gaps")) and not is_paths(
            config.get("ballfree_weights_path")):
        raise ValueError("sequence ball-free model needs ballfree_weights_path (a path or list of paths)")
    coefficients = config.get("ballfree_coefficients")
    if (not isinstance(coefficients, list) or len(coefficients) != len(FEATURES)
            or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in coefficients)):
        raise ValueError("ballfree_coefficients must list one finite value per feature")
    for key in ("ballfree_weight", "ballfree_support_lean"):
        value = config.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError("{} must be finite and positive".format(key))
    ahead = config.get("ballfree_lookahead_frames", 0)
    if not isinstance(ahead, int) or isinstance(ahead, bool) or ahead < 0:
        raise ValueError("ballfree_lookahead_frames must be a non-negative integer")
    frames = config.get("ballfree_minimum_unseen_frames")
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
        raise ValueError("ballfree_minimum_unseen_frames must be a positive integer")
