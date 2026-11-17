"""Ball-free possession from a small sequence model over team-shape features (numpy inference).

Each team gets a "view": permutation-invariant summaries of its own and the opponent's shape in its
own attacking frame (centroid, spread, extent, velocity, runners, pressure, nearest-opponent
distance, share in the opponent half) plus the five hand-crafted ball-free features signed for that
team. A shared 1D CNN scores each view over a centred ~+-3 s window at 5 fps and
logit(home has ball) = f(home view) - f(away view), so swapping the teams gives exactly 1 - p.

Weights are trained on external matches only (possession/external_ballfree_seq.py); inference
needs numpy alone.

The ball-aware variant (weights with a "ball_channels" entry, trained by tools/train_ballfree_ball.py)
appends per-view ball channels from the repaired (noisy, gappy) ball trajectory: observed flag, ball
position relative to the attacking direction and both team centroids, distances to the nearest players
of each team, nearby-player counts and ball velocity; all zero where the ball is missing.
"""
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from possession.ballfree import attack_directions, ballfree_features

TRAINING_FPS = 25


def _smooth(values, window):
    # Centred NaN-aware running mean along axis 0
    flat = np.asarray(values, float).reshape(len(values), -1)
    finite = np.isfinite(flat)
    total = np.vstack([np.zeros((1, flat.shape[1])), np.cumsum(np.where(finite, flat, 0.), 0)])
    count = np.vstack([np.zeros((1, flat.shape[1])), np.cumsum(finite, 0)])
    rows = np.arange(len(flat))
    lo = np.clip(rows - window // 2, 0, len(flat))
    hi = np.clip(rows + window - window // 2, 0, len(flat))
    seen = count[hi] - count[lo]
    out = np.full(flat.shape, np.nan)
    np.divide(total[hi] - total[lo], seen, out=out, where=seen > 0)
    return out.reshape(np.shape(values))


def _team_frame(xy, sign, pitch):
    # Pitch-centred coordinates rotated so the team attacks towards +x
    return (xy - np.array([pitch[0] / 2, pitch[1] / 2])) * sign[:, None, None]


def _velocity(xy, fps, half):
    # Centred per-player velocity over 2*half frames (m/s); implausible values removed
    v = np.full(xy.shape, np.nan)
    if len(xy) > 2 * half:
        v[half:-half] = (xy[2 * half:] - xy[:-2 * half]) * fps / (2 * half)
    v[np.linalg.norm(v, axis=2) > 12] = np.nan
    return v


def _summaries(own, opp, own_v, opp_v):
    """Per-frame summaries of one team (own) and its opponent, in own attacking frame."""
    cols = []
    for team, vel in ((own, own_v), (opp, opp_v)):
        x, y = team[:, :, 0], team[:, :, 1]
        vx, vy = vel[:, :, 0], vel[:, :, 1]
        speed = np.hypot(vx, vy)
        cols += [
            np.nanmean(x, 1) / 10, np.nanmean(np.abs(y), 1) / 10,
            np.nanstd(x, 1) / 10, np.nanstd(y, 1) / 10,
            np.nanmax(x, 1) / 10, np.nanmin(x, 1) / 10,
            np.nanmean(vx, 1) / 3, np.nanmean(speed, 1) / 3,
            np.nanmean(np.where(np.isfinite(vx), vx > 2.0, np.nan), 1) * 3,
            np.nanmean(np.where(np.isfinite(vx), vx < -2.0, np.nan), 1) * 3,
            np.nanmax(speed, 1) / 5,
        ]
    # Pressure: most opponents within 3 m / 6 m of any own player; nearest-opponent distance stats
    d = np.linalg.norm(own[:, :, None, :] - opp[:, None, :, :], axis=3)
    fin = np.isfinite(d)
    for radius in (3.0, 6.0):
        cols.append(np.where(fin, d <= radius, False).sum(2).max(1) / 3.0)
    nearest = np.nanmin(np.where(fin, d, np.nan), axis=2)
    cols += [np.nanmin(nearest, 1) / 5, np.nanmean(nearest, 1) / 5]
    # Share of own players in the opponent half
    cols.append(np.nanmean(np.where(np.isfinite(own[:, :, 0]), own[:, :, 0] > 0, np.nan), 1))
    return np.column_stack(cols)


def _quiet_summaries(own, opp, own_v, opp_v):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _summaries(own, opp, own_v, opp_v)


def _chunked_summaries(own, opp, own_v, opp_v, chunk=20000):
    # Summaries are per frame, so chunking only bounds the pairwise-distance memory
    return np.concatenate([_quiet_summaries(own[i:i + chunk], opp[i:i + chunk], own_v[i:i + chunk], opp_v[i:i + chunk])
                           for i in range(0, len(own), chunk)])


_TASKS = []


def _run_task(index):
    function, args = _TASKS[index]
    return function(*args)


def _fork_map(tasks):
    """Results of function(*args) for each task, computed in forked workers when worthwhile.

    Every task is exactly the call the sequential path makes on the same arrays, so results are
    identical; the inputs are inherited by fork and only results cross processes."""
    workers = min(len(tasks), os.cpu_count() or 1, 32)
    if workers > 1:
        _TASKS[:] = tasks
        try:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
                return list(pool.map(_run_task, range(len(tasks))))
        except Exception:
            pass
        finally:
            _TASKS.clear()
    return [function(*args) for function, args in tasks]


BALL_NEAREST = 3          # nearest players per team whose ball distances enter the view
BALL_DISTANCE_CAP = 40.0  # metres
BALL_MAX_SPEED = 40.0     # m/s; faster finite-difference velocities are dropped


def _ball_summaries(own, opp, ball, ball_v):
    """Per-frame ball channels of one view (own attacking frame); NaN where the ball is missing."""
    seen = np.isfinite(ball[:, 0])
    bx, by = ball[:, 0], ball[:, 1]
    cols = [seen.astype(float), bx / 10, np.abs(by) / 10,
            (bx - np.nanmean(own[:, :, 0], 1)) / 10, (bx - np.nanmean(opp[:, :, 0], 1)) / 10]
    nearest = []
    for team in (own, opp):
        d = np.linalg.norm(team - ball[:, None, :], axis=2)
        d = np.sort(np.where(np.isfinite(d), np.minimum(d, BALL_DISTANCE_CAP), BALL_DISTANCE_CAP), axis=1)
        if d.shape[1] < BALL_NEAREST:
            d = np.pad(d, ((0, 0), (0, BALL_NEAREST - d.shape[1])), constant_values=BALL_DISTANCE_CAP)
        nearest.append(d[:, 0])
        cols += [d[:, i] / 5 for i in range(BALL_NEAREST)]
        cols += [(d <= 2.0).sum(1) / 3.0, (d <= 5.0).sum(1) / 3.0]
    cols.append((nearest[0] - nearest[1]) / 5)
    cols += [ball_v[:, 0] / 10, np.abs(ball_v[:, 1]) / 10, np.hypot(ball_v[:, 0], ball_v[:, 1]) / 10]
    out = np.column_stack(cols)
    out[~seen, 1:] = np.nan
    return out


def _ball_velocity(ball, fps, half):
    v = np.full(ball.shape, np.nan)
    if len(ball) > 2 * half:
        v[half:-half] = (ball[2 * half:] - ball[:-2 * half]) * fps / (2 * half)
    v[np.linalg.norm(v, axis=1) > BALL_MAX_SPEED] = np.nan
    return v


def _ball_views(home, away, ball, direction, pitch, fps, per5):
    """Smoothed ball channels of the home and away views."""
    ball = np.asarray(ball, dtype=float)
    ball = np.where(np.isfinite(ball).all(1, keepdims=True), ball, np.nan)
    bv = _ball_velocity(ball, fps, per5)
    centre = lambda xy, s: (xy - np.array([pitch[0] / 2, pitch[1] / 2])) * s[:, None]  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        BH = _ball_summaries(_team_frame(home, direction, pitch), _team_frame(away, direction, pitch),
                             centre(ball, direction), bv * direction[:, None])
        BA = _ball_summaries(_team_frame(away, -direction, pitch), _team_frame(home, -direction, pitch),
                             centre(ball, -direction), bv * -direction[:, None])
        return np.nan_to_num(_smooth(BH, per5)), np.nan_to_num(_smooth(BA, per5))


def _period_view_tasks(home, away, pitch, fps, chunk=20000, ball=None):
    """Independent tasks of period_views (hand features, then per-chunk summaries of each view)."""
    home = np.asarray(home, dtype=float)
    away = np.asarray(away, dtype=float)
    n = len(home)
    per5 = max(1, int(round(fps / 5)))
    direction = attack_directions(home, away, [(0, n)], pitch)
    hv, av = _velocity(home, fps, per5), _velocity(away, fps, per5)
    rot = lambda v, s: v * s[:, None, None]
    views = (
        (_team_frame(home, direction, pitch), _team_frame(away, direction, pitch),
         rot(hv, direction), rot(av, direction)),
        (_team_frame(away, -direction, pitch), _team_frame(home, -direction, pitch),
         rot(av, -direction), rot(hv, -direction)),
    )
    tasks = [(ballfree_features, (home, away, direction, fps))]
    for own, opp, own_v, opp_v in views:
        tasks += [(_quiet_summaries, (own[i:i + chunk], opp[i:i + chunk], own_v[i:i + chunk], opp_v[i:i + chunk]))
                  for i in range(0, n, chunk)]
    if ball is not None:
        tasks.append((_ball_views, (home, away, ball, direction, pitch, fps, per5)))
    return tasks, per5


def _assemble_period_views(results, per5, with_ball=False):
    if with_ball:
        results, (BH, BA) = results[:-1], results[-1]
    hand, summaries = results[0], results[1:]
    H = np.concatenate(summaries[:len(summaries) // 2])
    A = np.concatenate(summaries[len(summaries) // 2:])
    # Hand-crafted antisymmetric features enter each view with the view's sign
    H = np.nan_to_num(np.column_stack([_smooth(H, per5), hand])).astype(np.float32)
    A = np.nan_to_num(np.column_stack([_smooth(A, per5), -hand])).astype(np.float32)
    if not with_ball:
        return H, A
    return np.column_stack([H, BH]).astype(np.float32), np.column_stack([A, BA]).astype(np.float32)


def period_views(home, away, pitch, fps=TRAINING_FPS, ball=None):
    """(home view, away view) float32 (frames, channels) for one period of player positions.

    With `ball` (frames, 2) the ball channels are appended to each view (ball-aware model)."""
    tasks, per5 = _period_view_tasks(home, away, pitch, fps, ball=ball)
    return _assemble_period_views([function(*args) for function, args in tasks], per5, ball is not None)


def load_weights(path):
    with np.load(path) as z:
        weights = {key: z[key] for key in z.files}
    for key in list(weights):
        if weights[key].dtype.kind == "f":
            weights[key] = weights[key].astype(np.float32)
    return weights


def _gelu(x):
    # Exact GELU 0.5 x (1 + erf(x / sqrt 2)); erf from Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
    z = np.abs(x) * np.float32(1 / np.sqrt(2))
    t = 1 / (1 + np.float32(0.3275911) * z)
    poly = t * (np.float32(0.254829592) + t * (np.float32(-0.284496736) + t * (
        np.float32(1.421413741) + t * (np.float32(-1.453152027) + t * np.float32(1.061405429)))))
    erf = np.sign(x) * (1 - poly * np.exp(-z * z))
    return np.float32(0.5) * x * (1 + erf)


def view_score(view, weights):
    """CNN score per step for one (steps, channels) view sequence; zero padding like torch Conv1d."""
    h = (view - weights["mean"]) / weights["std"]
    h = _gelu(h @ weights["input_weight"].T + weights["input_bias"])
    for i, dilation in enumerate(weights["dilations"]):
        w = weights["conv%d_weight" % i]  # (out, in, k)
        k = w.shape[2]
        pad = int(dilation) * (k // 2)
        padded = np.pad(h, ((pad, pad), (0, 0)))
        out = np.broadcast_to(weights["conv%d_bias" % i], (len(h), w.shape[0])).copy()
        for j in range(k):
            out += padded[j * dilation:j * dilation + len(h)] @ w[:, :, j].T
        h = _gelu(out)
    return (h @ weights["output_weight"].T + weights["output_bias"])[:, 0]


def views_home_probability(H, A, weights, step):
    """Per-frame P(home) for one period: each of `step` phases is scored at 5 fps and interleaved."""
    p = np.empty(len(H))
    for o in range(step):
        logit = view_score(H[o::step], weights) - view_score(A[o::step], weights)
        p[o::step] = 1 / (1 + np.exp(-logit.astype(float)))
    return p


def sequence_home_probability(players_xy, team_of_track, pitch, fps, period_bounds, weights_path, ball_xy=None):
    """Per-frame probability that home has the ball; frames outside period_bounds get 0.5.

    Ball-aware weights (a "ball_channels" entry) need `ball_xy` (frames, 2), NaN where unseen."""
    weights = load_weights(weights_path)
    uses_ball = int(weights.pop("ball_channels", 0)) > 0
    if uses_ball and ball_xy is None:
        raise ValueError("ball-aware sequence weights need the ball trajectory")
    team_of_track = np.asarray(team_of_track)
    home = players_xy[:, team_of_track == 0]
    away = players_xy[:, team_of_track == 1]
    step = max(1, int(round(fps / 5)))
    p = np.full(len(players_xy), 0.5)
    periods = [(start, stop) for start, stop in period_bounds if stop - start >= 1]
    # Periods, views and phases are independent: run the same calls as period_views and
    # views_home_probability in parallel, then assemble them in the original order.
    ball = np.asarray(ball_xy, dtype=float) if uses_ball else None
    planned = [_period_view_tasks(home[start:stop], away[start:stop], pitch, fps,
                                  ball=None if ball is None else ball[start:stop]) for start, stop in periods]
    results = _fork_map([task for tasks, _ in planned for task in tasks])
    views = []
    for tasks, per5 in planned:
        views.append(_assemble_period_views(results[:len(tasks)], per5, uses_ball))
        results = results[len(tasks):]
    scores = iter(_fork_map([(view_score, (view[o::step], weights))
                             for H, A in views for o in range(step) for view in (H, A)]))
    for (start, stop), (H, A) in zip(periods, views):
        probability = np.empty(len(H))
        for o in range(step):
            logit = next(scores) - next(scores)
            probability[o::step] = 1 / (1 + np.exp(-logit.astype(float)))
        p[start:stop] = probability
    return p
