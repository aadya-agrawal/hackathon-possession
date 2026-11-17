"""Fit ball-free possession coefficients from external matches only (Metrica sample, DFL).

python -m possession.external_ballfree --metrica-root /path/to/metrica-sample-data \
    --dfl /path/DFL_02_01_matchinformation_...xml:/path/DFL_04_03_positions_raw_observed_...xml ...

Labels are provider possession references on other games; nothing from the challenge match is used.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from possession import AWAY, HOME
from possession.ballfree import FEATURES, attack_directions, ballfree_features

METRICA_GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")


def _period_features(home, away, pitch, fps):
    home, away = home.astype(float), away.astype(float)
    direction = attack_directions(home, away, [(0, len(home))], pitch)
    return ballfree_features(home, away, direction, fps)


def metrica_samples(root, protocol):
    from possession.external_metrica_validation import _load_events, _load_game
    from possession.external_possession_events import possession_chain_references
    for game in METRICA_GAMES:
        data = _load_game(Path(root), game)
        events = _load_events(data["event_path"], {
            "PASS", "CARRY", "RECOVERY", "BALL LOST", "BALL OUT", "SHOT", "SET PIECE", "FAULT RECEIVED",
        })
        truth, _ = possession_chain_references(events, data["frames"], data["periods"], data["fps"],
                                               protocol["references"])
        edges = np.r_[0, np.flatnonzero(np.diff(data["periods"])) + 1, len(truth)]
        for start, stop in zip(edges[:-1], edges[1:]):
            yield (_period_features(data["home"][start:stop], data["away"][start:stop], (105.0, 68.0), data["fps"]),
                   truth[start:stop])


def dfl_samples(pairs):
    from possession.external_dfl_validation import load_dfl_match
    for metadata, positions in pairs:
        pitch, periods = load_dfl_match(metadata, positions)
        for period in periods:
            yield _period_features(period["home"], period["away"], pitch, 25), period["truth"]


def fit_logistic(features, home, iterations=50):
    """L2-regularised logistic regression without intercept (sum log-loss + 0.5 |w|^2), Newton steps."""
    w = np.zeros(features.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-features @ w))
        gradient = features.T @ (p - home) + w
        hessian = (features * (p * (1 - p))[:, None]).T @ features + np.eye(len(w))
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def fit(samples, stride=5):
    X, y = [], []
    for features, truth in samples:
        live = np.isin(truth, (HOME, AWAY))
        X.append(features[live][::stride])
        y.append((truth[live][::stride] == HOME).astype(float))
    return fit_logistic(np.concatenate(X), np.concatenate(y))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrica-root", required=True)
    parser.add_argument("--protocol", default="configs/metrica_core_development.json")
    parser.add_argument("--dfl", action="append", default=[], help="METADATA.xml:POSITIONS.xml")
    args = parser.parse_args(argv)
    protocol = json.loads(Path(args.protocol).read_text())
    pairs = [item.split(":", 1) for item in args.dfl]
    samples = list(metrica_samples(args.metrica_root, protocol)) + list(dfl_samples(pairs))
    coefficients = fit(samples)
    print(json.dumps({"features": list(FEATURES), "ballfree_coefficients": [round(float(c), 6) for c in coefficients]}))


if __name__ == "__main__":
    main()
