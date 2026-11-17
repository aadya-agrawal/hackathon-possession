"""External guard evaluation of a candidate config against a baseline (Metrica guards + DFL).

Runs, in parallel and with caching, the guarded Metrica calibration harness
(possession.external_guarded_calibration: 8 degradation scenarios x 60 guards per Metrica sample
game, each game evaluated as the development game with the other two excluded) and the clean DFL
match path (live-spell F1 over IoU 0.3..0.7, dead recall, false dead, live team accuracy), and
prints baseline vs candidate per game/match. This is the evaluator behind every promotion record
("guards N/60", "objective a -> b", DFL ranking F1).

Ball-free sequence probabilities on Metrica use leave-this-game-out weights from --fold-dir
(players_ex_<game>.npz for ballfree_model "sequence", ball_ex_<game>.npz for "sequence_ball";
train_folds.py builds them). They are only needed when the config enables the ball-free cue.

Caching: prepared Metrica scenarios are keyed by the baseline config and the data-preparation
sources only (decoder edits do not invalidate them); baseline results are keyed by all SOURCES.
Scores are identical to the serial harness.

Usage:
  python validation/fast_eval.py --baseline configs/submission_ball_aware.json \
      --candidate path/to/candidate.json --out result.json \
      --metrica-root $METRICA_ROOT --dfl-root $DFL_ROOT --fold-dir $FOLD_DIR \
      [--cache /dev/shm/<user>_valcache] [--workers 7] [--no-dfl] [--no-metrica] \
      [--metrica-games Sample_Game_1] [--dfl-matches J03WMX:000001,...] [--labels-dir DIR] [--repo WORKTREE]
Env fallbacks: METRICA_ROOT, DFL_ROOT, FOLD_DIR, VALIDATION_CACHE, MTAH_REPO (see common.py).
"""
import argparse
import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import common

GAMES = common.METRICA_GAMES
DFL = tuple((m, common.DFL_COMPETITION[m]) for m in common.DFL_DEV)
SOURCES = ("possession/decoder.py", "possession/tracking.py", "possession/trajectory.py",
           "possession/tagging.py", "possession/control_phase.py",
           "possession/external_guarded_calibration.py", "possession/external_metrica_validation.py",
           "possession/external_possession_events.py", "possession/external_transfer_check.py",
           "possession/external_dfl_validation.py", "configs/metrica_core_development.json")
# _prepare loads data, builds control features and localizes tags but never decodes.
PREP_SOURCES = tuple(source for source in SOURCES if source != "possession/decoder.py")
# Filled in main(); worker processes inherit it (fork).
SETTINGS = {}

_sha = common.sha256


def _key(*parts, sources=SOURCES):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if isinstance(part, bytes) else str(part).encode())
    for source in sources:
        digest.update((SETTINGS["repo"] / source).read_bytes())
    return digest.hexdigest()[:24]


def _cached(name, key, build):
    return common.cached(SETTINGS["cache"], name, key, build)


def _save_labels(labels_dir, name, before, after):
    if labels_dir:
        Path(labels_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(labels_dir) / "{}.npz".format(name),
                            **{"baseline:" + k: v for k, v in before.items()},
                            **{"candidate:" + k: v for k, v in after.items()})


def _inject_ballfree(game):
    # The guarded harness has no player arrays at build_evidence, so probabilities are computed inside
    # the wrapper for each case trajectory with leave-this-game-out weights, on clean players.
    import possession.external_guarded_calibration as harness
    from possession.ballfree import configured_home_probability
    fold_dir = SETTINGS["fold_dir"]
    data = pickle.load(open(next(Path(SETTINGS["cache"]).glob("metrica-prepared-{}-*.pkl".format(game))), "rb"))["data"]
    edges = np.r_[0, np.flatnonzero(np.diff(data["periods"])) + 1, len(data["periods"])]
    players = {b - a: np.concatenate([data["home"][a:b], data["away"][a:b]], 1).astype(float)
               for a, b in zip(edges[:-1], edges[1:])}
    team_of = np.r_[np.zeros(data["home"].shape[1], int), np.ones(data["away"].shape[1], int)]
    memo = {}
    original = harness.build_evidence

    def build_evidence(control, trajectory, tags, pitch, config, **kwargs):
        prob = None
        if config.get("semi_markov_ballfree_enabled", False):
            if fold_dir is None:
                raise SystemExit("the config enables the ball-free cue: pass --fold-dir or set FOLD_DIR")
            fold = dict(config)
            fold["ballfree_weights_path"] = str(fold_dir / "players_ex_{}.npz".format(game))
            fold["ballfree_ball_weights_path"] = str(fold_dir / "ball_ex_{}.npz".format(game))
            n = len(trajectory.source)
            key = (fold.get("ballfree_model"), fold.get("ballfree_ball_scope"), n,
                   hashlib.sha1(np.ascontiguousarray(trajectory.xy).tobytes() + trajectory.source.tobytes()).hexdigest()
                   if fold.get("ballfree_model") == "sequence_ball" else None)
            if key not in memo:
                memo[key] = configured_home_probability(players[n], team_of, pitch, 25, [(0, n)], fold,
                                                        trajectory=trajectory)
            prob = memo[key]
        return original(control, trajectory, tags, pitch, config, ballfree_home_probability=prob, **kwargs)
    harness.build_evidence = build_evidence


def _metrica(job):
    game, baseline_path, candidate_path, labels_dir = job
    common.use_repo(SETTINGS["repo"])
    from possession.cli import load_config
    from possession.external_guarded_calibration import _evaluate, _prepare, guarded_checks
    baseline_bytes = Path(baseline_path).read_bytes()
    protocol = json.loads(Path("configs/metrica_core_development.json").read_text())
    protocol.update(
        source_fps=25, baseline_config=str(baseline_path), baseline_sha256=_sha(baseline_bytes),
        development_game=game, excluded_games=[g for g in GAMES if g != game],
        diagnostic_spell_ious=[.3, .4, .5, .6, .7],
    )
    baseline = load_config(baseline_path)
    candidate = load_config(candidate_path)
    prepared = _cached("metrica-prepared-" + game, _key(game, baseline_bytes, sources=PREP_SOURCES),
                       lambda: _prepare(SETTINGS["metrica_root"], protocol, baseline))
    _inject_ballfree(game)
    before, before_labels = _cached("metrica-baseline-labels-" + game, _key(game, baseline_bytes),
                                    lambda: _evaluate(prepared, baseline, protocol, include_labels=True))
    after, after_labels = _evaluate(prepared, candidate, protocol, include_labels=True)
    _save_labels(labels_dir, "metrica-" + game, before_labels, after_labels)
    checks = guarded_checks(after, before, protocol)
    dead = lambda result: result["full_timeline_diagnostic"]["dead"]
    return game, {
        "objective": [before["objective"], after["objective"]],
        "guards": "{}/{}".format(sum(c["passed"] for c in checks), len(checks)),
        "failed": ["{} {:.4f} vs {:.4f}".format(c["name"], c["actual"], c.get("minimum", c.get("maximum")))
                   for c in checks if not c["passed"]],
        "cases": {
            name: {
                "ranking_iou_mean": [
                    round(before["cases"][name]["official_style_reference_spell"]["mean_f1"], 4),
                    round(case["official_style_reference_spell"]["mean_f1"], 4),
                ],
                "dead_recall": [
                    round(dead(before["cases"][name])["dead_recall_on_known_frames"], 4),
                    round(dead(case)["dead_recall_on_known_frames"], 4),
                ],
                "false_dead_live": [
                    dead(before["cases"][name])["false_dead_on_known_live_frames"],
                    dead(case)["false_dead_on_known_live_frames"],
                ],
                "labels_changed": before["cases"][name]["label_sha256"] != case["label_sha256"],
            }
            for name, case in after["cases"].items()
        },
    }


def _dfl_models(match, competition, config_path):
    common.use_repo(SETTINGS["repo"])
    from possession.cli import load_config
    from possession.decoder import build_evidence, decode
    from possession.external_dfl_validation import score_dfl_labels
    from possession.external_transfer_check import restart_only_tags
    from possession.tagging import infer_restart_ownership, localize_restarts
    from possession.tracking import control_features_from_arrays
    from possession.trajectory import repair_ball_trajectory
    config = load_config(config_path)
    pitch, periods = common.load_dfl_cached(SETTINGS["dfl_root"], SETTINGS["cache"], match, competition)
    labels, lengths = [], []
    # Mirrors evaluate_dfl_match for a single config.
    for index, period in enumerate(periods):
        n = len(period["ball"])
        lengths.append(n)
        trajectory = repair_ball_trajectory(
            period["ball"], 25, max_gap_frames=config["max_interpolation_gap_frames"],
            max_speed=config["max_ball_speed_metres_per_second"],
            prediction_history_frames=config["prediction_history_frames"],
            prediction_horizon_frames=config["prediction_horizon_frames"],
            max_acceleration=config["max_ball_acceleration_metres_per_second_squared"],
            pitch=pitch, pitch_margin=config["prediction_pitch_margin_metres"],
        )
        control = control_features_from_arrays(period["home"], period["away"], trajectory.xy, 25, config)
        tags = restart_only_tags([], np.full(n, index + 1, dtype=np.int8), 25, config, None, 2027)
        tags = infer_restart_ownership(localize_restarts(tags, trajectory, 25, config),
                                       trajectory, control, config)
        labels.append(decode(build_evidence(control, trajectory, tags, pitch, config, fps=25), tags, config))
    labels = np.concatenate(labels)
    truth = np.concatenate([period["truth"] for period in periods])
    return {**score_dfl_labels(labels, truth, lengths), "label_sha256": _sha(labels.tobytes()),
            "labels": labels}


def _dfl(job):
    match, competition, baseline_path, candidate_path, labels_dir = job
    baseline_bytes = Path(baseline_path).read_bytes()
    before = _cached("dfl-baseline-labels-" + match, _key(match, baseline_bytes),
                     lambda: _dfl_models(match, competition, baseline_path))
    after = _dfl_models(match, competition, candidate_path)
    _save_labels(labels_dir, "dfl-" + match, {"match": before["labels"]}, {"match": after["labels"]})
    mean = lambda s: float(np.mean([s["live_spell_f1"][str(v)]["f1"] for v in (.3, .4, .5, .6, .7)]))
    return match, {
        "ranking_iou_mean": [round(mean(before), 4), round(mean(after), 4)],
        "f1_0.75": [round(before["live_spell_f1"]["0.75"]["f1"], 4), round(after["live_spell_f1"]["0.75"]["f1"], 4)],
        "predicted_spells": [before["predicted_live_spells"], after["predicted_live_spells"]],
        "dead_recall": [round(before["dead"]["dead_recall_on_known_frames"], 4),
                        round(after["dead"]["dead_recall_on_known_frames"], 4)],
        "false_dead_live": [before["dead"]["false_dead_on_known_live_frames"],
                            after["dead"]["false_dead_on_known_live_frames"]],
        "live_team_accuracy": [round(before["live_team_accuracy"], 4), round(after["live_team_accuracy"], 4)],
        "labels_changed": before["label_sha256"] != after["label_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--no-dfl", action="store_true")
    parser.add_argument("--no-metrica", action="store_true")
    parser.add_argument("--metrica-games", help="comma-separated subset of " + ",".join(GAMES))
    parser.add_argument("--labels-dir", help="save per-scenario baseline/candidate labels as .npz")
    parser.add_argument("--dfl-matches", help="comma-separated MATCH:COMPETITION pairs overriding the default DFL set")
    parser.add_argument("--metrica-root", help="Metrica sample-data root (env METRICA_ROOT)")
    parser.add_argument("--dfl-root", help="DFL XML folder (env DFL_ROOT)")
    parser.add_argument("--fold-dir", help="out-of-fold ball-free weights (env FOLD_DIR)")
    parser.add_argument("--cache", help="cache directory (env VALIDATION_CACHE)")
    parser.add_argument("--repo", help="repository/worktree to evaluate (env MTAH_REPO; default this repo)")
    args = parser.parse_args()
    baseline = str(Path(args.baseline).resolve())
    candidate = str(Path(args.candidate).resolve())
    labels_dir = str(Path(args.labels_dir).resolve()) if args.labels_dir else None
    SETTINGS.update(
        repo=common.repo_root(args.repo), cache=common.env_path("VALIDATION_CACHE", args.cache),
        fold_dir=common.env_path("FOLD_DIR", args.fold_dir, required=False),
        metrica_root=None if args.no_metrica else common.env_path("METRICA_ROOT", args.metrica_root),
        dfl_root=None if args.no_dfl else common.env_path("DFL_ROOT", args.dfl_root))
    games = args.metrica_games.split(",") if args.metrica_games else GAMES
    jobs = [] if args.no_metrica else [("metrica", (game, baseline, candidate, labels_dir)) for game in games]
    if not args.no_dfl:
        matches = [tuple(pair.split(":")) for pair in args.dfl_matches.split(",")] if args.dfl_matches else DFL
        jobs += [("dfl", (match, comp, baseline, candidate, labels_dir)) for match, comp in matches]
    report = {"baseline": baseline, "baseline_sha256": _sha(Path(baseline).read_bytes()),
              "candidate": candidate, "candidate_sha256": _sha(Path(candidate).read_bytes()),
              "metrica": {}, "dfl": {}}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [(kind, pool.submit(_metrica if kind == "metrica" else _dfl, job)) for kind, job in jobs]
        for kind, future in futures:
            name, result = future.result()
            report[kind][name] = result
            print(kind, name, json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
