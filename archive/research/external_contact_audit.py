import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from possession import AWAY, DEAD, HOME, UNKNOWN
from possession.cli import load_config
from possession.control_phase import CONTACT, RECEPTION, TURNOVER
from possession.decoder import build_evidence, decode
from archive.research.jsonio import write_json_atomic
from possession.external_metrica_validation import (
    _file_sha256, _load_events, _load_game, _repair_by_period, config_at_fps,
)
from possession.external_transfer_check import restart_only_tags
from possession.tagging import infer_restart_ownership, localize_restarts
from possession.tracking import control_features_from_arrays


GROUP_QUOTAS = {
    "reception_disagreement": 4,
    "turnover_delayed": 4,
    "turnover_persistent": 4,
    "reception_agreement": 4,
    "turnover_agreement": 8,
}
REVIEW_FIELDS = (
    "status", "reviewer", "video_source", "permission_confirmed",
    "contact_team", "contact_frame", "possession_boundary_frame",
    "confidence", "notes",
)


def _source_candidates(data, baseline, config):
    fps = data["fps"]
    effective_base = config_at_fps(baseline, 25, fps)
    effective_phase = config_at_fps({**baseline, **config}, 25, fps)
    edges = np.r_[0, np.flatnonzero(np.diff(data["periods"])) + 1, len(data["frames"])]
    candidates, baseline_periods = [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        trajectory = _repair_by_period(data["ball"][start:stop], fps, effective_base)
        control = control_features_from_arrays(
            data["home"][start:stop], data["away"][start:stop],
            trajectory.xy, fps, effective_base,
        )
        tags = restart_only_tags(
            [], data["periods"][start:stop], fps, effective_base, None, 2027,
        )
        tags = infer_restart_ownership(
            localize_restarts(tags, trajectory, fps, effective_base),
            trajectory, control, effective_base,
        )
        baseline_evidence = build_evidence(
            control, trajectory, tags, (105., 68.), effective_base, fps=fps,
        )
        labels = decode(baseline_evidence, tags, effective_base)
        baseline_periods.append(labels)
        phase = build_evidence(
            control, trajectory, tags, (105., 68.), effective_phase, fps=fps,
        ).control_phase
        if phase is None or np.any(tags.dead_hint) or np.any(tags.restart_team != UNKNOWN):
            raise ValueError("contact audit must use source-only, no-tag evidence")
        for row in range(1, len(labels)):
            kind = int(phase.kind[row])
            if kind not in (CONTACT, RECEPTION, TURNOVER):
                continue
            receiver = int(phase.team[row])
            if (
                phase.kind[row - 1] in (CONTACT, RECEPTION, TURNOVER)
                and phase.team[row - 1] == receiver
                and control.nearest_player_slot[row - 1, receiver]
                == control.nearest_player_slot[row, receiver]
            ):
                continue
            agrees = labels[row] == receiver
            if kind == RECEPTION:
                group = "reception_agreement" if agrees else "reception_disagreement"
            elif kind == TURNOVER:
                if agrees:
                    group = "turnover_agreement"
                else:
                    future = labels[row + 1:min(len(labels), row + 21)]
                    group = "turnover_delayed" if np.any(future == receiver) else "turnover_persistent"
            else:
                continue
            absolute = int(start) + row
            candidates.append({
                "group": group, "row": absolute,
                "period": int(data["periods"][absolute]),
                "receiver_team": receiver, "baseline_team": int(labels[row]),
                "phase": kind,
                "nearest_distance_metres": float(control.nearest_distances[row, receiver, 0]),
                "nearest_track_slot": int(control.nearest_player_slot[row, receiver]),
                "nearest_track_age_frames": int(control.nearest_player_age_frames[row, receiver]),
                "ball_source": int(trajectory.source[row]),
            })
    return candidates, np.concatenate(baseline_periods)


def select_blinded_cases(candidates, seed=2027, spacing_frames=100):
    if not isinstance(seed, int) or isinstance(seed, bool) or spacing_frames < 1:
        raise ValueError("audit selection needs an integer seed and positive spacing")
    rng = np.random.default_rng(seed)
    selected = []
    for group, quota in GROUP_QUOTAS.items():
        pool = sorted(
            (item for item in candidates if item["group"] == group),
            key=lambda item: (item["period"], item["row"]),
        )
        if len(pool) < quota:
            raise ValueError("{}: only {} source-only candidates for {} tasks".format(
                group, len(pool), quota,
            ))
        periods = sorted({item["period"] for item in pool})
        picked = []
        for period in periods:
            if len(picked) >= quota:
                break
            choices = [item for item in pool if item["period"] == period]
            for index in rng.permutation(len(choices)):
                candidate = choices[index]
                if all(
                    item["period"] != period or abs(item["row"] - candidate["row"]) >= spacing_frames
                    for item in selected + picked
                ):
                    picked.append(candidate)
                    break
        for index in rng.permutation(len(pool)):
            if len(picked) >= quota:
                break
            candidate = pool[index]
            if candidate in picked:
                continue
            if all(
                item["period"] != candidate["period"]
                or abs(item["row"] - candidate["row"]) >= spacing_frames
                for item in selected + picked
            ):
                picked.append(candidate)
        if len(picked) != quota:
            raise ValueError("{}: cannot select {} nonoverlapping review windows".format(group, quota))
        selected.extend(picked)
    return sorted(selected, key=lambda item: (item["period"], item["row"]))


def validate_contact_audit_task(task, require_reviewed=False):
    if set(task) != {
        "schema_version", "dataset", "case_id", "period", "center_frame",
        "start_frame", "end_frame", "fps", "video_available",
        "review_instructions", "review",
    } or task["schema_version"] != 1 or task["dataset"] != "Metrica sample-data":
        raise ValueError("invalid or unblinded contact audit task")
    if (
        not isinstance(task["center_frame"], int) or not isinstance(task["start_frame"], int)
        or not isinstance(task["end_frame"], int)
        or not task["start_frame"] <= task["center_frame"] <= task["end_frame"]
        or not np.isfinite(task["fps"]) or task["fps"] <= 0
        or task["video_available"] is not False
    ):
        raise ValueError("contact audit task requires aligned external frame bounds and no implied video")
    review = task["review"]
    if set(review) != set(REVIEW_FIELDS) or review["status"] not in ("unreviewed", "unverifiable", "verified"):
        raise ValueError("invalid contact audit review")
    if review["status"] == "unreviewed":
        if any(value is not None for key, value in review.items() if key != "status"):
            raise ValueError("unreviewed contact audit cannot contain prefilled labels")
    elif review["status"] == "unverifiable":
        if review["contact_team"] is not None or review["contact_frame"] is not None or review["possession_boundary_frame"] is not None:
            raise ValueError("unverifiable contact audit cannot claim ground truth")
    else:
        if (
            not isinstance(review["reviewer"], str) or not review["reviewer"].strip()
            or not isinstance(review["video_source"], str) or not review["video_source"].strip()
            or review["permission_confirmed"] is not True
            or review["contact_team"] not in ("H", "A", "none", "ambiguous")
            or not isinstance(review["confidence"], (int, float))
            or isinstance(review["confidence"], bool)
            or not np.isfinite(review["confidence"]) or not 0 <= review["confidence"] <= 1
        ):
            raise ValueError("verified contact requires an independent permitted video and reviewer")
        for key in ("contact_frame", "possession_boundary_frame"):
            frame = review[key]
            if frame is not None and (
                not isinstance(frame, int) or not task["start_frame"] <= frame <= task["end_frame"]
            ):
                raise ValueError("{} is outside the review window".format(key))
        if (
            (review["contact_team"] in ("H", "A") and review["contact_frame"] is None)
            or (review["contact_team"] in ("none", "ambiguous")
                and review["contact_frame"] is not None)
        ):
            raise ValueError("verified contact frame must match a definite contact team")
    if require_reviewed and review["status"] == "unreviewed":
        raise ValueError("contact audit still needs independent review")
    return review["status"]


def _provider_context(events, data, selected):
    by_frame = {int(frame): row for row, frame in enumerate(data["frames"])}
    by_case, summary = {}, {}
    for case in selected:
        row = case["row"]
        nearby = []
        for event in events:
            anchor_frame = event["end_frame"] if event["type"] == "PASS" else event["frame"]
            event_row = by_frame.get(anchor_frame)
            if (
                event_row is None or event["period"] != case["period"]
                or data["periods"][event_row] != case["period"]
                or abs(event_row - row) > 12
            ):
                continue
            nearby.append({
                "type": event["type"], "team": event["team"],
                "anchor": "pass_end" if event["type"] == "PASS" else "event_start",
                "offset_frames": int(event_row - row),
            })
        by_case[case["case_id"]] = sorted(
            nearby, key=lambda item: (abs(item["offset_frames"]), item["type"], item["team"]),
        )
    for group in GROUP_QUOTAS:
        members = [case for case in selected if case["group"] == group]
        summary[group] = {
            "cases": len(members),
            "nearby_provider_recovery_same_team": sum(
                any(item["type"] == "RECOVERY"
                    and item["team"] == ("home" if case["receiver_team"] == HOME else "away")
                    for item in by_case[case["case_id"]])
                for case in members
            ),
            "nearby_provider_pass_end_same_team": sum(
                any(item["type"] == "PASS"
                    and item["team"] == ("home" if case["receiver_team"] == HOME else "away")
                    for item in by_case[case["case_id"]])
                for case in members
            ),
            "no_nearby_provider_event": sum(
                not by_case[case["case_id"]] for case in members
            ),
        }
    return by_case, summary


def build_external_contact_audit(root, workspace, benchmark_path):
    root = Path(root).resolve()
    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise FileExistsError("{}: audit workspace already exists; do not overwrite reviews".format(workspace))
    benchmark_path = Path(benchmark_path).resolve()
    benchmark = json.loads(benchmark_path.read_text())
    project = Path(__file__).resolve().parents[2]
    base_path = project / "configs" / "compliant_tracking.json"
    scenario_path = project / benchmark["scenario_protocol"]
    if (
        benchmark["development_game"] != "Sample_Game_1"
        or _file_sha256(base_path) != benchmark["baseline_sha256"]
        or _file_sha256(scenario_path) != benchmark["scenario_protocol_sha256"]
        or subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip() != benchmark["external_revision"]
        or any(
            _file_sha256(project / "possession" / name) != digest
            for name, digest in benchmark["model_files_sha256"].items()
        )
    ):
        raise ValueError("external audit requires pinned sample data, baseline, protocol and model code")
    data = _load_game(root, "Sample_Game_1")
    if abs(data["fps"] - benchmark["runtime_source_fps"]) > 1e-6:
        raise ValueError("external audit game frame rate differs from frozen runtime")
    base = load_config(base_path)
    candidates, labels = _source_candidates(data, base, benchmark["values"])
    baseline_digest = hashlib.sha256(labels.tobytes()).hexdigest()
    if baseline_digest != benchmark["game1"]["clean_no_tag_baseline_label_sha256"]:
        raise ValueError("no-tag baseline labels differ from the frozen external audit source")
    selected = select_blinded_cases(candidates)
    data_hashes = {path.name: _file_sha256(path) for path in data["input_paths"]}
    selected = [dict(item) for item in selected]
    edges = np.r_[0, np.flatnonzero(np.diff(data["periods"])) + 1, len(data["frames"])]
    tasks = []
    for index, case in enumerate(selected, start=1):
        case["case_id"] = "case-{:02d}".format(index)
        row = case["row"]
        start = int(edges[np.searchsorted(edges, row, side="right") - 1])
        stop = int(edges[np.searchsorted(edges, row, side="right")])
        task = {
            "schema_version": 1, "dataset": "Metrica sample-data",
            "case_id": case["case_id"], "period": case["period"],
            "center_frame": int(data["frames"][row]),
            "start_frame": int(data["frames"][max(start, row - 50)]),
            "end_frame": int(data["frames"][min(stop - 1, row + 50)]),
            "fps": float(data["fps"]), "video_available": False,
            "review_instructions": (
                "Do not infer physical contact from tracking proximity or provider events. "
                "Keep unreviewed until permitted synchronized external video is available; "
                "label before opening the separate provider-event context."
            ),
            "review": {key: ("unreviewed" if key == "status" else None) for key in REVIEW_FIELDS},
        }
        validate_contact_audit_task(task)
        tasks.append(task)
    events = _load_events(data["event_path"], {
        "PASS", "RECOVERY", "BALL LOST", "CARRY", "SET PIECE", "BALL OUT",
    })
    provider, provider_summary = _provider_context(events, data, selected)
    index = {
        "schema_version": 1, "dataset": "Metrica sample-data",
        "source_revision": benchmark["external_revision"],
        "benchmark_sha256": _file_sha256(benchmark_path),
        "generator_sha256": _file_sha256(Path(__file__)),
        "selection_policy": {
            "seed": 2027, "minimum_spacing_frames": 100,
            "group_quotas": GROUP_QUOTAS,
            "provider_events_used_for_selection": False,
            "tracking_only_cannot_verify_physical_touch": True,
            "independent_video_available": False,
        },
        "source_files_sha256": data_hashes,
        "baseline_labels_sha256": baseline_digest,
        "tasks": [task["case_id"] for task in tasks],
    }
    summary = {
        "schema_version": 1, "total_cases": len(tasks),
        "selection_groups": dict(Counter(case["group"] for case in selected)),
        "provider_event_proximity_only": provider_summary,
        "independent_contact_labels": 0,
        "physical_contact_verified": False,
        "source_benchmark_sha256": index["benchmark_sha256"],
        "source_files_sha256": data_hashes,
        "baseline_labels_sha256": index["baseline_labels_sha256"],
    }
    for value in (index, selected, provider, summary, *tasks):
        json.dumps(value, allow_nan=False)
    workspace.mkdir(parents=True)
    for task in tasks:
        write_json_atomic(workspace / "tasks" / (task["case_id"] + ".json"), task)
    write_json_atomic(workspace / "index.json", index)
    write_json_atomic(workspace / "selection_private.json", selected)
    write_json_atomic(workspace / "provider_context_private.json", provider)
    write_json_atomic(workspace / "summary.json", summary)
    return summary


def validate_contact_audit_workspace(workspace, require_reviewed=False):
    workspace = Path(workspace).resolve()
    index = json.loads((workspace / "index.json").read_text())
    if (
        index.get("schema_version") != 1
        or index.get("dataset") != "Metrica sample-data"
        or index.get("selection_policy", {}).get("provider_events_used_for_selection") is not False
        or index.get("selection_policy", {}).get("independent_video_available") is not False
    ):
        raise ValueError("contact audit index lacks blinded source-only provenance")
    statuses = Counter()
    for case_id in index["tasks"]:
        if not isinstance(case_id, str) or re.fullmatch(r"case-\d{2}", case_id) is None:
            raise ValueError("invalid contact audit case ID")
        task = json.loads((workspace / "tasks" / (case_id + ".json")).read_text())
        if task["case_id"] != case_id:
            raise ValueError("contact audit case ID differs from its index")
        statuses[validate_contact_audit_task(task, require_reviewed)] += 1
    return {"tasks": len(index["tasks"]), "review_status": dict(statuses)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare external, blinded Metrica contact audit")
    parser.add_argument("--root", help="official external Metrica sample-data checkout")
    parser.add_argument("--workspace", required=True, help="ignored local review workspace")
    parser.add_argument("--validate-only", action="store_true", help="validate existing blinded review cards")
    parser.add_argument("--require-reviewed", action="store_true", help="fail if any card remains unreviewed")
    parser.add_argument(
        "--benchmark", default=str(
            Path(__file__).resolve().parents[2] / "docs" / "benchmarks"
            / "reception_motion_validation.json"
        ),
    )
    args = parser.parse_args(argv)
    if args.validate_only:
        print(json.dumps(
            validate_contact_audit_workspace(args.workspace, args.require_reviewed),
            indent=2, sort_keys=True,
        ))
        return
    if not args.root or args.require_reviewed:
        parser.error("--root is required to build; --require-reviewed needs --validate-only")
    print(json.dumps(
        build_external_contact_audit(args.root, args.workspace, args.benchmark),
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
