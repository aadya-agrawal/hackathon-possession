import argparse
import csv
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from possession.cli import load_config
from possession.external_metrica_validation import _file_sha256
from possession.submission import validate_submission_file


def _ledger_evidence(project, family, variant):
    path = project / "experiments" / "results.csv"
    with path.open(newline="") as handle:
        for line, record in enumerate(csv.DictReader(handle), start=2):
            if record["family"] == family and record["variant"] == variant:
                return {
                    "source": "experiments/results.csv:{}".format(line),
                    "timestamp": record["recorded_at_utc"],
                    "decision": record["decision"],
                    "notes": record["notes"],
                }
    return {"source": "experiments/results.csv", "status": "lineage_not_found"}


def _plan_evidence(project, phrase):
    path = project / "docs" / "archive" / "IMPLEMENTATION_PLAN.md"
    for line, text in enumerate(path.read_text().splitlines(), start=1):
        if phrase in text:
            return {"source": "docs/archive/IMPLEMENTATION_PLAN.md:{}".format(line), "text": text.strip()}
    return {"source": "docs/archive/IMPLEMENTATION_PLAN.md", "status": "lineage_not_found"}


def _provenance_groups(project):
    return [
        (
            "semi_markov_costs_and_support",
            ("decoder_mode", "semi_markov_live_transition_cost",
             "semi_markov_strong_transition_cost", "semi_markov_strong_transition_window_frames",
             "semi_markov_strong_transition_minimum_frames",
             "semi_markov_minimum_live_frames", "semi_markov_minimum_dead_frames"),
            [
                _ledger_evidence(project, "semi-markov", "adaptive-strong-control-v1"),
                _plan_evidence(project, "Raising transition cost to 30"),
            ],
        ),
        (
            "restart_localization",
            ("restart_localization_enabled", "restart_offset_frames",
             "restart_localization_search_radius_frames", "restart_localization_motion_window_frames",
             "restart_localization_min_speed_gain_metres_per_second"),
            [_ledger_evidence(project, "restart-localization", "displacement-confidence-v3")],
        ),
        (
            "restart_ownership",
            ("restart_ownership_enabled", "restart_ownership_max_distance_metres",
             "restart_ownership_distance_margin_metres", "restart_ownership_minimum_votes",
             "restart_ownership_minimum_winner_fraction"),
            [_ledger_evidence(project, "restart-ownership", "strict-geometry-v1")],
        ),
        (
            "prediction_history_and_horizon",
            ("prediction_history_frames", "prediction_horizon_frames"),
            [_plan_evidence(project, "seven-frame quadratic fit outperformed")],
        ),
    ]


def _verified_kinematics(project, config):
    external = json.loads((project / "artifacts" / "external_parameter_validation.json").read_text())
    metrics_path = Path(config["compliance_provenance"]["external_metrics"])
    metrics = json.loads(metrics_path.read_text())
    keys = (
        "max_interpolation_gap_frames", "max_ball_speed_metres_per_second",
        "max_ball_acceleration_metres_per_second_squared",
    )
    verified = (
        external["passed"] and metrics["external_data"]
        and not metrics["policy"]["test_split_used"]
        and not metrics["policy"]["detector_outputs_used"]
        and _file_sha256(metrics_path) == external["external_metrics_sha256"]
        and all(config[key] == external["selected"][key] for key in keys)
    )
    return bool(verified), keys


def _inactive_parameters(config):
    return {
        key for key in config
        if key.startswith(("video_", "hsmm_", "motion_coherence_"))
        or key in {
            "live_confirm_frames", "dead_confirm_frames", "transition_confirm_frames",
            "strong_transition_confirm_frames", "dead_threshold",
            "strong_control_relative_speed_metres_per_second", "max_player_speed_metres_per_second",
        }
    }


def audit_active_parameters(project, protocol, external_root):
    project = Path(project)
    external_root = Path(external_root)
    candidate_path = project / protocol["candidate_config"]
    if _file_sha256(project / protocol["baseline_config"]) != protocol["baseline_sha256"]:
        raise ValueError("baseline changed before provenance audit")
    if _file_sha256(candidate_path) != protocol["candidate_sha256"]:
        raise ValueError("candidate changed before provenance audit")
    config = load_config(candidate_path)
    core_release = protocol.get("candidate_scope") == "core_release"
    lineage = config[
        "core_development_provenance" if core_release else "guarded_development_provenance"
    ]
    development_path = (
        Path(lineage["development_report"]) if core_release else
        project / "artifacts" / "development_guarded_possession.json"
    )
    development = json.loads(development_path.read_text())
    overrides = lineage["selected_overrides"]
    selected = next(
        item for item in development["experiments"]
        if item["name"] == development["selected_candidate"]
    )
    inputs_verified = bool(development["input_sha256"]) and all(
        name.startswith("Sample_Game_1")
        and _file_sha256(external_root / "data" / "Sample_Game_1" / name) == digest
        for name, digest in development["input_sha256"].items()
    )
    selection_verified = (
        development["selected_config_sha256"] == protocol["candidate_sha256"]
        and development["development_game"] == "Sample_Game_1"
        and development["selected_candidate"] == lineage["selected_candidate"]
        and selected["overrides"] == overrides
        and development["baseline_sha256"] == protocol["baseline_sha256"]
        and lineage["baseline_sha256"] == protocol["baseline_sha256"]
        and development["protocol_sha256"] == lineage["protocol_sha256"]
        and all(config[key] == value for key, value in overrides.items())
        and all(check["passed"] for check in selected["checks"])
        and inputs_verified
        and _file_sha256(
            Path(lineage["development_protocol"]) if core_release else
            project / "configs" / "metrica_guarded_development.json"
        )
        == lineage["protocol_sha256"]
    )
    kinematics_verified, kinematic_keys = _verified_kinematics(project, config)
    groups = _provenance_groups(project)
    verified_keys = (
        (set(overrides) if selection_verified else set())
        | (set(kinematic_keys) if kinematics_verified else set())
    )
    unresolved = [
        {
            "block": name,
            "values": {key: config[key] for key in keys},
            "externally_verified_values": {
                key: config[key] for key in keys if key in verified_keys
            },
            "unverified_values": {
                key: config[key] for key in keys if key not in verified_keys
            },
            "status": "external_only_reselection_not_verified",
            "evidence": evidence,
        } for name, keys, evidence in groups if not set(keys).issubset(verified_keys)
    ]
    traced_keys = {key for _, keys, _ in groups for key in keys}
    inactive = _inactive_parameters(config)
    metadata = {"compliance_provenance", "guarded_development_provenance", "core_development_provenance"}
    return {
        "schema_version": 1,
        "candidate_sha256": protocol["candidate_sha256"],
        "new_gap_overrides_externally_verified": bool(selection_verified) if not core_release else None,
        "core_overrides_externally_verified": bool(selection_verified) if core_release else None,
        "kinematic_overrides_externally_verified": bool(kinematics_verified),
        "verified_external_values": {
            key: config[key] for key in sorted(verified_keys)
        },
        "unresolved_active_blocks": unresolved,
        "remaining_inherited_keys_not_fully_traced": sorted(
            set(config) - verified_keys - traced_keys - inactive - metadata
        ),
        "inactive_for_this_runtime": sorted(inactive),
        "video_fusion_disabled": config["video_ball_observations_path"] is None,
        "active_tuned_parameter_provenance_cleared": bool(
            selection_verified and kinematics_verified and not unresolved
        ),
        "shared_baseline_issue": True,
        "conclusion": (
            "Verified external values are listed separately, but inherited active blocks "
            "have recorded competition-game proxy/structural selection or incomplete origin. "
            "External-only reselection or authoritative clarification is not recorded. "
            "This is unresolved provenance, not a finding that protected possession truth was used."
        ),
        "limitations": [
            "Technical lineage audit, not organizer eligibility approval or legal advice.",
            "Inactive video/HSMM settings do not certify or disqualify the active runtime.",
            "Remaining configured keys and hardcoded assumptions are not exhaustively audited.",
            "Retaining the baseline does not certify its shared inherited provenance.",
            "No parameters are changed or reselected during this audit.",
        ],
    }


def _exit_candidate_changes(baseline, candidate, overrides):
    changes = {
        key for key in set(baseline) | set(candidate)
        if baseline.get(key) != candidate.get(key)
    }
    expected = set(overrides) | {"exit_bracket_release_provenance"}
    if changes != expected or any(candidate.get(key) != value for key, value in overrides.items()):
        raise ValueError("exit bracket candidate changes more than its frozen evidence and metadata")
    return sorted(changes)


def audit_exit_bracket_release(project, protocol, external_root, independent_root):
    project = Path(project)
    external_root = Path(external_root)
    if independent_root is None:
        raise ValueError("exit-bracket audit requires the independent DFL checkout")
    independent_root = Path(independent_root)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("candidate_scope") != "exit_bracket_release"
        or protocol.get("require_fresh_independent_external_validation") is not True
        or protocol.get("require_active_parameter_provenance") is not True
    ):
        raise ValueError("unknown exit bracket release protocol")
    base_path = project / protocol["baseline_config"]
    candidate_path = project / protocol["candidate_config"]
    benchmark_path = project / protocol["external_benchmark"]
    for path, digest in (
        (base_path, protocol["baseline_sha256"]),
        (candidate_path, protocol["candidate_sha256"]),
        (benchmark_path, protocol["external_benchmark_sha256"]),
    ):
        if _file_sha256(path) != digest:
            raise ValueError("{}: frozen release input changed".format(path))
    if subprocess.check_output(
        ["git", "-C", str(external_root), "rev-parse", "HEAD"], text=True
    ).strip() != protocol["external_revision"]:
        raise ValueError("Metrica sample revision changed during exit release audit")
    baseline = load_config(base_path)
    candidate = load_config(candidate_path)
    benchmark = json.loads(benchmark_path.read_text())
    overrides = benchmark["frozen_values"]
    changes = _exit_candidate_changes(baseline, candidate, overrides)
    provenance = candidate["exit_bracket_release_provenance"]
    if (
        provenance["status"] != "candidate_not_selected"
        or provenance["baseline_sha256"] != protocol["baseline_sha256"]
        or provenance["source_fps"] != protocol["source_fps"]
        or protocol["source_fps"] != 25
        or provenance["only_active_overrides"] != overrides
        or provenance["inherited_active_parameter_provenance_cleared"] is not False
        or provenance["independent_holdout_verified"] is not False
        or candidate["video_ball_observations_path"] is not None
    ):
        raise ValueError("candidate provenance or tracking-only policy is inconsistent")
    expected_roles = {
        "Sample_Game_1": "development",
        "Sample_Game_2": "previously_inspected_diagnostic",
        "Sample_Game_3": "previously_inspected_diagnostic",
    }
    if set(protocol["frozen_reports"]) != set(expected_roles):
        raise ValueError("exit release protocol has unexpected game roles")
    external = {}
    source_code = None
    for game, role in expected_roles.items():
        declaration = protocol["frozen_reports"][game]
        path = project / declaration["path"]
        if declaration["role"] != role or _file_sha256(path) != declaration["sha256"]:
            raise ValueError("{}: frozen external report changed".format(game))
        report = json.loads(path.read_text())
        if (
            report["game"] != game or report["role"] != role
            or report["source_fps"] != protocol["source_fps"]
            or report["values"] != overrides
            or report["base_config_sha256"] != protocol["baseline_sha256"]
            or report["external_revision"] != protocol["external_revision"]
            or not report["safe"] or len(report["checks"]) != 60
            or not all(item["passed"] for item in report["checks"])
        ):
            raise ValueError("{}: frozen candidate evaluation or guard set changed".format(game))
        if any(
            _file_sha256(external_root / "data" / game / name) != digest
            for name, digest in report["input_sha256"].items()
        ):
            raise ValueError("{}: external input differs from its frozen report".format(game))
        if source_code is not None and report["code_sha256"] != source_code:
            raise ValueError("frozen games used different model code")
        source_code = report["code_sha256"]
        for name, digest in source_code.items():
            if _file_sha256(project / "possession" / name) != digest:
                raise ValueError("{}: frozen model code has changed".format(name))
        external[game] = {
            "role": role, "guards_passed": len(report["checks"]),
            "baseline_objective": report["results"]["deployment"]["objective"],
            "candidate_objective": report["results"]["exit_bracket"]["objective"],
            "report_sha256": declaration["sha256"],
        }
    if (
        provenance["development_report_sha256"] != protocol["frozen_reports"]["Sample_Game_1"]["sha256"]
        or provenance["inspected_diagnostic_report_sha256"] != {
            game: protocol["frozen_reports"][game]["sha256"]
            for game in ("Sample_Game_2", "Sample_Game_3")
        }
        or provenance["source_code_sha256"] != source_code
        or provenance["external_revision"] != protocol["external_revision"]
        or benchmark["code_sha256"] != source_code
    ):
        raise ValueError("release metadata differs from frozen development and transfer evidence")
    independent = protocol["independent_external"]
    if (
        independent["reselection_on_validation_match"] is not False
        or independent["synthetic_or_real_tagging_used"] is not False
        or independent["published_ball_status_codes"] != {"0": "Dead", "1": "Alive"}
        or independent["published_ball_possession_codes"] != {"1": "Home", "2": "Away"}
    ):
        raise ValueError("DFL validation must be frozen and no-tag")
    dfl = {}
    dfl_code = None
    for role in ("development", "unseen_frozen_diagnostic"):
        declared = independent[role]
        path = project / declared["path"]
        if _file_sha256(path) != declared["sha256"]:
            raise ValueError("{}: independent external report changed".format(role))
        result = json.loads(path.read_text())
        match = declared["match"]
        metadata_paths = list(independent_root.glob(
            "DFL_02_01_matchinformation_*{}*.xml".format(match)
        ))
        positions_paths = list(independent_root.glob(
            "DFL_04_03_positions_raw_observed_*{}*.xml".format(match)
        ))
        if len(metadata_paths) != 1 or len(positions_paths) != 1:
            raise ValueError("{}: DFL match files are missing or ambiguous".format(match))
        general = ET.parse(metadata_paths[0]).find(".//General")
        if (
            general is None or general.get("MatchId") != "DFL-MAT-" + match
            or _file_sha256(metadata_paths[0]) != result["metadata_sha256"]
            or _file_sha256(positions_paths[0]) != result["positions_sha256"]
        ):
            raise ValueError("{}: independent DFL match identity or input differs".format(match))
        if (
            result["baseline_config_sha256"] != protocol["baseline_sha256"]
            or result["candidate_config_sha256"] != protocol["candidate_sha256"]
            or result["selected_submission_changed"] is not False
        ):
            raise ValueError("{}: DFL validation used different model inputs".format(role))
        if dfl_code is not None and dfl_code != result["code_sha256"]:
            raise ValueError("DFL parser changed between development and validation")
        dfl_code = result["code_sha256"]
        if _file_sha256(project / "possession" / "external_dfl_validation.py") != dfl_code:
            raise ValueError("DFL validation parser changed after frozen transfer")
        dfl[role] = {
            "match": declared["match"], "report_sha256": declared["sha256"],
            "dead_recall": {
                name: result["models"][name]["dead"]["dead_recall_on_known_frames"]
                for name in ("baseline", "exit_bracket")
            },
            "spell_f1_0.75": {
                name: result["models"][name]["live_spell_f1"]["0.75"]["f1"]
                for name in ("baseline", "exit_bracket")
            },
        }
    inherited = [
        {
            "block": name, "values": {key: baseline[key] for key in keys},
            "status": "external_only_reselection_not_verified",
            "historical_evidence": evidence,
        } for name, keys, evidence in _provenance_groups(project)
    ]
    kinematics_verified, kinematic_keys = _verified_kinematics(project, candidate)
    verified_keys = set(overrides) | (set(kinematic_keys) if kinematics_verified else set())
    traced_keys = {key for _, keys, _ in _provenance_groups(project) for key in keys}
    remaining = sorted(
        set(baseline) - verified_keys - traced_keys - _inactive_parameters(candidate)
        - {"compliance_provenance", "exit_bracket_release_provenance"}
    )
    return {
        "schema_version": 1, "candidate_sha256": protocol["candidate_sha256"],
        "baseline_sha256": protocol["baseline_sha256"], "allowed_changes": changes,
        "new_exit_overrides_match_frozen_external_development": True,
        "kinematic_overrides_externally_verified": kinematics_verified,
        "verified_external_values": {key: candidate[key] for key in sorted(verified_keys)},
        "frozen_external": external, "independent_dfl_no_tag": dfl,
        "unresolved_active_blocks": inherited,
        "remaining_inherited_keys_not_fully_traced": remaining,
        "tag_code_team_semantics_externally_verified": False,
        "fresh_holdout_evaluated_without_reselection": True,
        "active_tuned_parameter_provenance_cleared": False,
        "decision": "retain_current_submission",
        "blockers": [
            "inherited_active_parameter_provenance",
            "remaining_inherited_parameter_origins",
            "tagged_input_transfer_unverified",
        ],
        "challenge_accuracy_claim": False, "selected_submission_modified": False,
    }


def release_decision(external, challenge, provenance, reproduced):
    gates = {
        "external_diagnostics": bool(external["complete"] and external["passed"]),
        "automatic_challenge_robustness": bool(challenge["passed"]),
        "byte_identical_reproduction": bool(reproduced),
        "active_parameter_provenance": bool(provenance["active_tuned_parameter_provenance_cleared"]),
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "decision": "promote" if not blockers else "retain_current_submission",
        "gates": gates, "blockers": blockers,
        "parameter_retuning_performed": False,
        "additional_candidate_search_performed": False,
    }


def _save(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit and finalize one frozen gap release pass.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--independent-root", help="public DFL data root for exit-bracket provenance")
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--external-report")
    parser.add_argument("--challenge-report")
    parser.add_argument("--candidate-csv")
    parser.add_argument("--repeat-csv")
    parser.add_argument("--decision-output")
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    try:
        protocol_path = Path(args.protocol)
        protocol = json.loads(protocol_path.read_text())
        exit_scope = protocol.get("candidate_scope") == "exit_bracket_release"
        provenance = (
            audit_exit_bracket_release(
                project, protocol, args.external_root, args.independent_root,
            )
            if exit_scope else audit_active_parameters(project, protocol, args.external_root)
        )
        _save(args.audit_output, provenance)
        if args.audit_only:
            print("Active provenance cleared: {}".format(
                provenance["active_tuned_parameter_provenance_cleared"]
            ))
            return 0
        if exit_scope:
            parser.error("exit-bracket audit supports --audit-only until tagged transfer and inherited provenance are cleared")
        required = (args.external_report, args.challenge_report, args.candidate_csv,
                    args.repeat_csv, args.decision_output)
        if not all(required):
            parser.error("final decision requires reports, candidate/repeat CSVs and output")
        external = json.loads(Path(args.external_report).read_text())
        challenge = json.loads(Path(args.challenge_report).read_text())
        for path in (args.candidate_csv, args.repeat_csv):
            validate_submission_file(path, 199319)
        reproduced = _file_sha256(args.candidate_csv) == _file_sha256(args.repeat_csv)
        if external["candidate_sha256"] != protocol["candidate_sha256"]:
            raise ValueError("external report evaluated a different candidate")
        if external["baseline_sha256"] != protocol["baseline_sha256"]:
            raise ValueError("external report evaluated a different baseline")
        if external["protocol_sha256"] != _file_sha256(protocol_path):
            raise ValueError("external report used a different release protocol")
        reported_config = Path(challenge["config"])
        if not reported_config.is_absolute():
            reported_config = project / reported_config
        if reported_config.resolve() != (project / protocol["candidate_config"]).resolve():
            raise ValueError("challenge report evaluated a different config")
        if challenge["config_sha256"] != protocol["candidate_sha256"]:
            raise ValueError("challenge report evaluated different candidate parameters")
        report = release_decision(external, challenge, provenance, reproduced)
        current_csv = project / "submissions" / "possession_compliant.csv"
        validate_submission_file(current_csv, 199319)
        current_hash = _file_sha256(current_csv)
        candidate_hash = _file_sha256(args.candidate_csv)
        report.update({
            "schema_version": 1, "protocol_sha256": _file_sha256(protocol_path),
            "baseline_sha256": protocol["baseline_sha256"],
            "candidate_sha256": protocol["candidate_sha256"],
            "candidate_csv_sha256": candidate_hash,
            "current_csv_sha256": current_hash,
            "candidate_equals_current_submission": candidate_hash == current_hash,
            "performance_interpretation": (
                "Byte-identical challenge CSV: no possible evaluation-score change for this match."
                if candidate_hash == current_hash else
                "Different CSV; challenge accuracy is not inferred without ground truth."
            ),
            "external_report_sha256": _file_sha256(args.external_report),
            "challenge_report_sha256": _file_sha256(args.challenge_report),
            "provenance_report_sha256": _file_sha256(args.audit_output),
            "external_failed_checks": [check["name"] for check in external["checks"] if not check["passed"]],
            "challenge_failed_checks": [check["name"] for check in challenge["checks"] if not check["passed"]],
            "external_spell_summary": external.get("gap_release_spell_summary"),
            "scope": "One frozen candidate; inspected external games are diagnostic only.",
            "release_check_code_sha256": _file_sha256(Path(__file__)),
            "default_submission_modified": False,
        })
        _save(args.decision_output, report)
        print("{}; blockers={}".format(report["decision"], report["blockers"]))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        parser.exit(1, "error: {}\n".format(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
