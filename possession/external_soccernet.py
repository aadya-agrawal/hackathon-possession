import argparse
import json
import sys
from pathlib import Path


SOURCE_SPLITS = ("train", "valid", "test")


def _version_tuple(value):
    try:
        parts = tuple(int(part) for part in str(value).split("."))
    except ValueError as error:
        raise ValueError("invalid SoccerNet dataset version {!r}".format(
            value
        )) from error
    if not parts:
        raise ValueError("missing SoccerNet dataset version")
    return parts


def load_soccernet_config(path):
    with Path(path).open() as handle:
        config = json.load(handle)
    split_mapping = config.get("split_mapping")
    if not isinstance(split_mapping, dict) or set(split_mapping) != {
        "development", "validation", "test"
    }:
        raise ValueError(
            "{}: split_mapping must define development, validation, and test"
            .format(path)
        )
    assigned = {}
    for target_split, source_splits in split_mapping.items():
        if not isinstance(source_splits, list) or not source_splits:
            raise ValueError(
                "{}: {} source splits must be a non-empty list".format(
                    path, target_split
                )
            )
        for source_split in source_splits:
            if source_split not in SOURCE_SPLITS:
                raise ValueError(
                    "{}: unsupported source split {}".format(
                        path, source_split
                    )
                )
            if source_split in assigned:
                raise ValueError(
                    "{}: source split {} appears in {} and {}".format(
                        path, source_split, assigned[source_split],
                        target_split
                    )
                )
            assigned[source_split] = target_split
    detector = config.get("detector_evaluation")
    if not isinstance(detector, dict):
        raise ValueError("{}: detector_evaluation is required".format(path))
    eligible = detector.get("eligible_source_splits")
    if not isinstance(eligible, list):
        raise ValueError(
            "{}: eligible_source_splits must be a list".format(path)
        )
    if detector.get("overlap_status") != "resolved" and eligible:
        raise ValueError(
            "{}: detector evaluation cannot be enabled before training "
            "overlap is resolved".format(path)
        )
    unknown = sorted(set(eligible) - set(assigned))
    if unknown:
        raise ValueError(
            "{}: detector evaluation uses unassigned splits {}".format(
                path, unknown
            )
        )
    deferred = config.get("deferred_target_splits", [])
    if (
        not isinstance(deferred, list)
        or not set(deferred).issubset({"development", "validation", "test"})
    ):
        raise ValueError(
            "{}: deferred_target_splits contains invalid values".format(path)
        )
    return config, assigned


def _load_label_file(path, minimum_version):
    with path.open() as handle:
        data = json.load(handle)
    for key in ("info", "images", "annotations"):
        if key not in data:
            raise ValueError("{}: missing {}".format(path, key))
    version = data["info"].get("version")
    if _version_tuple(version) < _version_tuple(minimum_version):
        raise ValueError(
            "{}: dataset version {} is older than required {}".format(
                path, version, minimum_version
            )
        )
    if not isinstance(data["images"], list):
        raise ValueError("{}: images must be a list".format(path))
    if not isinstance(data["annotations"], list):
        raise ValueError("{}: annotations must be a list".format(path))
    return data


def _image_frame_lookup(path, images):
    lookup = {}
    for frame, image in enumerate(images):
        image_id = image.get("image_id", image.get("id"))
        if image_id is None:
            raise ValueError(
                "{}: image {} has no image_id".format(path, frame)
            )
        key = str(image_id)
        if key in lookup:
            raise ValueError(
                "{}: duplicate image_id {}".format(path, image_id)
            )
        lookup[key] = frame
    return lookup


def _bbox_center(path, annotation):
    bbox = annotation.get("bbox_image")
    if not isinstance(bbox, dict):
        raise ValueError("{}: ball annotation has no bbox_image".format(path))
    values = []
    for key in ("x_center", "y_center", "w", "h"):
        try:
            value = float(bbox[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "{}: invalid ball bbox field {}".format(path, key)
            ) from error
        if value < 0:
            raise ValueError(
                "{}: negative ball bbox field {}".format(path, key)
            )
        values.append(value)
    return values


def load_gamestate_ball_annotations(path, minimum_version="1.3"):
    path = Path(path)
    data = _load_label_file(path, minimum_version)
    frame_by_image = _image_frame_lookup(path, data["images"])
    labels = []
    for annotation in data["annotations"]:
        if annotation.get("supercategory") != "object":
            continue
        attributes = annotation.get("attributes") or {}
        if attributes.get("role") != "ball":
            continue
        image_id = annotation.get("image_id")
        key = str(image_id)
        if key not in frame_by_image:
            raise ValueError(
                "{}: annotation references unknown image_id {}".format(
                    path, image_id
                )
            )
        x, y, width, height = _bbox_center(path, annotation)
        labels.append({
            "frame": frame_by_image[key],
            "visible": True,
            "pixel_x": x,
            "pixel_y": y,
            "bbox_width": width,
            "bbox_height": height,
            "track_id": annotation.get("track_id"),
            "source": "soccernet_gamestate_human_ground_truth",
        })
    labels.sort(key=lambda item: (item["frame"], str(item["track_id"])))
    frame_counts = {}
    for label in labels:
        frame_counts[label["frame"]] = frame_counts.get(label["frame"], 0) + 1
    for label in labels:
        label["ambiguous"] = frame_counts[label["frame"]] > 1
    return labels


def inspect_gamestate_clip(path, source_split, target_split, minimum_version):
    path = Path(path)
    data = _load_label_file(path, minimum_version)
    info = data["info"]
    ball_labels = load_gamestate_ball_annotations(path, minimum_version)
    ambiguous_ball_frames = {
        label["frame"] for label in ball_labels if label["ambiguous"]
    }
    roles = {}
    teams = set()
    pitch_boxes = 0
    camera_annotations = 0
    pitch_annotations = 0
    for annotation in data["annotations"]:
        supercategory = annotation.get("supercategory")
        if supercategory == "object":
            attributes = annotation.get("attributes") or {}
            role = attributes.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1
            team = attributes.get("team")
            if team is not None:
                teams.add(str(team))
            if annotation.get("bbox_pitch") is not None:
                pitch_boxes += 1
        elif supercategory == "camera":
            camera_annotations += 1
        elif supercategory == "pitch":
            pitch_annotations += 1

    sequence_id = path.parent.name
    return {
        "sequence_id": sequence_id,
        "source_split": source_split,
        "split": target_split,
        "frames": len(data["images"]),
        "fps": info.get("frame_rate"),
        "width": data["images"][0].get("width") if data["images"] else None,
        "height": data["images"][0].get("height") if data["images"] else None,
        "dataset_version": str(info.get("version")),
        "label_path": str(path.resolve()),
        "ground_truth": {
            "source": "SoccerNet Game State Reconstruction",
            "human_annotated": True,
            "ball_role_selector": {
                "supercategory": "object",
                "attributes.role": "ball",
            },
            "ball_annotations": len(ball_labels),
            "ball_labeled_frames": len({
                label["frame"] for label in ball_labels
            }),
            "ambiguous_ball_frames": len(ambiguous_ball_frames),
            "usable_ball_annotations": sum(
                not label["ambiguous"] for label in ball_labels
            ),
            "role_annotations": dict(sorted(roles.items())),
            "teams": sorted(teams),
            "bbox_pitch_annotations": pitch_boxes,
            "camera_annotations": camera_annotations,
            "pitch_annotations": pitch_annotations,
        },
    }


def build_soccernet_gamestate_manifest(root, config_path):
    root = Path(root).resolve()
    config, target_by_source = load_soccernet_config(config_path)
    minimum_version = str(config.get("minimum_dataset_version", "1.3"))
    deferred = set(config.get("deferred_target_splits", []))
    clips = []
    for source_split, target_split in sorted(target_by_source.items()):
        split_root = root / source_split
        if not split_root.is_dir():
            if target_split in deferred:
                continue
            raise ValueError(
                "{}: missing source split directory {}".format(
                    root, source_split
                )
            )
        paths = sorted(split_root.glob("*/Labels-GameState.json"))
        if not paths:
            if target_split in deferred:
                continue
            raise ValueError(
                "{}: no Labels-GameState.json files in {}".format(
                    root, source_split
                )
            )
        clips.extend(
            inspect_gamestate_clip(
                path, source_split, target_split, minimum_version
            )
            for path in paths
        )

    summary_splits = {}
    for target_split in ("development", "validation", "test"):
        selected = [
            clip for clip in clips if clip["split"] == target_split
        ]
        summary_splits[target_split] = {
            "source_splits": sorted({
                clip["source_split"] for clip in selected
            }),
            "available": bool(selected),
            "deferred": target_split in deferred,
            "clips": len(selected),
            "frames": sum(clip["frames"] for clip in selected),
            "ball_annotations": sum(
                clip["ground_truth"]["ball_annotations"]
                for clip in selected
            ),
        }

    detector = config["detector_evaluation"]
    return {
        "schema_version": 1,
        "dataset": "SoccerNet Game State Reconstruction",
        "external_data": True,
        "root": str(root),
        "minimum_dataset_version": minimum_version,
        "split_policy": {
            "unit": "official SoccerNet split",
            "mapping": config["split_mapping"],
            "test_usage": "untouched until all parameters are frozen",
            "deferred_target_splits": sorted(deferred),
            "challenge_split_used": False,
        },
        "ground_truth_policy": {
            "ball_annotations": (
                "Human ground-truth boxes selected only where "
                "attributes.role is ball."
            ),
            "missing_ball_annotation": (
                "Unknown or not labeled; never interpreted as dead ball."
            ),
            "multiple_ball_annotations": (
                "Ambiguous frame; every candidate is retained and excluded "
                "from calibration metrics."
            ),
            "possession_labels_available": False,
        },
        "detector_evaluation": {
            "checkpoint": detector.get("checkpoint"),
            "training_dataset": detector.get("training_dataset"),
            "overlap_status": detector.get("overlap_status"),
            "eligible_source_splits": detector[
                "eligible_source_splits"
            ],
            "enabled": bool(detector["eligible_source_splits"]),
            "policy": (
                "Detector metrics remain disabled until checkpoint training "
                "clip overlap is resolved."
            ),
        },
        "summary": {
            "clips": len(clips),
            "frames": sum(clip["frames"] for clip in clips),
            "ball_annotations": sum(
                clip["ground_truth"]["ball_annotations"] for clip in clips
            ),
            "splits": summary_splits,
        },
        "clips": clips,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build a split-safe SoccerNet Game State calibration manifest."
        )
    )
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "configs" / "soccernet_calibration.json"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_soccernet_gamestate_manifest(
            args.root, args.config
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (IOError, OSError, ValueError) as error:
        parser.exit(1, "error: {}\n".format(error))
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
