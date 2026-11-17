import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from possession.external_soccernet import load_gamestate_ball_annotations
from possession.video import detect_tiled_ball_pixels


PITCH_X_BOUNDS = (-52.5, 52.5)
PITCH_Y_BOUNDS = (-34.0, 34.0)


def _distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _raw_pitch_point(annotation):
    bbox = annotation.get("bbox_pitch")
    if not isinstance(bbox, dict):
        return None
    try:
        return (
            float(bbox["x_bottom_middle"]),
            float(bbox["y_bottom_middle"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _pitch_point(annotation):
    point = _raw_pitch_point(annotation)
    if point is None:
        return None
    if not (
        PITCH_X_BOUNDS[0] <= point[0] <= PITCH_X_BOUNDS[1]
        and PITCH_Y_BOUNDS[0] <= point[1] <= PITCH_Y_BOUNDS[1]
    ):
        return None
    return point


def _missing_runs(frame_count, observed_frames):
    observed = set(observed_frames)
    runs = []
    start = None
    for frame in range(frame_count):
        if frame not in observed and start is None:
            start = frame
        elif frame in observed and start is not None:
            runs.append(frame - start)
            start = None
    if start is not None:
        runs.append(frame_count - start)
    return runs


def _clip_metrics(path):
    with Path(path).open() as handle:
        data = json.load(handle)
    images = data["images"]
    frame_by_image = {
        str(image.get("image_id", image.get("id"))): frame
        for frame, image in enumerate(images)
    }
    balls = {}
    athletes = {}
    for annotation in data["annotations"]:
        if annotation.get("supercategory") != "object":
            continue
        image_id = str(annotation.get("image_id"))
        if image_id not in frame_by_image:
            continue
        frame = frame_by_image[image_id]
        attributes = annotation.get("attributes") or {}
        role = attributes.get("role")
        if role == "ball":
            balls.setdefault(frame, []).append(annotation)
        elif role in {"player", "goalkeeper"}:
            point = _pitch_point(annotation)
            if point is not None:
                athletes.setdefault(frame, []).append(point)

    unambiguous = {
        frame: annotations[0]
        for frame, annotations in balls.items()
        if len(annotations) == 1
    }
    ambiguous_frames = {
        frame for frame, annotations in balls.items()
        if len(annotations) > 1
    }
    diameters = []
    nearest_player_distances = []
    pitch_positions = {}
    rejected_ball_pitch_projections = 0
    for frame, annotation in unambiguous.items():
        bbox = annotation.get("bbox_image") or {}
        try:
            width = float(bbox["w"])
            height = float(bbox["h"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            if width >= 0 and height >= 0:
                diameters.append(float(np.sqrt(width * height)))
        raw_point = _raw_pitch_point(annotation)
        point = _pitch_point(annotation)
        if point is None:
            if raw_point is not None:
                rejected_ball_pitch_projections += 1
            continue
        pitch_positions[frame] = (
            point,
            annotation.get("track_id"),
        )
        player_points = athletes.get(frame, [])
        if player_points:
            nearest_player_distances.append(min(
                float(np.hypot(
                    point[0] - player[0],
                    point[1] - player[1],
                ))
                for player in player_points
            ))

    fps = float(data["info"]["frame_rate"])
    velocities = {}
    speeds = []
    for frame in sorted(pitch_positions):
        previous = pitch_positions.get(frame - 1)
        current = pitch_positions[frame]
        if previous is None or previous[1] != current[1]:
            continue
        velocity = (
            (current[0][0] - previous[0][0]) * fps,
            (current[0][1] - previous[0][1]) * fps,
        )
        velocities[frame] = velocity
        speeds.append(float(np.hypot(*velocity)))

    accelerations = []
    for frame, velocity in velocities.items():
        previous = velocities.get(frame - 1)
        if previous is None:
            continue
        accelerations.append(float(np.hypot(
            velocity[0] - previous[0],
            velocity[1] - previous[1],
        ) * fps))

    missing_runs = _missing_runs(
        len(images), set(unambiguous) | ambiguous_frames
    )
    return {
        "frames": len(images),
        "unambiguous_ball_frames": len(unambiguous),
        "ambiguous_ball_frames": len(ambiguous_frames),
        "missing_ball_frames": (
            len(images) - len(unambiguous) - len(ambiguous_frames)
        ),
        "rejected_ball_pitch_projections": (
            rejected_ball_pitch_projections
        ),
        "ball_diameters_px": diameters,
        "nearest_player_distances_m": nearest_player_distances,
        "ball_speeds_mps": speeds,
        "ball_accelerations_mps2": accelerations,
        "missing_runs_frames": missing_runs,
        "missing_runs_seconds": [
            length / fps for length in missing_runs
        ],
    }


def build_soccernet_geometry_metrics(manifest_path):
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    split_results = {}
    for split in ("development", "validation"):
        clips = [
            clip for clip in manifest["clips"]
            if clip["split"] == split
        ]
        totals = {
            "frames": 0,
            "unambiguous_ball_frames": 0,
            "ambiguous_ball_frames": 0,
            "missing_ball_frames": 0,
            "rejected_ball_pitch_projections": 0,
        }
        values = {
            "ball_diameters_px": [],
            "nearest_player_distances_m": [],
            "ball_speeds_mps": [],
            "ball_accelerations_mps2": [],
            "missing_runs_frames": [],
            "missing_runs_seconds": [],
        }
        for clip in clips:
            metrics = _clip_metrics(clip["label_path"])
            for key in totals:
                totals[key] += metrics[key]
            for key in values:
                values[key].extend(metrics[key])
        split_results[split] = {
            "clips": len(clips),
            **totals,
            "coverage": (
                totals["unambiguous_ball_frames"] / totals["frames"]
                if totals["frames"] else 0.0
            ),
            "distributions": {
                key: _distribution(items)
                for key, items in values.items()
            },
        }

    return {
        "schema_version": 1,
        "dataset": "SoccerNet Game State Reconstruction",
        "external_data": True,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "policy": {
            "test_split_used": False,
            "ambiguous_ball_frames_used": False,
            "detector_outputs_used": False,
            "roles_used_for_nearest_player": ["player", "goalkeeper"],
            "pitch_x_bounds_m": list(PITCH_X_BOUNDS),
            "pitch_y_bounds_m": list(PITCH_Y_BOUNDS),
            "out_of_pitch_projections_used": False,
        },
        "splits": split_results,
    }


def score_annotated_ball_candidates(reference_xy, candidates, position_key="pixel"):
    if not candidates:
        return {
            "candidate_count": 0,
            "nearest_error_pixels": None,
            "highest_confidence_error_pixels": None,
        }
    distances = [
        float(np.linalg.norm(candidate[position_key] - reference_xy))
        for candidate in candidates
    ]
    highest_confidence = max(
        range(len(candidates)),
        key=lambda index: candidates[index]["confidence"],
    )
    return {
        "candidate_count": len(candidates),
        "nearest_error_pixels": min(distances),
        "highest_confidence_error_pixels": distances[highest_confidence],
    }


def _player_ground_homography(annotations, cv2):
    pixels = []
    ground = []
    for annotation in annotations:
        if annotation.get("supercategory") != "object":
            continue
        if (annotation.get("attributes") or {}).get("role") not in {
            "player", "goalkeeper", "referee",
        }:
            continue
        box = annotation.get("bbox_image") or {}
        position = _pitch_point(annotation)
        if position is None:
            continue
        try:
            foot = (float(box["x_center"]), float(box["y"]) + float(box["h"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(foot).all():
            continue
        pixels.append(foot)
        ground.append(position)
    if len(pixels) < 6:
        return None
    src = np.asarray(pixels, dtype=np.float32)
    dst = np.asarray(ground, dtype=np.float32)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 1.5)
    if matrix is None or mask is None or np.count_nonzero(mask) < 6:
        return None
    predicted = cv2.perspectiveTransform(src[:, None, :], matrix)[:, 0, :]
    errors = np.linalg.norm(predicted - dst, axis=1)
    if np.percentile(errors[mask[:, 0] != 0], 90) > 1.0:
        return None
    return matrix


def _project_ground_candidates(pixels, matrix, cv2):
    candidates = []
    for candidate in pixels:
        pixel = np.asarray(candidate["pixel"], dtype=np.float32)
        ground = cv2.perspectiveTransform(
            pixel.reshape(1, 1, 2), matrix
        )[0, 0].astype(np.float64)
        if (
            np.isfinite(ground).all()
            and -54.6 <= ground[0] <= 54.6
            and -35.36 <= ground[1] <= 35.36
        ):
            candidates.append({
                "ground_metres": ground,
                "normalized_pitch": (
                    (ground + np.asarray([52.5, 34.0]))
                    / np.asarray([105.0, 68.0])
                ),
                "confidence": candidate["confidence"],
            })
    return candidates


def _generic_coco_model(model_path, YOLO, settings):
    model_path = Path(model_path)
    settings.update({"sync": False})
    model = YOLO(str(model_path))
    training = (model.ckpt or {}).get("train_args", {}).get("data")
    if Path(str(training)).name != "coco.yaml":
        raise ValueError(
            "{}: checkpoint is not documented as generic COCO training; "
            "SoccerNet checkpoint overlap with these clips is unresolved"
            .format(model_path)
        )
    if model.names.get(32) != "sports ball":
        raise ValueError("{}: COCO sports-ball class 32 is missing".format(model_path))
    return model, training


def _sample_soccernet_clips(manifest, split, maximum_clips):
    clips = sorted(
        (clip for clip in manifest["clips"] if clip["split"] == split),
        key=lambda clip: clip["sequence_id"],
    )
    if not clips:
        raise ValueError("no clips in {} split".format(split))
    if len(clips) > maximum_clips:
        indices = np.linspace(0, len(clips) - 1, maximum_clips, dtype=int)
        clips = [clips[index] for index in indices]
    return clips


def _anchor_perturbations(clip_id, radius, trials=4):
    seed = int.from_bytes(
        hashlib.sha256(clip_id.encode("utf-8")).digest()[:8], "big"
    )
    angles = np.random.default_rng(seed).uniform(0, 2 * np.pi, size=2)
    for trial in range(trials):
        rotated = angles + trial * np.pi / 2
        yield radius * np.column_stack((np.cos(rotated), np.sin(rotated)))


def benchmark_soccernet_detector(
    manifest_path, model_path, *,
    split, frame_stride, maximum_clips, confidence, image_size, device,
):
    if split not in ("development", "validation"):
        raise ValueError("detector benchmark is restricted to train/valid")
    if frame_stride <= 0 or maximum_clips <= 0 or not 0 < confidence <= 1:
        raise ValueError("frame stride, maximum clips and confidence must be positive")
    if image_size <= 0:
        raise ValueError("image size must be positive")

    try:
        import cv2
        from ultralytics import YOLO, settings
    except ImportError as error:
        raise RuntimeError(
            "detector benchmark requires OpenCV and Ultralytics"
        ) from error

    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    clips = _sample_soccernet_clips(manifest, split, maximum_clips)

    model_path = Path(model_path)
    model, training = _generic_coco_model(model_path, YOLO, settings)

    counts = {
        "annotated_frames": 0,
        "ambiguous_frames_excluded": 0,
        "unannotated_frames_excluded": 0,
        "frames_with_candidates": 0,
        "total_candidates": 0,
    }
    nearest_errors = []
    highest_errors = []
    ground_frames = 0
    ground_nearest_errors = []
    ground_highest_errors = []
    for clip in clips:
        label_path = Path(clip["label_path"])
        with label_path.open() as handle:
            data = json.load(handle)
        image_dir = label_path.parent / data["info"]["im_dir"]
        labels_by_frame = {}
        for label in load_gamestate_ball_annotations(label_path):
            labels_by_frame.setdefault(label["frame"], []).append(label)
        objects_by_image = {}
        for annotation in data["annotations"]:
            if annotation.get("supercategory") == "object":
                objects_by_image.setdefault(
                    str(annotation["image_id"]), []
                ).append(annotation)
        for frame in range(0, len(data["images"]), frame_stride):
            annotations = labels_by_frame.get(frame, [])
            if len(annotations) > 1:
                counts["ambiguous_frames_excluded"] += 1
                continue
            if not annotations:
                counts["unannotated_frames_excluded"] += 1
                continue
            image_path = image_dir / data["images"][frame]["file_name"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("{}: cannot read annotated frame".format(image_path))
            candidates = detect_tiled_ball_pixels(
                image, model, class_id=32, confidence=confidence,
                image_size=image_size, device=device, overlap_fraction=0.1,
                merge_radius_pixels=20,
            )
            reference = np.asarray([
                annotations[0]["pixel_x"], annotations[0]["pixel_y"],
            ])
            score = score_annotated_ball_candidates(reference, candidates)
            counts["annotated_frames"] += 1
            counts["total_candidates"] += score["candidate_count"]
            if candidates:
                counts["frames_with_candidates"] += 1
                nearest_errors.append(score["nearest_error_pixels"])
                highest_errors.append(score["highest_confidence_error_pixels"])

            objects = objects_by_image.get(
                str(data["images"][frame]["image_id"]), []
            )
            ball_objects = [
                item for item in objects
                if (item.get("attributes") or {}).get("role") == "ball"
            ]
            ground_ball = _pitch_point(ball_objects[0]) if len(ball_objects) == 1 else None
            ground_map = _player_ground_homography(objects, cv2)
            if ground_ball is None or ground_map is None:
                continue
            ground_frames += 1
            ground_candidates = _project_ground_candidates(
                candidates, ground_map, cv2
            )
            ground_score = score_annotated_ball_candidates(
                np.asarray(ground_ball), ground_candidates,
                position_key="ground_metres",
            )
            if ground_candidates:
                ground_nearest_errors.append(
                    ground_score["nearest_error_pixels"]
                )
                ground_highest_errors.append(
                    ground_score["highest_confidence_error_pixels"]
                )

    if not counts["annotated_frames"]:
        raise ValueError("{}: no unambiguous annotated frames sampled".format(manifest_path))
    total = counts["annotated_frames"]
    return {
        "schema_version": 1,
        "dataset": "SoccerNet Game State Reconstruction",
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_training_data": training,
        "split": split,
        "sampled_clips": [clip["sequence_id"] for clip in clips],
        "frame_stride": frame_stride,
        "confidence": confidence,
        "image_size": image_size,
        "device": device,
        "policy": {
            "training_overlap": "generic COCO; no clip-level overlap guarantee",
            "SoccerNet_specific_checkpoint_evaluated": False,
            "unannotated_frames_are_not_negatives": True,
            "ambiguous_frames_excluded": True,
            "test_split_used": False,
            "possession_truth_available": False,
            "challenge_data_used": False,
        },
        "metrics": {
            **counts,
            "mean_candidates_per_annotated_frame": (
                counts["total_candidates"] / total
            ),
            "frames_with_candidates_fraction": (
                counts["frames_with_candidates"] / total
            ),
            "nearest_candidate_recall": {
                str(radius): sum(error <= radius for error in nearest_errors) / total
                for radius in (20, 40, 80)
            },
            "highest_confidence_recall": {
                str(radius): sum(error <= radius for error in highest_errors) / total
                for radius in (20, 40, 80)
            },
            "highest_confidence_error_pixels_if_detected": (
                _distribution(highest_errors)
            ),
            "annotated_player_homography_frames": ground_frames,
            "ground_nearest_recall_within_1_5m": (
                sum(error <= 1.5 for error in ground_nearest_errors) / ground_frames
                if ground_frames else None
            ),
            "ground_highest_confidence_recall_within_1_5m": (
                sum(error <= 1.5 for error in ground_highest_errors) / ground_frames
                if ground_frames else None
            ),
        },
    }


def benchmark_soccernet_anchored_gaps(
    manifest_path, model_path, *,
    split, maximum_clips, confidence, image_size, device, gap_frames,
    anchor_noise_metres=(),
):
    if split not in ("development", "validation") or maximum_clips <= 0:
        raise ValueError("anchored gap benchmark needs train/valid and positive clips")
    if gap_frames < 8 or gap_frames > 300 or not 0 < confidence <= 1 or image_size <= 0:
        raise ValueError("anchored gap length, confidence or image size is invalid")
    if (
        len(anchor_noise_metres) != len(set(anchor_noise_metres))
        or any(
            not np.isfinite(radius) or radius <= 0 or radius > 5
            for radius in anchor_noise_metres
        )
    ):
        raise ValueError("anchor noise radii must be distinct and in (0, 5] metres")
    try:
        import cv2
        from ultralytics import YOLO, settings
    except ImportError as error:
        raise RuntimeError(
            "anchored gap benchmark requires OpenCV and Ultralytics"
        ) from error
    manifest_bytes = Path(manifest_path).read_bytes()
    clips = _sample_soccernet_clips(
        json.loads(manifest_bytes), split, maximum_clips
    )
    model_path = Path(model_path)
    model, training = _generic_coco_model(model_path, YOLO, settings)
    from possession.video import select_edge_anchored_ball_paths

    compared = []
    reference_frames = 0
    raw_frames = 0
    raw_correct = 0
    anchored_frames = 0
    anchored_correct = 0
    stress_counts = {
        str(radius): {"accepted": 0, "correct": 0}
        for radius in anchor_noise_metres
    }
    gap_start = 200
    for clip in clips:
        label_path = Path(clip["label_path"])
        with label_path.open() as handle:
            data = json.load(handle)
        if gap_start + gap_frames >= len(data["images"]):
            continue
        objects_by_image = {}
        for item in data["annotations"]:
            if item.get("supercategory") == "object":
                objects_by_image.setdefault(str(item["image_id"]), []).append(item)

        def objects(frame):
            return objects_by_image.get(str(data["images"][frame]["image_id"]), [])

        def ball(frame):
            annotations = [
                item for item in objects(frame)
                if (item.get("attributes") or {}).get("role") == "ball"
            ]
            return _pitch_point(annotations[0]) if len(annotations) == 1 else None

        left, right = ball(gap_start - 1), ball(gap_start + gap_frames)
        if left is None or right is None:
            continue
        image_dir = label_path.parent / data["info"]["im_dir"]
        candidates_by_frame = []
        references = []
        for frame in range(gap_start, gap_start + gap_frames):
            annotations = objects(frame)
            references.append(ball(frame))
            offset = frame - gap_start
            if gap_frames > 50 and 25 <= offset < gap_frames - 25:
                candidates_by_frame.append([])
                continue
            matrix = _player_ground_homography(annotations, cv2)
            if matrix is None:
                candidates_by_frame.append([])
                continue
            image_path = image_dir / data["images"][frame]["file_name"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("{}: cannot read annotated frame".format(image_path))
            pixels = detect_tiled_ball_pixels(
                image, model, class_id=32, confidence=confidence,
                image_size=image_size, device=device, overlap_fraction=0.1,
                merge_radius_pixels=20,
            )
            candidates_by_frame.append(
                _project_ground_candidates(pixels, matrix, cv2)
            )

        options = {
            "fps": float(data["info"]["frame_rate"]),
            "pitch": (105.0, 68.0),
            "beam_width": 24,
            "maximum_missing_frames": 5,
            "process_acceleration_std": 15.0,
            "measurement_noise_base": 0.5,
            "measurement_noise_minimum": 0.1,
            "measurement_noise_maximum": 2.0,
            "initial_velocity_std": 30.0,
            "innovation_gate": 4.0,
            "missing_penalty": 3.0,
            "confidence_reward": 2.0,
        }
        anchor_offset = np.asarray([52.5, 34.0])
        path = select_edge_anchored_ball_paths(
            candidates_by_frame,
            np.asarray(left) + anchor_offset,
            np.asarray(right) + anchor_offset,
            **options
        )
        xy = path.xy - np.asarray([52.5, 34.0])
        selected = path.candidate_index
        for index, reference in enumerate(references):
            if reference is None:
                continue
            reference_frames += 1
            if candidates_by_frame[index]:
                strongest = max(
                    candidates_by_frame[index], key=lambda item: item["confidence"]
                )
                raw_frames += 1
                raw_correct += (
                    np.linalg.norm(strongest["ground_metres"] - reference) <= 1.5
                )
            if selected[index] >= 0:
                anchored_frames += 1
                anchored_correct += (
                    np.linalg.norm(xy[index] - reference) <= 1.5
                )
        for radius in anchor_noise_metres:
            stress = stress_counts[str(radius)]
            for offsets in _anchor_perturbations(
                clip["sequence_id"], radius
            ):
                perturbed = select_edge_anchored_ball_paths(
                    candidates_by_frame,
                    np.asarray(left) + anchor_offset + offsets[0],
                    np.asarray(right) + anchor_offset + offsets[1],
                    **options
                )
                for index, reference in enumerate(references):
                    if reference is None or perturbed.candidate_index[index] < 0:
                        continue
                    stress["accepted"] += 1
                    stress["correct"] += (
                        np.linalg.norm(
                            perturbed.xy[index] - [52.5, 34.0] - reference
                        ) <= 1.5
                    )
        compared.append(clip["sequence_id"])

    if not reference_frames:
        raise ValueError("no human-ball frames with paired annotated anchors")
    result = {
        "schema_version": 1,
        "dataset": "SoccerNet Game State Reconstruction",
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_training_data": training,
        "split": split,
        "gap_start_frame": gap_start,
        "gap_frames": gap_frames,
        "sampled_clips": [clip["sequence_id"] for clip in clips],
        "compared_clips": compared,
        "confidence": confidence,
        "image_size": image_size,
        "policy": {
            "anchors": "external human annotated balls at both masked gap endpoints",
            "homography": "external human annotated player feet only",
            "inference_ball_labels_used_inside_gap": False,
            "SoccerNet_specific_checkpoint_evaluated": False,
            "test_split_used": False,
            "possession_truth_available": False,
            "challenge_data_used": False,
            "training_overlap": "generic COCO; no clip-level overlap guarantee",
        },
        "metrics": {
            "human_ball_reference_frames": reference_frames,
            "highest_confidence_detected_frames": raw_frames,
            "highest_confidence_correct_within_1_5m": int(raw_correct),
            "highest_confidence_recall_within_1_5m": raw_correct / reference_frames,
            "anchored_detected_frames": anchored_frames,
            "anchored_correct_within_1_5m": int(anchored_correct),
            "anchored_precision_within_1_5m": (
                anchored_correct / anchored_frames if anchored_frames else None
            ),
            "anchored_recall_within_1_5m": anchored_correct / reference_frames,
        },
    }
    if anchor_noise_metres:
        result["anchor_noise_stress"] = {
            "method": "four paired, deterministic 90-degree rotations per clip; independent left/right directions at each fixed radius",
            "trials_per_clip": 4,
            "results": {
                radius: {
                    "reference_frames": reference_frames * 4,
                    "highest_confidence_correct": int(raw_correct) * 4,
                    "accepted_positions": counts["accepted"],
                    "correct_positions": int(counts["correct"]),
                    "precision_within_1_5m": (
                        counts["correct"] / counts["accepted"]
                        if counts["accepted"] else None
                    ),
                    "recall_within_1_5m": (
                        counts["correct"] / (reference_frames * 4)
                    ),
                }
                for radius, counts in stress_counts.items()
            },
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize external SoccerNet ball geometry and motion."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector-model")
    parser.add_argument("--split", choices=("development", "validation"),
                        default="development")
    parser.add_argument("--frame-stride", type=int, default=75)
    parser.add_argument("--maximum-clips", type=int, default=12)
    parser.add_argument("--confidence", type=float, default=0.03)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--anchored-gap-frames", type=int)
    parser.add_argument(
        "--anchor-noise-metres", type=float, action="append", default=[],
        help="repeat for deterministic noisy endpoint stress (anchored gaps only)",
    )
    args = parser.parse_args(argv)
    try:
        if args.anchor_noise_metres and args.anchored_gap_frames is None:
            raise ValueError("--anchor-noise-metres requires --anchored-gap-frames")
        if args.anchored_gap_frames is not None:
            if not args.detector_model:
                raise ValueError("anchored gap benchmark requires --detector-model")
            if args.frame_stride != 75:
                raise ValueError(
                    "--frame-stride applies only to sparse detector benchmarks"
                )
            metrics = benchmark_soccernet_anchored_gaps(
                args.manifest, args.detector_model, split=args.split,
                maximum_clips=args.maximum_clips, confidence=args.confidence,
                image_size=args.image_size, device=args.device,
                gap_frames=args.anchored_gap_frames,
                anchor_noise_metres=args.anchor_noise_metres,
            )
        elif args.detector_model:
            metrics = benchmark_soccernet_detector(
                args.manifest, args.detector_model,
                split=args.split, frame_stride=args.frame_stride,
                maximum_clips=args.maximum_clips, confidence=args.confidence,
                image_size=args.image_size, device=args.device,
            )
        else:
            metrics = build_soccernet_geometry_metrics(args.manifest)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (IOError, OSError, ValueError) as error:
        parser.exit(1, "error: {}\n".format(error))
    print(json.dumps(
        metrics["metrics"] if args.detector_model else metrics["splits"],
        indent=2, sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
