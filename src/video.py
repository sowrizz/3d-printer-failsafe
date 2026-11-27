from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from typing import Any, Callable

import cv2
from PIL import Image

from src.decision import evaluate_combined
from src.hf_detector import predict_batch as predict_detector_batch
from src.inference import predict_batch, predict_batch as predict_classifier_batch
from src.safety import evaluate_safety


MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_SAMPLED_FRAMES = 120
LOGGER = logging.getLogger(__name__)


def process_detector_video(
    video_bytes: bytes,
    model: Any | None,
    sample_interval_seconds: float = 1.0,
    batch_size: int = 8,
    max_sampled_frames: int = 40,
    confidence_threshold: float = 0.70,
    required_consecutive_frames: int = 2,
    progress_callback: Callable[[float], None] | None = None,
    filename_suffix: str = ".mp4",
    classifier_model: Any | None = None,
    class_to_idx: dict[str, int] | None = None,
    preprocessing: dict[str, Any] | None = None,
    device: Any = None,
    classifier_confidence_threshold: float | None = None,
    detector_confidence_threshold: float | None = None,
    classifier_error: str | None = None,
    detector_error: str | None = None,
) -> dict[str, Any]:
    if not video_bytes:
        raise ValueError("The uploaded video is empty.")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise ValueError("Video exceeds the 100 MB processing limit.")
    if sample_interval_seconds <= 0:
        raise ValueError("Sample interval must be greater than zero.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least one.")
    if not 1 <= max_sampled_frames <= MAX_SAMPLED_FRAMES:
        raise ValueError(f"Sample count must be between 1 and {MAX_SAMPLED_FRAMES}.")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    classifier_threshold = (
        confidence_threshold
        if classifier_confidence_threshold is None
        else classifier_confidence_threshold
    )
    detector_threshold = (
        confidence_threshold
        if detector_confidence_threshold is None
        else detector_confidence_threshold
    )
    if not 0 <= classifier_threshold <= 1 or not 0 <= detector_threshold <= 1:
        raise ValueError("Model confidence thresholds must be between 0 and 1.")
    if required_consecutive_frames < 1:
        raise ValueError("Consecutive-frame rule must be at least one.")
    if filename_suffix.casefold() not in {
        ".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"
    }:
        raise ValueError("Unsupported video file extension.")

    temp_path: str | None = None
    capture: cv2.VideoCapture | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=filename_suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(video_bytes)

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError(
                "OpenCV could not open this video; it may be unreadable or unsupported."
            )

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video metadata is invalid or the video has no readable frames.")
        if fps <= 0:
            raise ValueError("OpenCV could not determine the video frame rate.")

        frame_step = max(1, round(fps * sample_interval_seconds))
        sampled_total = min(
            max_sampled_frames,
            (frame_count + frame_step - 1) // frame_step,
        )
        pending_images: list[Image.Image] = []
        pending_indices: list[int] = []
        frame_records: list[dict[str, Any]] = []
        sampled_count = 0
        if classifier_model is None and classifier_error is None:
            classifier_error = "Custom classifier unavailable."
        elif classifier_model is not None and class_to_idx is None:
            classifier_error = "Custom classifier class mapping unavailable."
        if model is None and detector_error is None:
            detector_error = "YOLO detector unavailable."

        def flush_batch() -> None:
            nonlocal classifier_error, detector_error
            if not pending_images:
                return
            batch_detections: list[list[dict[str, Any]]] = [
                [] for _ in pending_images
            ]
            classifier_predictions: list[dict[str, Any] | None] = [
                None for _ in pending_images
            ]
            if model is not None and detector_error is None:
                try:
                    batch_detections = predict_detector_batch(pending_images, model)
                except (RuntimeError, TypeError, ValueError, OSError, KeyError, IndexError) as exc:
                    detector_error = f"{type(exc).__name__}: {exc}"
                    LOGGER.exception("YOLO inference failed for sampled video frames.")
            if (
                classifier_model is not None
                and classifier_error is None
                and class_to_idx is not None
            ):
                try:
                    classifier_predictions = predict_classifier_batch(
                        pending_images,
                        classifier_model,
                        class_to_idx,
                        preprocessing,
                        device,
                    )
                except (RuntimeError, TypeError, ValueError, OSError, KeyError, IndexError) as exc:
                    classifier_error = f"{type(exc).__name__}: {exc}"
                    LOGGER.exception(
                        "Custom classifier inference failed for sampled video frames."
                    )

            for image, frame_index, detections, classifier_result in zip(
                pending_images,
                pending_indices,
                batch_detections,
                classifier_predictions,
            ):
                qualified = [
                    detection
                    for detection in detections
                    if detection["confidence"] >= detector_threshold
                ]
                best = max(
                    qualified or detections,
                    key=lambda detection: detection["confidence"],
                    default=None,
                )
                assessment = evaluate_combined(
                    classifier_result,
                    detections if model is not None and detector_error is None else None,
                    classifier_threshold=classifier_threshold,
                    detector_threshold=detector_threshold,
                    classifier_error=classifier_error,
                    detector_error=detector_error,
                )
                frame_records.append(
                    {
                        "timestamp_seconds": frame_index / fps,
                        "frame_index": frame_index,
                        "thumbnail": image_thumbnail(image),
                        "detections": detections,
                        "classifier_result": classifier_result,
                        "classifier_prediction": (
                            classifier_result["predicted_class"]
                            if classifier_result
                            else None
                        ),
                        "classifier_confidence": (
                            classifier_result["confidence"]
                            if classifier_result
                            else None
                        ),
                        "classifier_top_predictions": (
                            classifier_result.get("top_predictions", [])
                            if classifier_result
                            else []
                        ),
                        "classifier_error": classifier_error,
                        "detector_error": detector_error,
                        "assessment": assessment,
                        "detection_count": len(qualified),
                        "predicted_class": (
                            best["class"]
                            if best
                            else "NO SUPPORTED DEFECT DETECTED"
                        ),
                        "confidence": best["confidence"] if best else None,
                        "safety_status": assessment["status"],
                        "action": assessment["action"],
                        "stop_requested": assessment["alert"],
                    }
                )
            pending_images.clear()
            pending_indices.clear()

        current_frame_index = 0
        while sampled_count < max_sampled_frames:
            success, frame = capture.read()
            if not success:
                if current_frame_index == 0:
                    raise ValueError("OpenCV could not decode a video frame.")
                break
            if current_frame_index % frame_step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image.thumbnail((1024, 1024))
                pending_images.append(image)
                pending_indices.append(current_frame_index)
                sampled_count += 1
                if len(pending_images) >= batch_size:
                    flush_batch()
                if progress_callback:
                    progress_callback(
                        min(1.0, sampled_count / max(1, sampled_total))
                    )
            current_frame_index += 1

        flush_batch()
        if not frame_records:
            raise ValueError("No frames could be sampled from this video.")

        streak_counts: dict[str, int] = {}
        for result in frame_records:
            frame_defects = result["assessment"]["alert_defects"]
            active_classes = {item["class"].casefold() for item in frame_defects}
            streak_counts = {
                key: value + 1 if key in active_classes else 0
                for key, value in streak_counts.items()
            }
            for item in frame_defects:
                key = item["class"].casefold()
                if key not in streak_counts:
                    streak_counts[key] = 1
            matching_streak = max(streak_counts.values(), default=0)
            alert_confirmed = (
                result["assessment"]["alert"]
                and matching_streak >= required_consecutive_frames
            )
            if result["assessment"]["alert"] and not alert_confirmed:
                result["safety_status"] = "WARNING — CONFIRMING DEFECT"
                result["action"] = "Continue analysis"
            result.update(
                stop_requested=alert_confirmed,
                alert_decision=alert_confirmed,
                defect_streak=matching_streak,
                required_consecutive_frames=required_consecutive_frames,
            )

        return {
            "metadata": {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_seconds": duration,
                "sampled_frames": len(frame_records),
                "sample_interval_seconds": sample_interval_seconds,
                "confidence_threshold": confidence_threshold,
                "classifier_confidence_threshold": classifier_threshold,
                "detector_confidence_threshold": detector_threshold,
                "required_consecutive_frames": required_consecutive_frames,
                "classifier_available": classifier_model is not None,
                "detector_available": model is not None,
                "classifier_error": classifier_error,
                "detector_error": detector_error,
            },
            "frames": frame_records,
            "events": group_defect_events(
                frame_records,
                confidence_threshold=confidence_threshold,
                required_consecutive_frames=required_consecutive_frames,
            ),
        }
    except cv2.error as exc:
        raise ValueError(f"OpenCV could not decode this video: {exc}") from exc
    finally:
        if capture is not None:
            capture.release()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def image_thumbnail(image: Image.Image) -> Image.Image:
    thumbnail = image.copy()
    thumbnail.thumbnail((480, 320))
    return thumbnail


def process_video(
    video_bytes: bytes,
    model: Any,
    class_to_idx: dict[str, int],
    preprocessing: dict[str, Any] | None = None,
    device: Any = None,
    sample_interval_seconds: float = 1.0,
    batch_size: int = 8,
    max_sampled_frames: int = 40,
    confidence_threshold: float = 0.70,
    required_consecutive_frames: int = 2,
    progress_callback: Callable[[float], None] | None = None,
    filename_suffix: str = ".mp4",
) -> dict[str, Any]:
    if not video_bytes:
        raise ValueError("The uploaded video is empty.")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise ValueError("Video exceeds the 100 MB processing limit.")
    if sample_interval_seconds <= 0:
        raise ValueError("Sample interval must be greater than zero.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least one.")
    if not 1 <= max_sampled_frames <= MAX_SAMPLED_FRAMES:
        raise ValueError(f"Sample count must be between 1 and {MAX_SAMPLED_FRAMES}.")
    if required_consecutive_frames < 1:
        raise ValueError("Consecutive-frame rule must be at least one.")
    if filename_suffix.casefold() not in {
        ".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"
    }:
        raise ValueError("Unsupported video file extension.")

    temp_path: str | None = None
    capture: cv2.VideoCapture | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=filename_suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(video_bytes)

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError("OpenCV could not open this video; it may be unreadable or unsupported.")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video metadata is invalid or the video contains no readable frames.")
        if fps <= 0:
            raise ValueError("OpenCV could not determine the video frame rate.")

        frame_step = max(1, round(fps * sample_interval_seconds))
        frame_indices: list[int] = []
        frames: list[Image.Image] = []
        thumbnails: list[Image.Image] = []
        sampled_total = min(
            max_sampled_frames,
            (frame_count + frame_step - 1) // frame_step,
        )
        current_frame_index = 0

        def flush_batch() -> None:
            if not frames:
                return
            predictions = predict_batch(
                frames, model, class_to_idx, preprocessing, device
            )
            for frame_index, thumbnail, prediction in zip(
                frame_indices, thumbnails, predictions
            ):
                timestamp = frame_index / fps
                result = dict(prediction)
                result.update(
                    timestamp_seconds=timestamp,
                    frame_index=frame_index,
                    thumbnail=thumbnail,
                )
                frame_records.append(result)
            frames.clear()
            thumbnails.clear()
            frame_indices.clear()

        frame_records: list[dict[str, Any]] = []
        while len(frame_records) + len(frames) < max_sampled_frames:
            success, frame = capture.read()
            if not success:
                if current_frame_index == 0:
                    raise ValueError("OpenCV could not decode a video frame.")
                break

            frame_index = current_frame_index
            if frame_index % frame_step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inference_image = Image.fromarray(rgb)
                inference_image.thumbnail((1024, 1024))
                frames.append(inference_image)
                thumbnail = inference_image.copy()
                thumbnail.thumbnail((480, 320))
                thumbnails.append(thumbnail)
                frame_indices.append(frame_index)
                if len(frames) >= batch_size:
                    flush_batch()
                if progress_callback:
                    progress_callback(
                        min(1.0, len(frame_records) / max(1, sampled_total))
                    )
            current_frame_index += 1

        flush_batch()
        if not frame_records:
            raise ValueError("No frames could be sampled from this video.")

        streak_class: str | None = None
        streak_count = 0
        for result in frame_records:
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]
            if predicted_class != "Normal" and confidence >= confidence_threshold:
                if predicted_class == streak_class:
                    streak_count += 1
                else:
                    streak_class = predicted_class
                    streak_count = 1
            else:
                streak_class = None
                streak_count = 0

            decision = evaluate_safety(
                predicted_class,
                confidence,
                confidence_threshold,
                source="video",
                consecutive_defect_frames=streak_count,
                required_consecutive_frames=required_consecutive_frames,
            )
            result.update(
                safety_status=decision.status,
                action=decision.action,
                stop_requested=decision.stop_requested,
                defect_streak=streak_count,
            )

        return {
            "metadata": {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_seconds": duration,
                "sampled_frames": len(frame_records),
                "sample_interval_seconds": sample_interval_seconds,
                "confidence_threshold": confidence_threshold,
                "required_consecutive_frames": required_consecutive_frames,
            },
            "frames": frame_records,
            "events": group_defect_events(
                frame_records,
                confidence_threshold=confidence_threshold,
                required_consecutive_frames=required_consecutive_frames,
            ),
        }
    except cv2.error as exc:
        raise ValueError(f"OpenCV could not decode this video: {exc}") from exc
    finally:
        if capture is not None:
            capture.release()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def group_defect_events(
    frame_records: list[dict[str, Any]],
    confidence_threshold: float,
    required_consecutive_frames: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    active: dict[str, list[dict[str, Any]]] = {}

    def save_event(class_key: str, frames: list[dict[str, Any]]) -> None:
        if len(frames) < required_consecutive_frames:
            return
        label = class_key
        sources: set[str] = set()
        classifier_confidences: list[float] = []
        detector_confidences: list[float] = []
        legacy_confidences: list[float] = []
        for frame in frames:
            assessment = frame.get("assessment")
            if assessment:
                defect = next(
                    (
                        item
                        for item in assessment["alert_defects"]
                        if item["class"].casefold() == class_key
                    ),
                    None,
                )
                if defect:
                    label = defect["class"]
                    sources.update(defect["sources"])
            elif frame.get("predicted_class"):
                label = frame["predicted_class"]
                sources.add("Custom classifier")
                if frame.get("confidence") is not None:
                    legacy_confidences.append(frame["confidence"])
            classifier_result = frame.get("classifier_result")
            classifier_threshold = (
                frame["assessment"]["thresholds"]["classifier"]
                if frame.get("assessment")
                else confidence_threshold
            )
            detector_threshold = (
                frame["assessment"]["thresholds"]["detector"]
                if frame.get("assessment")
                else confidence_threshold
            )
            if (
                classifier_result
                and classifier_result["predicted_class"].casefold() == class_key
                and classifier_result["confidence"] >= classifier_threshold
            ):
                classifier_confidences.append(classifier_result["confidence"])
            detector_confidences.extend(
                item["confidence"]
                for item in frame.get("detections", [])
                if item["class"].casefold() == class_key
                and item["confidence"] >= detector_threshold
            )
        classifier_mean = (
            sum(classifier_confidences) / len(classifier_confidences)
            if classifier_confidences
            else None
        )
        detector_mean = (
            sum(detector_confidences) / len(detector_confidences)
            if detector_confidences
            else None
        )
        if legacy_confidences and classifier_mean is None and detector_mean is None:
            classifier_mean = sum(legacy_confidences) / len(legacy_confidences)
        events.append(
            {
                "start_seconds": frames[0]["timestamp_seconds"],
                "end_seconds": frames[-1]["timestamp_seconds"],
                "predicted_defect": label,
                "source": ", ".join(sorted(sources)),
                "confidence": (
                    classifier_mean
                    if classifier_mean is not None and detector_mean is None
                    else detector_mean
                    if detector_mean is not None and classifier_mean is None
                    else None
                ),
                "classifier_confidence": classifier_mean,
                "yolo_confidence": detector_mean,
                "frames": len(frames),
                "action": "STOP REQUESTED — SIMULATION ONLY",
            }
        )

    for frame in frame_records:
        assessment = frame.get("assessment")
        if assessment is None:
            prediction = frame.get("predicted_class")
            confidence = frame.get("confidence")
            items = (
                [{"class": prediction, "sources": ["YOLO detector"]}]
                if prediction not in {"Normal", "NO SUPPORTED DEFECT DETECTED"}
                and confidence is not None
                and confidence >= confidence_threshold
                else []
            )
        else:
            items = assessment["alert_defects"]
        current_keys = {item["class"].casefold() for item in items}
        for key in list(active):
            if key not in current_keys:
                save_event(key, active.pop(key))
        for item in items:
            key = item["class"].casefold()
            active.setdefault(key, []).append(frame)
    for key, frames in active.items():
        save_event(key, frames)
    return events


def frames_to_csv(frame_records: list[dict[str, Any]]) -> bytes:
    import io

    output = io.StringIO(newline="")
    fields = (
        "timestamp_seconds",
        "frame_index",
        "classifier_prediction",
        "classifier_confidence",
        "classifier_top_predictions",
        "predicted_class",
        "confidence",
        "safety_status",
        "action",
        "defect_streak",
        "alert_decision",
        "detection_count",
        "detections",
        "assessment",
    )
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for result in frame_records:
        row = {field: result.get(field, "") for field in fields}
        for field in (
            "detections",
            "classifier_top_predictions",
            "assessment",
        ):
            if isinstance(row[field], (list, dict)):
                row[field] = json.dumps(row[field], ensure_ascii=False)
        writer.writerow(row)
    return output.getvalue().encode("utf-8")
