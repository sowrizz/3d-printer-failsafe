from __future__ import annotations

import csv
import io
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.datasets import ImageFolder

from src.dataset import CLASS_NAMES, ImageRecord, inspect_dataset, stratified_split
from src.decision import evaluate_combined
from src.inference import decode_uploaded_image, predict_image
from src.hf_detector import predict_image as predict_yolo_image
from src.model import checkpoint_mapping
from src.safety import evaluate_safety
from src.telemetry import (
    acknowledge_stop,
    acknowledge_warning,
    initial_state,
    request_stop,
    request_warning,
    reset_print,
    start_print,
    telemetry_snapshot,
)
from src.video import frames_to_csv, process_detector_video, process_video


class ConstantModel(nn.Module):
    def __init__(self, predicted_index: int = 0) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(
            [6.0 if index == predicted_index else 0.0 for index in range(6)]
        ))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(batch.shape[0], -1)


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (24, 24), color).save(path, format="PNG")


def test_dataset_inventory_ignores_non_images_and_reports_corrupt_files(
    tmp_path: Path,
) -> None:
    for label, class_name in enumerate(CLASS_NAMES):
        (tmp_path / class_name).mkdir()
        _make_image(
            tmp_path / class_name / "good.png",
            (20 + label * 30, 40, 60),
        )
        (tmp_path / class_name / "notes.csv").write_text("decision\n", encoding="utf-8")
    (tmp_path / "Normal" / "broken.jpg").write_bytes(b"not an image")

    inventory = inspect_dataset(tmp_path)

    assert inventory.image_counts["Normal"] == 2
    assert len(inventory.corrupt_images) == 1
    assert len(inventory.records) == len(CLASS_NAMES)
    assert inventory.missing_classes == []


def test_stratified_split_is_seeded_and_has_no_content_leakage() -> None:
    records = [
        ImageRecord(
            path=Path(f"{class_name}_{index}.jpg"),
            label=label,
            pixel_hash=f"{label}-{index}",
        )
        for label, class_name in enumerate(CLASS_NAMES)
        for index in range(20)
    ]

    first = stratified_split(records, seed=17)
    second = stratified_split(records, seed=17)

    assert [[item.pixel_hash for item in group] for group in first] == [
        [item.pixel_hash for item in group] for group in second
    ]
    hashes = [{item.pixel_hash for item in group} for group in first]
    assert hashes[0].isdisjoint(hashes[1])
    assert hashes[0].isdisjoint(hashes[2])
    assert hashes[1].isdisjoint(hashes[2])
    for split in first:
        assert {item.label for item in split} == set(range(len(CLASS_NAMES)))


def test_checkpoint_class_mapping_must_match_folder_order() -> None:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    assert checkpoint_mapping({"class_to_idx": mapping}) == mapping

    mapping["Normal"], mapping["Cracking"] = mapping["Cracking"], mapping["Normal"]
    with pytest.raises(ValueError, match="does not match"):
        checkpoint_mapping({"class_to_idx": mapping})


def test_class_order_matches_torchvision_imagefolder(tmp_path: Path) -> None:
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        _make_image(class_dir / "sample.png", (20 + label * 30, 40, 60))
    image_folder = ImageFolder(tmp_path)
    assert image_folder.class_to_idx == {
        name: index for index, name in enumerate(CLASS_NAMES)
    }


@pytest.mark.parametrize(
    ("predicted", "confidence", "expected_status", "stop_requested"),
    [
        ("Normal", 0.95, "PRINT LOOKS NORMAL", False),
        ("Cracking", 0.95, "DEFECT DETECTED", True),
        ("Cracking", 0.40, "UNCERTAIN — REVIEW REQUIRED", False),
        (None, None, "UNCERTAIN — REVIEW REQUIRED", False),
    ],
)
def test_safety_decisions(
    predicted: str | None,
    confidence: float | None,
    expected_status: str,
    stop_requested: bool,
) -> None:
    result = evaluate_safety(predicted, confidence, threshold=0.70)
    assert result.status == expected_status
    assert result.stop_requested is stop_requested


def test_video_stop_requires_same_confident_defect_streak() -> None:
    before_rule = evaluate_safety(
        "Warping",
        0.9,
        threshold=0.7,
        source="video",
        consecutive_defect_frames=1,
        required_consecutive_frames=2,
    )
    after_rule = evaluate_safety(
        "Warping",
        0.9,
        threshold=0.7,
        source="video",
        consecutive_defect_frames=2,
        required_consecutive_frames=2,
    )
    assert not before_rule.stop_requested
    assert before_rule.status == "WARNING — CONFIRMING DEFECT"
    assert after_rule.stop_requested


def test_image_inference_returns_real_model_probabilities() -> None:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    image = Image.new("RGB", (256, 256), "white")

    result = predict_image(
        image,
        ConstantModel(predicted_index=3),
        mapping,
        device=torch.device("cpu"),
    )

    assert result["predicted_class"] == "Off_platform"
    assert result["top_predictions"][0]["class"] == "Off_platform"
    assert result["confidence"] > 0.9
    assert len(result["top_predictions"]) == 3


def test_uploaded_image_decoding_accepts_rgb_and_rejects_invalid_bytes() -> None:
    output = io.BytesIO()
    Image.new("RGB", (20, 20), (12, 34, 56)).save(output, format="PNG")

    decoded = decode_uploaded_image(output.getvalue())

    assert decoded.mode == "RGB"
    assert decoded.size == (20, 20)
    with pytest.raises(ValueError, match="invalid or unreadable"):
        decode_uploaded_image(b"not an image")


def test_pretrained_detector_image_results_keep_labels_confidence_and_boxes() -> None:
    class FakeBoxes:
        xyxy = torch.tensor([[1.25, 2.5, 20.0, 30.0]])
        conf = torch.tensor([0.91])
        cls = torch.tensor([1.0])

    class FakeDetector:
        def predict(self, **_kwargs):
            return [
                SimpleNamespace(
                    names={0: "spaghetti", 1: "stringing", 2: "zits"},
                    boxes=FakeBoxes(),
                )
            ]

    result = predict_yolo_image(
        Image.new("RGB", (32, 32), "white"),
        FakeDetector(),
        confidence_threshold=0.70,
    )

    assert result["predicted_class"] == "stringing"
    assert result["confidence"] == pytest.approx(0.91, abs=1e-5)
    assert result["qualified_detections"][0]["bbox_xyxy"] == [
        1.25,
        2.5,
        20.0,
        30.0,
    ]


def test_pretrained_detector_no_detections_is_not_labeled_defect_free() -> None:
    class FakeDetector:
        def predict(self, **_kwargs):
            return [
                SimpleNamespace(
                    names={0: "spaghetti", 1: "stringing", 2: "zits"},
                    boxes=None,
                )
            ]

    result = predict_yolo_image(
        Image.new("RGB", (32, 32), "white"),
        FakeDetector(),
        confidence_threshold=0.70,
    )

    assert result["predicted_class"] == "NO SUPPORTED DEFECT DETECTED"
    assert result["detections"] == []
    assert result["confidence"] is None


def test_combined_decision_alerts_on_custom_defect_without_yolo_detection() -> None:
    result = evaluate_combined(
        {"predicted_class": "Warping", "confidence": 0.91},
        [],
        classifier_threshold=0.70,
        detector_threshold=0.75,
    )

    assert result["alert"]
    assert result["status"] == "POTENTIAL DEFECT DETECTED"
    assert result["alert_sources"] == ["Custom classifier"]
    assert result["manual_review_recommended"]


def test_combined_decision_preserves_model_disagreement() -> None:
    result = evaluate_combined(
        {"predicted_class": "Normal", "confidence": 0.94},
        [{"class": "spaghetti", "confidence": 0.88, "bbox_xyxy": [1, 2, 3, 4]}],
    )

    assert result["alert"]
    assert result["conflict"]
    assert result["status"] == "CONFLICTING RESULTS — REVIEW REQUIRED"
    assert result["manual_review_recommended"]
    assert result["alert_sources"] == ["YOLO detector"]


def test_combined_decision_does_not_alert_on_low_confidence_or_empty_yolo() -> None:
    result = evaluate_combined(
        {"predicted_class": "Cracking", "confidence": 0.42},
        [],
    )

    assert not result["alert"]
    assert result["manual_review_recommended"]
    assert result["status"] == "REVIEW REQUIRED"


def test_combined_decision_treats_invalid_outputs_as_uncertain() -> None:
    result = evaluate_combined(
        {"predicted_class": "Cracking", "confidence": float("nan")},
        [{"class": "spaghetti", "confidence": 1.2}],
    )

    assert not result["alert"]
    assert result["status"] == "REVIEW REQUIRED"
    assert result["manual_review_recommended"]
    assert "Custom classifier returned an invalid prediction" in result["evidence"]
    assert "YOLO returned an invalid detection" in result["evidence"]


def test_combined_decision_keeps_compatible_model_evidence_separate() -> None:
    result = evaluate_combined(
        {"predicted_class": "Stringing", "confidence": 0.91},
        [{"class": "stringing", "confidence": 0.83, "bbox_xyxy": [1, 2, 3, 4]}],
    )

    assert result["alert"]
    assert not result["conflict"]
    assert result["alert_sources"] == ["Custom classifier", "YOLO detector"]
    assert result["alert_defects"] == [
        {
            "class": "Stringing",
            "sources": ["Custom classifier", "YOLO detector"],
        }
    ]


def _make_test_video(temp_path: Path) -> bytes:
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 32),
    )
    if not writer.isOpened():
        writer.release()
        if temp_path.exists():
            temp_path.unlink()
        pytest.skip("This OpenCV build has no MJPG video writer.")
    for index in range(8):
        frame = np.full((32, 32, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    video_bytes = temp_path.read_bytes()
    temp_path.unlink()
    return video_bytes


def test_pretrained_video_detection_groups_events_and_exports_boxes(
    monkeypatch, tmp_path: Path
) -> None:
    import src.video as video_module

    video_bytes = _make_test_video(tmp_path / "detector_video.avi")
    labels = [
        [{"class": "spaghetti", "confidence": 0.92, "bbox_xyxy": [1, 2, 20, 25]}],
        [{"class": "spaghetti", "confidence": 0.91, "bbox_xyxy": [1, 2, 20, 25]}],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    call_index = 0

    def fake_predict_batch(images, model):
        nonlocal call_index
        predictions = labels[call_index : call_index + len(images)]
        call_index += len(images)
        return predictions

    monkeypatch.setattr(video_module, "predict_detector_batch", fake_predict_batch)
    result = process_detector_video(
        video_bytes,
        model=object(),
        sample_interval_seconds=0.1,
        batch_size=3,
        max_sampled_frames=20,
        confidence_threshold=0.70,
        required_consecutive_frames=2,
        filename_suffix=".avi",
    )

    assert result["metadata"]["sampled_frames"] == 8
    assert len(result["events"]) == 1
    assert result["events"][0]["predicted_defect"] == "spaghetti"
    assert result["frames"][0]["detection_count"] == 1
    assert result["frames"][2]["predicted_class"] == "NO SUPPORTED DEFECT DETECTED"
    csv_rows = list(
        csv.DictReader(io.StringIO(frames_to_csv(result["frames"]).decode()))
    )
    assert len(csv_rows) == 8
    assert "spaghetti" in csv_rows[0]["detections"]
    assert csv_rows[0]["timestamp_seconds"] == "0.0"


def test_combined_video_runs_both_models_per_sampled_frame(
    monkeypatch, tmp_path: Path
) -> None:
    import src.video as video_module

    video_bytes = _make_test_video(tmp_path / "combined_video.avi")
    detector_calls: list[Image.Image] = []
    classifier_calls: list[Image.Image] = []

    def fake_detector_batch(images, _model):
        detector_calls.extend(images)
        return [
            [
                {
                    "class": "stringing",
                    "confidence": 0.89,
                    "bbox_xyxy": [1, 2, 20, 25],
                }
            ]
            for _image in images
        ]

    def fake_classifier_batch(images, _model, _mapping, _preprocessing, _device):
        classifier_calls.extend(images)
        return [
            {
                "predicted_class": "Stringing",
                "confidence": 0.92,
                "top_predictions": [
                    {"class": "Stringing", "confidence": 0.92}
                ],
            }
            for _image in images
        ]

    monkeypatch.setattr(video_module, "predict_detector_batch", fake_detector_batch)
    monkeypatch.setattr(video_module, "predict_classifier_batch", fake_classifier_batch)
    result = process_detector_video(
        video_bytes,
        model=object(),
        classifier_model=object(),
        class_to_idx={name: index for index, name in enumerate(CLASS_NAMES)},
        sample_interval_seconds=0.1,
        batch_size=3,
        max_sampled_frames=20,
        confidence_threshold=0.70,
        required_consecutive_frames=2,
        filename_suffix=".avi",
    )

    assert len(detector_calls) == len(classifier_calls) == 8
    assert all(
        frame["classifier_prediction"] == "Stringing"
        and frame["assessment"]["alert"]
        and not frame["assessment"]["conflict"]
        for frame in result["frames"]
    )
    assert len(result["events"]) == 1
    event = result["events"][0]
    assert event["source"] == "Custom classifier, YOLO detector"
    assert event["classifier_confidence"] == pytest.approx(0.92)
    assert event["yolo_confidence"] == pytest.approx(0.89)
    assert event["confidence"] is None
    csv_rows = list(
        csv.DictReader(io.StringIO(frames_to_csv(result["frames"]).decode()))
    )
    assert len(csv_rows) == 8
    assert csv_rows[0]["classifier_prediction"] == "Stringing"


def test_video_keeps_yolo_results_when_classifier_inference_fails(
    monkeypatch, tmp_path: Path
) -> None:
    import src.video as video_module

    video_bytes = _make_test_video(tmp_path / "partial_model_failure.avi")

    def fake_detector_batch(images, _model):
        return [
            [
                {
                    "class": "spaghetti",
                    "confidence": 0.91,
                    "bbox_xyxy": [1, 2, 20, 25],
                }
            ]
            for _image in images
        ]

    def failed_classifier_batch(*_args):
        raise RuntimeError("test classifier failure")

    monkeypatch.setattr(video_module, "predict_detector_batch", fake_detector_batch)
    monkeypatch.setattr(
        video_module, "predict_classifier_batch", failed_classifier_batch
    )
    result = process_detector_video(
        video_bytes,
        model=object(),
        classifier_model=object(),
        class_to_idx={name: index for index, name in enumerate(CLASS_NAMES)},
        sample_interval_seconds=0.1,
        batch_size=4,
        max_sampled_frames=8,
        required_consecutive_frames=2,
        filename_suffix=".avi",
    )

    assert result["metadata"]["classifier_error"] is not None
    assert result["frames"][0]["classifier_result"] is None
    assert result["frames"][0]["detections"][0]["class"] == "spaghetti"
    assert result["frames"][0]["assessment"]["alert"]
    assert result["frames"][0]["assessment"]["alert_sources"] == ["YOLO detector"]
    assert result["events"][0]["predicted_defect"] == "spaghetti"


def test_video_processing_samples_groups_events_and_exports_csv(
    monkeypatch, tmp_path: Path
) -> None:
    import src.video as video_module

    video_bytes = _make_test_video(tmp_path / "test_video_workflow.avi")
    labels = [
        ("Normal", 0.92),
        ("Normal", 0.91),
        ("Cracking", 0.90),
        ("Cracking", 0.93),
        ("Normal", 0.92),
        ("Warping", 0.45),
        ("Normal", 0.93),
        ("Normal", 0.94),
    ]
    call_index = 0

    def fake_predict_batch(images, model, class_to_idx, preprocessing, device):
        nonlocal call_index
        predictions = []
        for _image in images:
            label, confidence = labels[call_index]
            call_index += 1
            predictions.append(
                {
                    "predicted_class": label,
                    "confidence": confidence,
                    "top_predictions": [{"class": label, "confidence": confidence}],
                    "device": "cpu",
                }
            )
        return predictions

    monkeypatch.setattr(video_module, "predict_batch", fake_predict_batch)
    result = process_video(
        video_bytes,
        model=object(),
        class_to_idx={name: index for index, name in enumerate(CLASS_NAMES)},
        sample_interval_seconds=0.1,
        batch_size=3,
        max_sampled_frames=20,
        confidence_threshold=0.7,
        required_consecutive_frames=2,
        filename_suffix=".avi",
    )

    assert result["metadata"]["sampled_frames"] == 8
    assert len(result["frames"]) == 8
    assert result["events"][0]["predicted_defect"] == "Cracking"
    assert result["events"][0]["frames"] == 2
    assert not result["frames"][5]["stop_requested"]
    csv_rows = list(csv.DictReader(io.StringIO(frames_to_csv(result["frames"]).decode())))
    assert len(csv_rows) == 8
    assert csv_rows[2]["predicted_class"] == "Cracking"


def test_unreadable_video_is_reported() -> None:
    with pytest.raises(ValueError, match="could not open"):
        process_video(
            b"not a video",
            model=object(),
            class_to_idx={name: index for index, name in enumerate(CLASS_NAMES)},
            filename_suffix=".mp4",
        )


def test_telemetry_and_stop_state_are_explicit_and_persistent() -> None:
    state = initial_state()
    start_print(state, now=100.0)
    snapshot = telemetry_snapshot(state, now=160.0)
    assert snapshot["simulated"] is True
    assert snapshot["printer_status"] == "PRINTING"
    assert snapshot["progress"] == 20.0
    assert 20 <= snapshot["nozzle_c"] <= 230
    assert 20 <= snapshot["bed_c"] <= 80

    request_stop(state, "Confident simulated defect")
    assert state["status"] == "STOP REQUESTED"
    acknowledge_stop(state)
    assert state["status"] == "STOPPED (SIMULATED)"
    telemetry_snapshot(state, now=200.0)
    assert state["status"] == "STOPPED (SIMULATED)"
    reset_print(state)
    assert state["status"] == "IDLE"


def test_uncertainty_warns_without_erasing_print_progress() -> None:
    state = initial_state()
    start_print(state, now=100.0)
    request_warning(state, "Low confidence")
    snapshot = telemetry_snapshot(state, now=130.0)
    assert snapshot["printer_status"] == "WARNING"
    assert snapshot["progress"] == 10.0
    telemetry_snapshot(state, now=500.0)
    assert state["status"] == "WARNING"
    acknowledge_warning(state)
    assert state["status"] == "IDLE"
