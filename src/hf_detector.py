from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from ultralytics import YOLO


MODEL_REPOSITORY = "ApatheticWithoutTheA/3D-Print-Failure-Detector"
MODEL_REVISION = "ab1f9479f5561ae7b50bb50518bafc2da06b622b"
MODEL_FILENAME = "yolov11-3d-print-failure-detection"
DETECTOR_CLASS_MAPPING = {0: "spaghetti", 1: "stringing", 2: "zits"}
SUPPORTED_DEFECTS = tuple(DETECTOR_CLASS_MAPPING.values())
INFERENCE_CONFIDENCE_FLOOR = 0.001


def load_detector() -> YOLO:
    downloaded_checkpoint = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
        )
    )
    checkpoint = downloaded_checkpoint.with_name(f"{downloaded_checkpoint.name}.pt")
    if not checkpoint.is_file():
        shutil.copy2(downloaded_checkpoint, checkpoint)
    model = YOLO(checkpoint, task="detect")
    if model.task != "detect":
        raise ValueError(f"Expected an object-detection model, got {model.task!r}.")

    names = model.names
    labels = (
        [str(names[index]) for index in sorted(names)]
        if isinstance(names, dict)
        else [str(name) for name in names]
    )
    if labels != [DETECTOR_CLASS_MAPPING[index] for index in sorted(DETECTOR_CLASS_MAPPING)]:
        raise ValueError(
            "Detector checkpoint class IDs do not match the expected mapping: "
            f"{labels!r}."
        )
    return model


def predict_batch(images: list[Image.Image], model: Any) -> list[list[dict[str, Any]]]:
    if not images:
        return []
    results = model.predict(
        source=images,
        conf=INFERENCE_CONFIDENCE_FLOOR,
        verbose=False,
    )
    if len(results) != len(images):
        raise RuntimeError(
            f"Detector returned {len(results)} results for {len(images)} images."
        )

    detections_by_image: list[list[dict[str, Any]]] = []
    for result in results:
        names = result.names
        boxes = result.boxes
        detections: list[dict[str, Any]] = []
        if boxes is not None:
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.cpu().tolist()
            for box, confidence, class_id in zip(
                coordinates, confidences, class_ids
            ):
                label = str(names[int(class_id)])
                if label not in SUPPORTED_DEFECTS:
                    continue
                detections.append(
                    {
                        "class": label,
                        "confidence": float(confidence),
                        "bbox_xyxy": [round(float(value), 2) for value in box],
                    }
                )
        detections_by_image.append(detections)
    return detections_by_image


def predict_image(
    image: Image.Image,
    model: Any,
    confidence_threshold: float,
) -> dict[str, Any]:
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    detections = predict_batch([image.convert("RGB")], model)[0]
    qualified = [
        detection
        for detection in detections
        if detection["confidence"] >= confidence_threshold
    ]
    best = max(qualified, key=lambda detection: detection["confidence"], default=None)
    return {
        "detections": detections,
        "qualified_detections": qualified,
        "predicted_class": (
            best["class"] if best else "NO SUPPORTED DEFECT DETECTED"
        ),
        "confidence": best["confidence"] if best else None,
    }


def annotate_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        label = f"{detection['class']} {detection['confidence']:.1%}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        text_bounds = draw.textbbox((x1, y1), label)
        draw.rectangle(text_bounds, fill="red")
        draw.text((x1, y1), label, fill="white")
    return annotated
