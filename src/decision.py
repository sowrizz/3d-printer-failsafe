from __future__ import annotations

import math
from typing import Any

from src.dataset import CLASS_NAMES
from src.hf_detector import SUPPORTED_DEFECTS


def evaluate_combined(
    classifier_result: dict[str, Any] | None,
    detections: list[dict[str, Any]] | None,
    classifier_threshold: float = 0.70,
    detector_threshold: float = 0.70,
    classifier_error: str | None = None,
    detector_error: str | None = None,
) -> dict[str, Any]:
    for label, threshold in (
        ("classifier_threshold", classifier_threshold),
        ("detector_threshold", detector_threshold),
    ):
        if not 0 <= threshold <= 1:
            raise ValueError(f"{label} must be between 0 and 1.")

    classifier_prediction: str | None = None
    classifier_confidence: float | None = None
    classifier_valid = False
    if isinstance(classifier_result, dict):
        prediction = classifier_result.get("predicted_class")
        confidence = classifier_result.get("confidence")
        if (
            prediction in CLASS_NAMES
            and isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(float(confidence))
            and 0 <= float(confidence) <= 1
        ):
            classifier_prediction = str(prediction)
            classifier_confidence = float(confidence)
            classifier_valid = True

    classifier_qualified = bool(
        classifier_valid
        and classifier_confidence is not None
        and classifier_confidence >= classifier_threshold
    )
    classifier_defect = (
        classifier_prediction
        if classifier_qualified and classifier_prediction != "Normal"
        else None
    )

    qualified_detections: list[dict[str, Any]] = []
    valid_detections: list[dict[str, Any]] = []
    invalid_detections = detections is not None and not isinstance(
        detections, (list, tuple)
    )
    detection_items = detections if isinstance(detections, (list, tuple)) else []
    for detection in detection_items:
        if not isinstance(detection, dict):
            invalid_detections = True
            continue
        label = detection.get("class")
        confidence = detection.get("confidence")
        if (
            label not in SUPPORTED_DEFECTS
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            invalid_detections = True
            continue
        valid = dict(detection)
        valid["confidence"] = float(confidence)
        valid_detections.append(valid)
        if float(confidence) >= detector_threshold:
            qualified_detections.append(valid)

    alert_defects: dict[str, dict[str, Any]] = {}
    if classifier_defect:
        alert_defects[classifier_defect.casefold()] = {
            "class": classifier_defect,
            "sources": ["Custom classifier"],
        }
    for detection in qualified_detections:
        label = str(detection["class"])
        key = label.casefold()
        item = alert_defects.setdefault(
            key, {"class": label, "sources": []}
        )
        item["sources"].append("YOLO detector")

    classifier_confident_normal = (
        classifier_qualified and classifier_prediction == "Normal"
    )
    conflict = bool(
        (classifier_confident_normal and qualified_detections)
        or (
            classifier_defect
            and any(
                detection["class"].casefold() != classifier_defect.casefold()
                for detection in qualified_detections
            )
        )
    )
    alert = bool(alert_defects)
    evidence: list[str] = []
    if classifier_valid:
        evidence.append(
            f"Custom classifier: {classifier_prediction} "
            f"({classifier_confidence:.1%})"
        )
    elif classifier_error:
        evidence.append("Custom classifier unavailable")
    elif classifier_result is not None:
        evidence.append("Custom classifier returned an invalid prediction")
    if qualified_detections:
        evidence.extend(
            f"YOLO detector: {item['class']} ({item['confidence']:.1%})"
            for item in qualified_detections
        )
    elif valid_detections:
        evidence.append(
            f"YOLO candidates below {detector_threshold:.0%} threshold"
        )
    elif detector_error:
        evidence.append("YOLO detector unavailable")
    elif invalid_detections:
        evidence.append("YOLO returned an invalid detection")
    elif detections is not None:
        evidence.append("YOLO found no supported objects")

    if conflict:
        status = "CONFLICTING RESULTS — REVIEW REQUIRED"
        manual_review = True
        reason = "The models produced incompatible high-confidence results."
    elif alert:
        status = "POTENTIAL DEFECT DETECTED"
        manual_review = bool(
            classifier_error
            or detector_error
            or not classifier_valid
            or not classifier_defect
            or not qualified_detections
        )
        sources = sorted(
            {
                source
                for item in alert_defects.values()
                for source in item["sources"]
            }
        )
        reason = "Sufficient-confidence defect evidence from " + " and ".join(sources) + "."
    elif not classifier_valid and not valid_detections and (
        classifier_error or detector_error
    ):
        status = "REVIEW REQUIRED"
        manual_review = True
        reason = "No usable result is available from the failed model(s)."
    elif not classifier_valid and classifier_result is None and not valid_detections:
        status = "REVIEW REQUIRED"
        manual_review = True
        reason = "No model produced a usable prediction."
    elif (
        (classifier_result is not None and not classifier_qualified)
        or (valid_detections and not qualified_detections)
        or (classifier_result is not None and not classifier_valid)
        or invalid_detections
    ):
        status = "REVIEW REQUIRED"
        manual_review = True
        reason = "Available prediction confidence is low or invalid."
    else:
        status = "NO HIGH-CONFIDENCE DEFECT SIGNAL"
        manual_review = True
        reason = (
            "No high-confidence defect was identified; this is not proof that "
            "the print is defect-free."
        )

    alert_sources = sorted(
        {
            source
            for item in alert_defects.values()
            for source in item["sources"]
        }
    )
    return {
        "status": status,
        "alert": alert,
        "action": "STOP REQUESTED — SIMULATION ONLY" if alert else "No stop request",
        "manual_review_recommended": manual_review,
        "conflict": conflict,
        "reason": reason,
        "evidence": evidence,
        "alert_sources": alert_sources,
        "alert_defects": list(alert_defects.values()),
        "classifier_valid": classifier_valid,
        "classifier_qualified": classifier_qualified,
        "classifier_prediction": classifier_prediction,
        "classifier_confidence": classifier_confidence,
        "detector_qualified": qualified_detections,
        "detector_candidates": valid_detections,
        "thresholds": {
            "classifier": classifier_threshold,
            "detector": detector_threshold,
        },
        "model_errors": {
            "classifier": classifier_error,
            "detector": detector_error,
        },
    }
