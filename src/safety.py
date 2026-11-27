from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyDecision:
    status: str
    action: str
    message: str
    stop_requested: bool


def evaluate_safety(
    predicted_class: str | None,
    confidence: float | None,
    threshold: float = 0.70,
    source: str = "image",
    consecutive_defect_frames: int = 1,
    required_consecutive_frames: int = 1,
) -> SafetyDecision:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if required_consecutive_frames < 1:
        raise ValueError("required_consecutive_frames must be at least 1.")
    if confidence is None or predicted_class is None:
        return SafetyDecision(
            "UNCERTAIN — REVIEW REQUIRED",
            "Review required",
            "No reliable model result is available; safety cannot be confirmed.",
            False,
        )
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1.")
    if confidence < threshold:
        return SafetyDecision(
            "UNCERTAIN — REVIEW REQUIRED",
            "Review required",
            f"Confidence is below the {threshold:.0%} review threshold.",
            False,
        )
    if predicted_class == "Normal":
        return SafetyDecision(
            "PRINT LOOKS NORMAL",
            "No stop request",
            "The model predicts Normal. This is not a guarantee of print quality.",
            False,
        )

    if source.casefold() == "video" and consecutive_defect_frames < required_consecutive_frames:
        return SafetyDecision(
            "WARNING — CONFIRMING DEFECT",
            "Continue monitoring (simulated)",
            f"Defect prediction has not met the {required_consecutive_frames}-frame rule.",
            False,
        )

    return SafetyDecision(
        "DEFECT DETECTED",
        "STOP REQUESTED — SIMULATION ONLY",
        f"Confident {predicted_class} prediction from {source} analysis.",
        True,
    )
