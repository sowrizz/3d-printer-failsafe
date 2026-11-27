from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.dataset import CLASS_NAMES


def build_model(
    num_classes: int = len(CLASS_NAMES),
    pretrained: bool = False,
) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, num_classes)
    return model


def checkpoint_mapping(checkpoint: dict[str, Any]) -> dict[str, int]:
    mapping = checkpoint.get("class_to_idx")
    expected = {name: index for index, name in enumerate(CLASS_NAMES)}
    if mapping != expected:
        raise ValueError(
            "Model class mapping does not match the six dataset folders. "
            f"Expected {expected}; received {mapping}."
        )
    return mapping
