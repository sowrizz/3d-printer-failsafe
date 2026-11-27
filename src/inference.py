from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from src.dataset import CLASS_NAMES
from src.model import build_model, checkpoint_mapping


DEFAULT_PREPROCESSING = {
    "image_size": 224,
    "resize_size": 256,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode_uploaded_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            return source.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"The uploaded image is invalid or unreadable: {exc}") from exc


def preprocessing_transform(config: dict[str, Any] | None = None) -> transforms.Compose:
    config = config or DEFAULT_PREPROCESSING
    return transforms.Compose(
        [
            transforms.Resize(int(config["resize_size"])),
            transforms.CenterCrop(int(config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"]),
        ]
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, int], dict[str, Any], torch.device]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {checkpoint_path}. "
            "Train it first with: python train.py"
        )
    device = device or select_device()
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load model checkpoint {checkpoint_path}: {exc}") from exc

    mapping = checkpoint_mapping(checkpoint)
    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError(
            "Checkpoint architecture must be 'mobilenet_v3_small'; "
            f"received {checkpoint.get('architecture')!r}."
        )
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Model class names do not match the dataset folders.")
    model = build_model(num_classes=len(mapping), pretrained=False)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise ValueError(f"Checkpoint has incompatible model weights: {exc}") from exc
    model.to(device)
    model.eval()
    config = checkpoint.get("preprocessing", DEFAULT_PREPROCESSING)
    return model, mapping, config, device


def predict_batch(
    images: list[Image.Image],
    model: torch.nn.Module,
    class_to_idx: dict[str, int],
    preprocessing: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    if not images:
        return []
    model_device = next(model.parameters()).device
    if device is None or model_device != device:
        device = model_device
    transform = preprocessing_transform(preprocessing)
    batch = torch.stack([transform(image.convert("RGB")) for image in images])
    model.eval()

    try:
        with torch.inference_mode():
            logits = model(batch.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu()
    except RuntimeError as exc:
        if device.type != "cuda" or "out of memory" not in str(exc).casefold():
            raise
        torch.cuda.empty_cache()
        model.to("cpu")
        device = torch.device("cpu")
        with torch.inference_mode():
            probabilities = torch.softmax(model(batch), dim=1)

    idx_to_class = {index: name for name, index in class_to_idx.items()}
    if set(idx_to_class) != set(range(probabilities.shape[1])):
        raise ValueError("Checkpoint mapping does not cover every model output.")

    results: list[dict[str, Any]] = []
    for row in probabilities:
        top_indices = torch.topk(row, k=min(3, row.numel())).indices.tolist()
        top_predictions = [
            {
                "class": idx_to_class[index],
                "confidence": float(row[index].item()),
            }
            for index in top_indices
        ]
        results.append(
            {
                "predicted_class": top_predictions[0]["class"],
                "confidence": top_predictions[0]["confidence"],
                "top_predictions": top_predictions,
                "device": str(device),
            }
        )
    return results


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    class_to_idx: dict[str, int],
    preprocessing: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    return predict_batch([image], model, class_to_idx, preprocessing, device)[0]
