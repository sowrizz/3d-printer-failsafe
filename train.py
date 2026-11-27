from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.dataset import CLASS_NAMES, ImageRecord, inspect_dataset, stratified_split
from src.inference import DEFAULT_PREPROCESSING
from src.model import build_model


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}


class PrintImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        records: list[ImageRecord],
        transform: transforms.Compose,
        split_name: str,
    ) -> None:
        self.records = records
        self.transform = transform
        self.encoded_images: list[bytes] = []
        resize = transforms.Resize(DEFAULT_PREPROCESSING["resize_size"])
        print(f"Caching resized {split_name} images in memory…", flush=True)
        for index, record in enumerate(records, start=1):
            with Image.open(record.path) as source:
                resized = resize(source.convert("RGB"))
                output = BytesIO()
                resized.save(output, format="JPEG", quality=90, optimize=True)
                self.encoded_images.append(output.getvalue())
            if index % 100 == 0 or index == len(records):
                print(
                    f"  Cached {index}/{len(records)} {split_name} images",
                    flush=True,
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(BytesIO(self.encoded_images[index])) as source:
            image = source.convert("RGB")
        return self.transform(image), self.records[index].label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
    }
    return metrics, np.asarray(targets), np.asarray(predictions)


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    dataset_dir = Path(args.dataset).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Inspecting image files before creating the train/validation/test splits…")
    inventory = inspect_dataset(
        dataset_dir,
        progress_callback=lambda checked, total, path: print(
            f"  Checked {checked}/{total} images ({path.parent.name}/{path.name})",
            flush=True,
        ),
    )
    print("Dataset image counts (readable files):")
    for class_name in CLASS_NAMES:
        print(
            f"  {class_name}: {inventory.image_counts[class_name]} "
            f"({inventory.unique_counts[class_name]} unique usable images)"
        )
    print(f"Corrupt/unreadable images skipped: {len(inventory.corrupt_images)}")
    print(f"Exact duplicate images skipped: {inventory.duplicate_images}")
    if inventory.label_conflicts:
        print(
            "Warning: exact duplicate image content was present in multiple classes; "
            f"{len(inventory.label_conflicts)} conflicting copy/copies were excluded."
        )
    if inventory.corrupt_images:
        for path, error in inventory.corrupt_images[:10]:
            print(f"  Unreadable: {path} ({error})")
    if inventory.missing_classes:
        raise ValueError(
            "Required class folders are missing or empty: "
            + ", ".join(inventory.missing_classes)
        )
    if not inventory.records:
        raise ValueError(f"No usable images found in {dataset_dir}.")

    train_records, validation_records, test_records = stratified_split(
        inventory.records, seed=args.seed
    )
    split_counts = {
        "train": dict(Counter(CLASS_NAMES[item.label] for item in train_records)),
        "validation": dict(Counter(CLASS_NAMES[item.label] for item in validation_records)),
        "test": dict(Counter(CLASS_NAMES[item.label] for item in test_records)),
    }
    print(f"Training device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Stratified split counts: {json.dumps(split_counts, sort_keys=True)}")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                DEFAULT_PREPROCESSING["image_size"], scale=(0.85, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DEFAULT_PREPROCESSING["mean"],
                std=DEFAULT_PREPROCESSING["std"],
            ),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.CenterCrop(DEFAULT_PREPROCESSING["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DEFAULT_PREPROCESSING["mean"],
                std=DEFAULT_PREPROCESSING["std"],
            ),
        ]
    )
    train_loader = DataLoader(
        PrintImageDataset(train_records, train_transform, "training"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        PrintImageDataset(validation_records, evaluation_transform, "validation"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        PrintImageDataset(test_records, evaluation_transform, "test"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    model = build_model(pretrained=not args.no_pretrained).to(device)

    training_counts = Counter(record.label for record in train_records)
    class_weights = torch.tensor(
        [
            len(train_records) / (len(CLASS_NAMES) * training_counts[label])
            for label in range(len(CLASS_NAMES))
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.3
    )

    checkpoint_path = model_dir / "best_model.pth"
    best_validation_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            sample_count += labels.size(0)

        validation_metrics, _, _ = _evaluate(model, validation_loader, device)
        validation_accuracy = validation_metrics["accuracy"]
        scheduler.step(validation_accuracy)
        average_loss = running_loss / max(sample_count, 1)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {average_loss:.4f} | "
            f"validation accuracy {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": list(CLASS_NAMES),
                "class_to_idx": CLASS_TO_IDX,
                "preprocessing": DEFAULT_PREPROCESSING,
                "architecture": "mobilenet_v3_small",
            }
            temp_checkpoint = checkpoint_path.with_suffix(".pth.tmp")
            torch.save(checkpoint, temp_checkpoint)
            temp_checkpoint.replace(checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping: validation accuracy has not improved.")
                break

    best_checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    test_metrics, _, _ = _evaluate(model, test_loader, device)
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "mobilenet_v3_small",
        "device": str(device),
        "seed": args.seed,
        "requested_epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "elapsed_seconds": round(time.time() - started_at, 2),
        "dataset_dir": str(dataset_dir),
        "dataset_image_counts": inventory.image_counts,
        "unique_usable_counts": inventory.unique_counts,
        "split_counts": split_counts,
        "corrupt_images_skipped": len(inventory.corrupt_images),
        "exact_duplicates_skipped": inventory.duplicate_images,
        "class_to_idx": CLASS_TO_IDX,
        "preprocessing": DEFAULT_PREPROCESSING,
        "test_metrics": test_metrics,
    }
    best_checkpoint["training_metadata"] = metadata
    torch.save(best_checkpoint, checkpoint_path)
    _atomic_json(model_dir / "class_mapping.json", CLASS_TO_IDX)
    _atomic_json(model_dir / "training_metadata.json", metadata)
    _atomic_json(output_dir / "test_metrics.json", test_metrics)

    print(f"Selected model saved to {checkpoint_path}")
    print(f"Test accuracy (held-out test set): {test_metrics['accuracy']:.4f}")
    print("Per-class test metrics:")
    for class_name in CLASS_NAMES:
        scores = test_metrics["classification_report"][class_name]
        print(
            f"  {class_name}: precision={scores['precision']:.3f}, "
            f"recall={scores['recall']:.3f}, f1={scores['f1-score']:.3f}"
        )
    print(f"Confusion matrix: {test_metrics['confusion_matrix']}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the 3D Print Failsafe MobileNetV3-Small model."
    )
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download ImageNet weights; train from random initialization.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience, and batch-size must all be positive.")
    if args.workers < 0:
        parser.error("workers cannot be negative.")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive.")
    return args


if __name__ == "__main__":
    train(parse_args())
