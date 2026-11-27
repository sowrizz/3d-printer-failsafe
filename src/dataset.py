from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError


CLASS_NAMES = (
    "Cracking",
    "Layer_shifting",
    "Normal",
    "Off_platform",
    "Stringing",
    "Warping",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    pixel_hash: str


@dataclass
class DatasetInventory:
    records: list[ImageRecord]
    image_counts: dict[str, int]
    unique_counts: dict[str, int]
    missing_classes: list[str]
    corrupt_images: list[tuple[Path, str]]
    duplicate_images: int
    label_conflicts: list[tuple[Path, Path]]


def _pixel_hash(image: Image.Image) -> str:
    image = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}\0".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def inspect_dataset(
    dataset_dir: Path,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> DatasetInventory:
    """Validate images and remove exact pixel duplicates from training records."""
    records: list[ImageRecord] = []
    image_counts: dict[str, int] = {}
    corrupt_images: list[tuple[Path, str]] = []
    label_conflicts: list[tuple[Path, Path]] = []
    seen_hashes: dict[str, ImageRecord] = {}
    duplicate_images = 0
    missing_classes: list[str] = []
    image_paths: dict[int, list[Path]] = {}

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            missing_classes.append(class_name)
            image_counts[class_name] = 0
            image_paths[label] = []
            continue

        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        )
        image_counts[class_name] = len(paths)
        image_paths[label] = paths
        if not paths:
            missing_classes.append(class_name)

    total_images = sum(image_counts.values())
    checked_images = 0
    for label, class_name in enumerate(CLASS_NAMES):
        for path in image_paths[label]:
            try:
                with Image.open(path) as image:
                    image.load()
                    pixel_hash = _pixel_hash(image)
            except (
                OSError,
                ValueError,
                UnidentifiedImageError,
                Image.DecompressionBombError,
            ) as exc:
                corrupt_images.append((path, str(exc)))
            else:
                existing = seen_hashes.get(pixel_hash)
                if existing is not None:
                    duplicate_images += 1
                    if existing.label != label:
                        label_conflicts.append((existing.path, path))
                else:
                    record = ImageRecord(path=path, label=label, pixel_hash=pixel_hash)
                    records.append(record)
                    seen_hashes[pixel_hash] = record

            checked_images += 1
            if progress_callback and (
                checked_images % 100 == 0 or checked_images == total_images
            ):
                progress_callback(checked_images, total_images, path)

    unique_counts = {
        class_name: sum(record.label == label for record in records)
        for label, class_name in enumerate(CLASS_NAMES)
    }
    return DatasetInventory(
        records=records,
        image_counts=image_counts,
        unique_counts=unique_counts,
        missing_classes=missing_classes,
        corrupt_images=corrupt_images,
        duplicate_images=duplicate_images,
        label_conflicts=label_conflicts,
    )


def stratified_split(
    records: list[ImageRecord],
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    """Split unique image records per class to keep class proportions balanced."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1.")

    train: list[ImageRecord] = []
    validation: list[ImageRecord] = []
    test: list[ImageRecord] = []
    rng = random.Random(seed)

    for label in range(len(CLASS_NAMES)):
        group = [record for record in records if record.label == label]
        if len(group) < 3:
            raise ValueError(
                f"Class {CLASS_NAMES[label]} needs at least 3 valid unique images "
                "for a train/validation/test split."
            )
        rng.shuffle(group)
        test_count = max(1, round(len(group) * (1 - train_fraction - validation_fraction)))
        validation_count = max(1, round(len(group) * validation_fraction))
        if test_count + validation_count >= len(group):
            raise ValueError(f"Not enough images to split class {CLASS_NAMES[label]}.")

        test.extend(group[:test_count])
        validation.extend(group[test_count:test_count + validation_count])
        train.extend(group[test_count + validation_count:])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test
