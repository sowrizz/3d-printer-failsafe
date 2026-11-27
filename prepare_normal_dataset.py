"""Download and manually review candidates for the Normal ImageFolder class."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any


DATASET_ID = "cmudrc/3d-printed-or-not"
MAX_CANDIDATES = 300
CLASS_NAMES = (
    "Normal",
    "Cracking",
    "Layer_shifting",
    "Off_platform",
    "Stringing",
    "Warping",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
CSV_FIELDS = (
    "candidate_file",
    "source",
    "source_index",
    "pixel_sha256",
    "downloaded_at",
    "decision",
    "reviewed_at",
)


def project_root() -> Path:
    return Path(__file__).resolve().parent


def find_dataset_dir(root: Path | None = None) -> Path:
    """Find the existing dataset folder, preferring the script's project root."""
    root = root or project_root()
    roots = [root, Path.cwd()]
    checked: set[Path] = set()
    for candidate_root in roots:
        for parent in (candidate_root, *candidate_root.parents):
            dataset_dir = parent / "dataset"
            resolved = dataset_dir.resolve()
            if resolved in checked:
                continue
            checked.add(resolved)
            if dataset_dir.is_dir() and any(
                (dataset_dir / class_name).is_dir() for class_name in CLASS_NAMES[1:]
            ):
                return dataset_dir

    expected = root / "dataset"
    if expected.exists():
        return expected
    expected.mkdir(parents=True, exist_ok=True)
    return expected


def paths() -> tuple[Path, Path, Path, Path]:
    dataset_dir = find_dataset_dir()
    normal_dir = dataset_dir / "Normal"
    review_dir = dataset_dir.parent / "normal_candidates_review"
    decisions_file = review_dir / "review_decisions.csv"
    normal_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir, normal_dir, review_dir, decisions_file


def _require_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required. Install it with: "
            "py -m pip install Pillow"
        ) from exc
    return Image


def _read_decisions(csv_file: Path) -> list[dict[str, str]]:
    if not csv_file.exists():
        return []
    with csv_file.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _write_decisions(csv_file: Path, rows: list[dict[str, str]]) -> None:
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=csv_file.parent,
            prefix="review_decisions_",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_name = file.name
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in CSV_FIELDS} for row in rows
            )
        os.replace(temp_name, csv_file)
    except OSError:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def _image_pixel_hash(image: Any) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}\0".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _as_pil_image(value: Any) -> Any:
    Image = _require_pillow()
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        raw = value.get("bytes")
        image_path = value.get("path")
        if raw is not None:
            return Image.open(BytesIO(raw)).convert("RGB")
        if image_path:
            return Image.open(image_path).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value)).convert("RGB")
    raise ValueError(f"Unsupported image value: {type(value).__name__}")


def _safe_candidate_path(review_dir: Path, filename: str) -> Path:
    base = Path(filename)
    destination = review_dir / base.name
    suffix = 1
    while destination.exists():
        destination = review_dir / f"{base.stem}_{suffix}{base.suffix}"
        suffix += 1
    return destination


def download_candidates(limit: int = MAX_CANDIDATES) -> int:
    """Stream the source dataset and save up to `limit` 3d_printed images."""
    if not 1 <= limit <= MAX_CANDIDATES:
        raise ValueError(f"limit must be between 1 and {MAX_CANDIDATES}.")

    _, _, review_dir, decisions_file = paths()
    Image = _require_pillow()
    try:
        from datasets import load_dataset, load_dataset_builder
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face datasets package is required. Install it with: "
            "py -m pip install datasets"
        ) from exc

    if not decisions_file.exists():
        _write_decisions(decisions_file, [])

    try:
        builder = load_dataset_builder(DATASET_ID)
        info = builder.info
        splits = list(info.splits or {})
        features = info.features
    except Exception as exc:
        raise RuntimeError(
            f"Could not inspect {DATASET_ID}. Check your internet connection "
            f"and Hugging Face access. Details: {exc}"
        ) from exc

    print(f"Dataset: {DATASET_ID}")
    print(f"Available splits: {splits}")
    print(f"Features: {features}")
    if "image" not in features or "label" not in features:
        raise RuntimeError("Expected the dataset to contain 'image' and 'label' features.")

    label_names = getattr(features["label"], "names", None)
    if not label_names:
        raise RuntimeError("The dataset label feature does not provide class names.")
    print(f"Label names: {list(label_names)}")
    target_label = next(
        (index for index, name in enumerate(label_names) if str(name).casefold() == "3d_printed"),
        None,
    )
    if target_label is None:
        raise RuntimeError("The expected '3d_printed' label is not present.")
    if not splits:
        raise RuntimeError("The dataset does not report any available splits.")

    split = "train" if "train" in splits else splits[0]
    print(f"Using split: {split} (streaming; image files are not bulk-downloaded)")

    rows = _read_decisions(decisions_file)
    known_indices = {row.get("source_index", "") for row in rows}
    known_hashes = {
        row.get("pixel_sha256", "")
        for row in rows
        if row.get("pixel_sha256")
    }
    existing_images = [
        path
        for path in review_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    for image_path in existing_images:
        try:
            with Image.open(image_path) as image:
                known_hashes.add(_image_pixel_hash(image))
        except (OSError, ValueError) as exc:
            print(f"Skipping unreadable existing candidate {image_path.name}: {exc}")

    saved_count = sum(
        1
        for row in rows
        if row.get("candidate_file")
        and (review_dir / Path(row["candidate_file"]).name).is_file()
    )
    if saved_count >= limit:
        print(f"Already have {saved_count} candidate images; nothing more to download.")
        return saved_count

    try:
        stream = load_dataset(DATASET_ID, split=split, streaming=True)
        iterator = iter(stream)
    except Exception as exc:
        raise RuntimeError(
            f"Could not start streaming the dataset. Check your internet connection. "
            f"Details: {exc}"
        ) from exc

    scanned = 0
    target_seen = 0
    while saved_count < limit:
        try:
            example = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            print(
                f"Download stopped after {scanned} rows because the dataset stream "
                f"failed: {exc}"
            )
            break

        source_index = str(scanned)
        scanned += 1
        if source_index in known_indices:
            continue

        label = example.get("label")
        if isinstance(label, str):
            is_target = label.casefold() == "3d_printed"
        else:
            try:
                is_target = int(label) == target_label
            except (TypeError, ValueError):
                print(f"Skipping row {source_index}: unreadable label {label!r}.")
                continue
        if not is_target:
            continue

        target_seen += 1
        try:
            with _as_pil_image(example.get("image")) as image:
                rgb_image = image.convert("RGB")
                pixel_hash = _image_pixel_hash(rgb_image)
                if pixel_hash in known_hashes:
                    print(f"Skipping duplicate image at source row {source_index}.")
                    known_indices.add(source_index)
                    continue
                buffer = BytesIO()
                rgb_image.save(buffer, format="JPEG", quality=95, optimize=True)
                image_bytes = buffer.getvalue()
        except Exception as exc:
            print(f"Skipping unusable image at source row {source_index}: {exc}")
            known_indices.add(source_index)
            continue

        if shutil.disk_usage(review_dir).free < len(image_bytes):
            print("Download stopped: insufficient disk space for the next image.")
            break

        filename = (
            f"normal_candidate_{int(source_index):06d}_{pixel_hash[:12]}.jpg"
        )
        destination = _safe_candidate_path(review_dir, filename)
        try:
            with destination.open("xb") as file:
                file.write(image_bytes)
        except OSError as exc:
            if destination.exists():
                try:
                    destination.unlink()
                except OSError:
                    pass
            if getattr(exc, "errno", None) == 28:
                print("Download stopped: the disk is full.")
                break
            print(f"Could not save row {source_index} ({exc}); stopping the download.")
            break

        row = {
            "candidate_file": destination.name,
            "source": DATASET_ID,
            "source_index": source_index,
            "pixel_sha256": pixel_hash,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "decision": "",
            "reviewed_at": "",
        }
        rows.append(row)
        try:
            _write_decisions(decisions_file, rows)
        except OSError as exc:
            try:
                destination.unlink()
            except OSError:
                pass
            print(
                f"Could not save the review CSV ({exc}); removed the candidate "
                "image so it cannot be left out of the review log."
            )
            break

        saved_count += 1
        known_indices.add(source_index)
        known_hashes.add(pixel_hash)
        if saved_count % 25 == 0:
            print(
                f"Saved {saved_count}/{limit} candidates "
                f"({target_seen} matching-label images checked)."
            )

    print(f"Candidate download complete: {saved_count} image(s) in {review_dir}")
    print(f"Review decisions CSV: {decisions_file}")
    return saved_count


def _hash_image_file(path: Path) -> str:
    Image = _require_pillow()
    with Image.open(path) as image:
        return _image_pixel_hash(image)


def _copy_candidate_to_normal(
    row: dict[str, str],
    review_dir: Path,
    normal_dir: Path,
    known_hashes: set[str],
) -> str:
    candidate_name = Path(row.get("candidate_file", "")).name
    candidate = review_dir / candidate_name
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate image is missing: {candidate}")

    pixel_hash = row.get("pixel_sha256") or _hash_image_file(candidate)
    if pixel_hash in known_hashes:
        return "Already present (duplicate content skipped)."

    destination = _safe_candidate_path(normal_dir, candidate.name)
    try:
        with candidate.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
    except OSError:
        if destination.exists():
            try:
                destination.unlink()
            except OSError:
                pass
        raise
    known_hashes.add(pixel_hash)
    return f"Copied to {destination.name}."


def sync_approved_images() -> None:
    """Copy saved Keep decisions into dataset/Normal without overwriting files."""
    _, normal_dir, review_dir, decisions_file = paths()
    rows = _read_decisions(decisions_file)
    approved_rows = [
        row for row in rows if row.get("decision") == "Keep as Normal"
    ]
    normal_images = [
        path
        for path in normal_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    if not approved_rows and not normal_images:
        return

    known_hashes: set[str] = set()
    Image = _require_pillow()
    for image_path in normal_images:
        try:
            with Image.open(image_path) as image:
                known_hashes.add(_image_pixel_hash(image))
        except (OSError, ValueError) as exc:
            print(f"Could not check existing Normal image {image_path.name}: {exc}")

    for row in approved_rows:
        try:
            result = _copy_candidate_to_normal(row, review_dir, normal_dir, known_hashes)
            print(f"{row.get('candidate_file', '(unnamed)')}: {result}")
        except OSError as exc:
            print(f"Could not copy approved candidate {row.get('candidate_file')}: {exc}")


def print_dataset_report() -> None:
    dataset_dir, _, review_dir, decisions_file = paths()
    rows = _read_decisions(decisions_file)
    candidate_count = sum(
        1
        for path in review_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )
    accepted_count = sum(row.get("decision") == "Keep as Normal" for row in rows)
    rejected_count = sum(row.get("decision") == "Reject" for row in rows)

    print("\nDataset validation report")
    print(f"Dataset folder: {dataset_dir}")
    print(f"Candidate images downloaded: {candidate_count}")
    print(f"Accepted as Normal (Keep decisions): {accepted_count}")
    print(f"Rejected: {rejected_count}")
    print("Image counts by class:")

    missing_or_empty: list[str] = []
    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            print(f"  {class_name}: MISSING")
            missing_or_empty.append(class_name)
            continue
        count = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        )
        print(f"  {class_name}: {count}")
        if count == 0:
            missing_or_empty.append(class_name)

    if missing_or_empty:
        print(f"Missing or empty class folders: {', '.join(missing_or_empty)}")
    else:
        print("Missing or empty class folders: none")


def create_review_widget() -> Any:
    """Create a one-image-at-a-time Jupyter interface that saves every decision."""
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as exc:
        raise RuntimeError(
            "Jupyter review needs ipywidgets and IPython. Install with: "
            "py -m pip install ipywidgets"
        ) from exc

    _, normal_dir, review_dir, decisions_file = paths()
    rows = _read_decisions(decisions_file)
    pending = [
        row
        for row in rows
        if row.get("decision") not in {"Keep as Normal", "Reject"}
        and (review_dir / Path(row.get("candidate_file", "")).name).is_file()
    ]

    known_hashes: set[str] = set()
    Image = _require_pillow()
    for image_path in normal_dir.iterdir():
        if image_path.is_file() and image_path.suffix.casefold() in IMAGE_EXTENSIONS:
            try:
                with Image.open(image_path) as image:
                    known_hashes.add(_image_pixel_hash(image))
            except (OSError, ValueError) as exc:
                print(f"Could not check existing Normal image {image_path.name}: {exc}")

    # Retry any Keep decisions saved before a notebook was closed.
    for row in rows:
        if row.get("decision") == "Keep as Normal":
            try:
                _copy_candidate_to_normal(row, review_dir, normal_dir, known_hashes)
            except OSError as exc:
                print(f"Could not sync approved candidate {row.get('candidate_file')}: {exc}")

    output = widgets.Output()
    status = widgets.HTML()
    keep_button = widgets.Button(description="Keep as Normal", button_style="success")
    reject_button = widgets.Button(description="Reject", button_style="danger")
    index = 0

    def show_current() -> None:
        nonlocal index
        with output:
            clear_output(wait=True)
            if index >= len(pending):
                display(widgets.HTML("<b>All candidates have been reviewed.</b>"))
                status.value = "Review complete. A dataset count report follows."
                print_dataset_report()
                keep_button.disabled = True
                reject_button.disabled = True
                return

            row = pending[index]
            candidate = review_dir / Path(row["candidate_file"]).name
            display(
                widgets.HTML(
                    f"<b>Candidate {index + 1} of {len(pending)}</b><br>"
                    f"{candidate.name}<br>"
                    "Keep only if it is clearly a usable, defect-free 3D print. "
                    "Reject ambiguous, visibly defective, non-printed, or unusable images."
                )
            )
            try:
                display(
                    widgets.Image(
                        value=candidate.read_bytes(),
                        format="jpeg",
                        layout=widgets.Layout(max_width="600px"),
                    )
                )
            except OSError as exc:
                display(widgets.HTML(f"<b>Image cannot be displayed:</b> {exc}"))

    def decide(decision: str) -> None:
        nonlocal index
        row = pending[index]
        row["decision"] = decision
        row["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        try:
            _write_decisions(decisions_file, rows)
        except OSError as exc:
            row["decision"] = ""
            row["reviewed_at"] = ""
            status.value = f"<b>Could not save decision:</b> {exc}"
            return

        if decision == "Keep as Normal":
            try:
                decision_status = _copy_candidate_to_normal(
                    row, review_dir, normal_dir, known_hashes
                )
            except OSError as exc:
                decision_status = (
                    f"<b>Decision saved, but copy failed:</b> {exc}. "
                    "It will be retried when the review is reopened."
                )
        else:
            decision_status = "Rejected; this image was not copied into the dataset."

        index += 1
        show_current()
        if index < len(pending):
            status.value = decision_status

    keep_button.on_click(lambda _button: decide("Keep as Normal"))
    reject_button.on_click(lambda _button: decide("Reject"))
    controls = widgets.HBox((keep_button, reject_button))
    status.value = f"Reviewed decisions are saved in {decisions_file.name}."
    show_current()
    return widgets.VBox((status, output, controls))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and review Normal-class image candidates."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("download", "report"),
        default="download",
        help="'download' candidates or print the current dataset report",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_CANDIDATES,
        help=f"maximum candidate images to save (1-{MAX_CANDIDATES})",
    )
    args = parser.parse_args()

    if args.command == "download":
        download_candidates(args.limit)
    else:
        sync_approved_images()
    print_dataset_report()


if __name__ == "__main__":
    main()
