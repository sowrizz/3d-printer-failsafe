# PrintGuard project file guide

## Run the app

- `app.py` — Streamlit entry point. Builds the dashboard, accepts image and video uploads, runs inference, and records simulated printer alerts.
- `requirements.txt` — Python packages needed by the app and the training utilities.
- `.streamlit/config.toml` — Streamlit upload limit and the app's dark color theme.
- `models/best_model.pth` — Trained image-classifier weights and saved preprocessing information. The app loads this local model.
- `models/class_mapping.json` — Class-name to numeric-index mapping saved with the classifier.
- `src/__init__.py` — Marks `src` as the Python package used by the app and training code.

## Application modules (`src/`)

- `src/dataset.py` — Image dataset scanning, duplicate detection, and train/validation/test split helpers.
- `src/decision.py` — Combines classifier and object-detector evidence into a print safety decision.
- `src/hf_detector.py` — Loads the hosted YOLO model, maps its labels, and draws detection boxes.
- `src/inference.py` — Loads and runs the local image classifier.
- `src/model.py` — Defines the MobileNetV3-Small classifier architecture.
- `src/safety.py` — Validates individual prediction values and applies confidence-based safety rules.
- `src/telemetry.py` — Simulates printer state, progress, and stop/acknowledge actions; it does not control a physical printer.
- `src/video.py` — Samples uploaded videos, analyzes frames, groups events, and prepares downloadable results.

## Model training and data

- `train.py` — Trains and evaluates the image classifier using `dataset/`; writes a checkpoint and metrics.
- `prepare_normal_dataset.py` — Downloads candidate normal-print images and provides a review workflow before adding approved images to the dataset.
- `requirements-normal-review.txt` — Extra packages for the optional notebook-based normal-image review workflow.
- `dataset/Cracking/` — Training images labeled as cracking.
- `dataset/Layer_shifting/` — Training images labeled as layer shifting.
- `dataset/Normal/` — Training images labeled as normal prints.
- `dataset/Off_platform/` — Training images labeled as off-platform prints.
- `dataset/Stringing/` — Training images labeled as stringing.
- `dataset/Warping/` — Training images labeled as warping.
- Image files inside those six folders are the samples for their respective labels; the app does not read them when running its pretrained checkpoint.
- `models/training_metadata.json` — Training configuration and dataset summary associated with the saved classifier.
- `outputs/test_metrics.json` — Held-out test-set metrics from model evaluation.

## Documentation and checks

- `README.md` — Setup, launch, model, training, and usage instructions.
- `PROJECT_FILES.md` — This guide to the purpose of project files and folders.
- `tests/test_app.py` — Streamlit UI and navigation checks.
- `tests/test_core.py` — Unit tests for dataset, model decisions, telemetry, inference, and video behavior.

## Local or generated folders

- `.venv/` — Local Python virtual environment with installed packages. It is machine-specific and can be recreated from `requirements.txt`; it is kept here so the existing demo environment remains available.
- `__pycache__/`, `src/__pycache__/`, `tests/__pycache__/` — Generated Python bytecode caches; safe to delete and regenerated automatically.
- `.pytest_cache/` — Generated pytest state; safe to delete.
- `dataset/Cracking/.ipynb_checkpoints/` — Jupyter's generated checkpoint copy of an image; it is not a training sample and can be deleted.
- `archive/empty-notebooks/` — Two placeholder notebooks with no project code; safe to delete.
- `3d printer failsafw/` — A redundant nested Git repository containing only its own tiny README and Git metadata; it does not participate in the app.

## Files that can be removed for an app-only handoff

For a smaller app-only copy, `train.py`, `prepare_normal_dataset.py`, `requirements-normal-review.txt`, `dataset/`, `models/training_metadata.json`, `outputs/`, and `tests/` are not needed to launch the app. Keep `models/best_model.pth`, `models/class_mapping.json`, `app.py`, `requirements.txt`, `.streamlit/config.toml`, and all of `src/`. This workspace keeps the training and evaluation assets intact.
