# PrintGuard

AI-Powered 3D Print Quality Monitoring. PrintGuard reviews uploaded images and
sampled video frames for supported defect categories. Printer telemetry and alert
actions are simulated; the app does not connect to a physical printer.

## Project layout

```text
app.py
train.py
prepare_normal_dataset.py
requirements.txt
requirements-normal-review.txt
dataset/
  Normal/
  Cracking/
  Layer_shifting/
  Off_platform/
  Stringing/
  Warping/
src/
  dataset.py
  decision.py
  hf_detector.py
  inference.py
  model.py
  safety.py
  telemetry.py
  video.py
models/
outputs/                # created by training
tests/
PROJECT_FILES.md        # purpose of each project file and folder
```

See [PROJECT_FILES.md](PROJECT_FILES.md) for the purpose of each application
module, model artifact, training asset, and generated folder.

The six class directory names are used as model labels. Images are never removed
or changed by training. Non-image files in a class folder are ignored. Corrupt
images and exact duplicate image content are reported and excluded from the
training split. Exact duplicate content is kept in only one split to prevent
train/validation/test leakage.
On high-resolution source images, the integrity scan and one-time resized-image
cache can take several minutes; later epochs reuse the in-memory cache.

The saved class-to-index mapping follows `torchvision.datasets.ImageFolder`'s
alphabetical ordering: `Cracking`, `Layer_shifting`, `Normal`, `Off_platform`,
`Stringing`, `Warping`.

## Integrated image models

Image inspections and sampled video frames are passed to both models
independently. The local MobileNetV3-Small classifier uses the checkpoint's
saved preprocessing (resize 256, center crop 224, ImageNet RGB normalization)
and predicts `Cracking`, `Layer_shifting`, `Normal`, `Off_platform`, `Stringing`,
or `Warping`. Its checkpoint and class mapping are stored in `models\`.

The YOLO detector uses its own Ultralytics preprocessing and returns bounding
boxes for the verified model class mapping: `0=spaghetti`, `1=stringing`,
`2=zits`. The custom classifier and YOLO labels are not merged. The decision
module keeps their confidence scores separate, flags incompatible confident
results as `CONFLICTING RESULTS — REVIEW REQUIRED`, and can raise a simulated
stop request from either model. A missing YOLO detection is not treated as
evidence that a print is normal.

The app caches each model for its process lifetime. The custom model loads from
`models\best_model.pth`; the detector checkpoint is cached locally after its
first download. If one model is unavailable, the other may still analyze the
image/video. Each model has its own configurable confidence threshold. Video
alerts also require the configured number of consecutive sampled frames.

The first analysis needs an internet connection to download the checkpoint.
Hugging Face Hub caches it in the user's local cache (normally
`%USERPROFILE%\.cache\huggingface\hub`); subsequent runs reuse that file and can
load it without downloading again. Streamlit also caches the loaded model for
the life of the app process.

**License discrepancy:** the Hugging Face model card declares MIT, but the
checkpoint's embedded training metadata declares AGPL-3.0. The Ultralytics
library itself is AGPL-3.0. Review the applicable terms before redistribution
or deployment; this demo does not resolve that discrepancy.

To fetch and manually review more `Normal` candidates, run:

```powershell
python prepare_normal_dataset.py download
python -m notebook
```

In a Jupyter cell, use:

```python
from IPython.display import display
from prepare_normal_dataset import create_review_widget

display(create_review_widget())
```

Only candidates marked **Keep as Normal** are copied into `dataset\Normal`.
Review decisions are saved in
`normal_candidates_review\review_decisions.csv`; the dataset source label alone
is not treated as proof of print quality.

## Windows setup

Open PowerShell and run:

```powershell
Set-Location 'D:\3D_Print_Failsafe'
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

The CUDA PyTorch wheels in this setup target CUDA 12.8 and work with a recent
NVIDIA driver. The app automatically uses CUDA when available and otherwise
falls back to CPU. Confirm GPU access with:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If PowerShell blocks virtual-environment activation, either allow it for the
current terminal or run the interpreter by its full path:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

## Train the model

Ensure all six folders contain reviewed images, then run:

```powershell
python train.py
```

The default is MobileNetV3-Small with ImageNet initialization, a fixed seed,
stratified 70/15/15 train/validation/test splits, training-only augmentation,
an in-memory cache of resized training images, class-weighted loss,
validation-based checkpoint selection, and a final
held-out test evaluation. The test set is not used for training or model
selection. Training prints actual accuracy, per-class precision/recall/F1, and
the confusion matrix; no result is assumed in advance.

Useful options:

```powershell
python train.py --epochs 12 --batch-size 32 --seed 42
python train.py --no-pretrained
```

By default, the trained checkpoint and metadata are saved to:

- `models\best_model.pth` — weights, class mapping, preprocessing configuration,
  and training metadata.
- `models\class_mapping.json` — the saved class-to-index mapping.
- `models\training_metadata.json` — dataset and training summary.
- `outputs\test_metrics.json` — held-out evaluation metrics.

## Launch PrintGuard

The custom classifier checkpoint is already provided at
`models\best_model.pth`; no retraining is required. After installing
dependencies, launch Streamlit:

```powershell
python -m streamlit run app.py
```

The dashboard includes Overview, Image Inspection, Video Analysis, Printer
Telemetry, Incident History, and System Settings.

For a quick test, open **Image Inspection**, upload a JPG, JPEG, or PNG photo,
and select **Analyze image**. Then open **Video Analysis**, upload a short MP4,
AVI, MOV, MKV, M4V, or WebM clip, set the frame interval in **System Settings**,
and select **Analyze video**. Video analysis is offline, samples frames at the configured interval,
limits processing to 120 sampled frames and 100 MB per upload, and exports
timestamps, predictions, confidence scores, and bounding boxes to CSV.

Predictions show the model-supported labels, confidence, and bounding boxes.
Each model's confidence threshold is independently configurable and can
contribute to a simulated stop request. For video, a matching defect must meet
its model's threshold across the configured number of consecutive sampled
frames before an event is logged. Classifier and detector confidence are
reported separately rather than averaged. These actions are mock telemetry
only; no physical printer is connected or stopped.

All telemetry is generated locally for demonstration. A simulated stop request
never sends hardware commands. Acknowledge or reset the simulated printer from
the dashboard; later detections do not automatically clear an alert.

## Run tests

```powershell
python -m pytest
```
