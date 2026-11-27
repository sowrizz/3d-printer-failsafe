from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.errors import XetDownloadError

from src.hf_detector import (
    DETECTOR_CLASS_MAPPING,
    SUPPORTED_DEFECTS,
    annotate_detections,
    load_detector,
    predict_image as predict_detector_image,
)
from src.decision import evaluate_combined
from src.dataset import CLASS_NAMES
from src.inference import (
    decode_uploaded_image,
    load_model as load_classifier,
    predict_image as predict_classifier_image,
    select_device,
)
from src.telemetry import (
    acknowledge_warning,
    acknowledge_stop,
    initial_state,
    request_warning,
    request_stop,
    reset_print,
    start_print,
    telemetry_snapshot,
)
from src.video import (
    MAX_SAMPLED_FRAMES,
    MAX_VIDEO_BYTES,
    frames_to_csv,
    process_detector_video,
)


IMAGE_UPLOAD_TYPES = ["jpg", "jpeg", "png"]
VIDEO_UPLOAD_TYPES = ["mp4", "avi", "mov", "mkv", "m4v", "webm"]
LOGGER = logging.getLogger(__name__)
APP_PAGES = (
    "Overview",
    "Image Inspection",
    "Video Analysis",
    "Printer Telemetry",
    "Incident History",
    "System Settings",
)

st.set_page_config(
    page_title="PrintGuard",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --pg-bg: #0b1220;
        --pg-panel: #111c2c;
        --pg-panel-strong: #152338;
        --pg-border: #26364d;
        --pg-text: #e6edf7;
        --pg-muted: #9aabc1;
        --pg-accent: #38bdf8;
        --pg-good: #34d399;
        --pg-warn: #fbbf24;
        --pg-danger: #fb7185;
    }
    .stApp { background: var(--pg-bg); color: var(--pg-text); }
    [data-testid="stHeader"] { background: rgba(11, 18, 32, .94); }
    [data-testid="stSidebar"] {
        background: #0e1726;
        border-right: 1px solid var(--pg-border);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: .65rem;
        padding: .38rem .55rem;
        transition: background .15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: var(--pg-panel-strong);
    }
    .pg-brand {
        color: var(--pg-text);
        font-size: 1.45rem;
        font-weight: 750;
        letter-spacing: -.035em;
        margin: .15rem 0 0;
    }
    .pg-subtitle {
        color: var(--pg-muted);
        font-size: .79rem;
        line-height: 1.45;
        margin: .2rem 0 1.35rem;
    }
    .pg-eyebrow {
        color: var(--pg-accent);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .pg-page-title {
        color: var(--pg-text);
        font-size: clamp(1.7rem, 3vw, 2.25rem);
        font-weight: 720;
        letter-spacing: -.04em;
        margin: 0;
    }
    .pg-page-description { color: var(--pg-muted); margin: .3rem 0 1.25rem; }
    .pg-panel {
        background: var(--pg-panel);
        border: 1px solid var(--pg-border);
        border-radius: .9rem;
        padding: 1rem 1.1rem;
    }
    .pg-status {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        border: 1px solid var(--pg-border);
        border-radius: 999px;
        color: var(--pg-muted);
        font-size: .78rem;
        padding: .34rem .68rem;
    }
    .pg-dot {
        width: .48rem;
        height: .48rem;
        border-radius: 50%;
        background: var(--pg-good);
        display: inline-block;
    }
    .pg-dot-idle { background: var(--pg-warn); }
    [data-testid="stMetric"] {
        background: var(--pg-panel);
        border: 1px solid var(--pg-border);
        border-radius: .85rem;
        padding: .9rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--pg-muted); }
    [data-testid="stMetricValue"] { color: var(--pg-text); }
    .stButton > button, .stDownloadButton > button {
        border-radius: .62rem;
        min-height: 2.55rem;
        font-weight: 600;
    }
    .stButton > button[kind="primary"] {
        background: var(--pg-accent);
        border-color: var(--pg-accent);
        color: #071321;
    }
    [data-testid="stFileUploader"] section {
        background: var(--pg-panel);
        border: 1px dashed #3c526f;
        border-radius: .8rem;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--pg-border);
        border-radius: .7rem;
        overflow: hidden;
    }
    h1, h2, h3 { letter-spacing: -.025em; }
    hr { border-color: var(--pg-border); }
    @media (max-width: 760px) {
        .pg-panel { padding: .8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def initialize_state() -> None:
    if "printer_state" not in st.session_state:
        st.session_state.printer_state = initial_state()
    if "incident_log" not in st.session_state:
        st.session_state.incident_log = []
    if "last_detection" not in st.session_state:
        st.session_state.last_detection = None
    if "image_result" not in st.session_state:
        st.session_state.image_result = None
    if "image_name" not in st.session_state:
        st.session_state.image_name = None
    if "image_source" not in st.session_state:
        st.session_state.image_source = None
    if "video_result" not in st.session_state:
        st.session_state.video_result = None
    if "video_name" not in st.session_state:
        st.session_state.video_name = None
    if "inspection_history" not in st.session_state:
        st.session_state.inspection_history = []
    if "image_upload_version" not in st.session_state:
        st.session_state.image_upload_version = 0
    if "video_upload_version" not in st.session_state:
        st.session_state.video_upload_version = 0
    if "detector_loaded" not in st.session_state:
        st.session_state.detector_loaded = False
    if "model_load_error" not in st.session_state:
        st.session_state.model_load_error = False
    if "classifier_loaded" not in st.session_state:
        st.session_state.classifier_loaded = False
    if "classifier_load_error" not in st.session_state:
        st.session_state.classifier_load_error = False
    settings_defaults = {
        "settings_classifier_confidence": 0.70,
        "settings_detector_confidence": 0.70,
        "settings_interval": 1.0,
        "settings_consecutive": 2,
    }
    for key, value in settings_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "app_page" not in st.session_state:
        st.session_state.app_page = "Overview"


def page_header(title: str, description: str) -> None:
    st.markdown(
        f'<div class="pg-eyebrow">PrintGuard · Quality monitoring</div>'
        f'<h1 class="pg-page-title">{title}</h1>'
        f'<p class="pg-page-description">{description}</p>',
        unsafe_allow_html=True,
    )


def navigate_to(page: str) -> None:
    st.session_state.app_page = page


def formatted_time(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return value
    return parsed.astimezone().strftime("%b %d, %Y · %I:%M:%S %p")


def record_inspection(
    source: str,
    filename: str,
    detections: list[dict[str, Any]],
    status: str,
    classifier_result: dict[str, Any] | None = None,
    combined_decision: dict[str, Any] | None = None,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    st.session_state.inspection_history.append(
        {
            "timestamp": timestamp,
            "source": source,
            "filename": filename,
            "detections": [dict(item) for item in detections],
            "status": status,
            "classifier_result": classifier_result,
            "combined_decision": combined_decision,
        }
    )
    st.session_state.last_detection = {
        "source": source,
        "predicted_class": (
            combined_decision["status"]
            if combined_decision and combined_decision["alert"]
            else classifier_result["predicted_class"]
            if classifier_result
            else max(detections, key=lambda item: item["confidence"])["class"]
            if detections
            else status
        ),
        "confidence": classifier_result["confidence"]
        if classifier_result
        else max((item["confidence"] for item in detections), default=None),
        "status": status,
        "timestamp": timestamp,
    }


@st.cache_resource(max_entries=1, show_spinner="Loading detection engine…")
def cached_model() -> Any:
    return load_detector()


@st.cache_resource(max_entries=1, show_spinner="Loading custom classifier…")
def cached_classifier() -> tuple[Any, dict[str, int], dict[str, Any], Any]:
    return load_classifier(
        Path(__file__).resolve().parent / "models" / "best_model.pth",
        select_device(),
    )


def get_model() -> Any | None:
    try:
        model = cached_model()
        st.session_state.detector_loaded = True
        st.session_state.model_load_error = False
        return model
    except (
        HfHubHTTPError,
        XetDownloadError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
    ) as exc:
        LOGGER.exception("PrintGuard detection engine failed to load.", exc_info=exc)
        st.session_state.detector_loaded = False
        st.session_state.model_load_error = True
        st.error("YOLO detector could not be loaded; the custom classifier can still run.")
        return None


def get_classifier() -> tuple[Any, dict[str, int], dict[str, Any], Any] | None:
    try:
        classifier = cached_classifier()
        st.session_state.classifier_loaded = True
        st.session_state.classifier_load_error = False
        return classifier
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, OSError, KeyError) as exc:
        LOGGER.exception("PrintGuard custom classifier failed to load.", exc_info=exc)
        st.session_state.classifier_loaded = False
        st.session_state.classifier_load_error = True
        return None


def model_download_message() -> None:
    st.caption(
        "Initial setup may require an internet connection. Once cached, the "
        "detection engine can be loaded locally."
    )


def add_incident(
    source: str,
    predicted_class: str,
    confidence: float | None,
    action: str,
    reason: str,
    timestamp_in_video: float | None = None,
    classifier_result: dict[str, Any] | None = None,
    yolo_detections: list[dict[str, Any]] | None = None,
    combined_decision: dict[str, Any] | None = None,
    model_confidences: dict[str, float | None] | None = None,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "predicted_defect": predicted_class,
        "confidence": round(confidence, 6) if confidence is not None else None,
        "action": action,
        "reason": reason,
        "classifier_result": classifier_result,
        "yolo_detections": yolo_detections or [],
        "combined_decision": combined_decision,
        "model_confidences": model_confidences or {},
    }
    if timestamp_in_video is not None:
        event["video_timestamp_seconds"] = round(timestamp_in_video, 3)
    st.session_state.incident_log.append(event)
    telemetry_snapshot(st.session_state.printer_state)
    request_stop(st.session_state.printer_state, reason)


def record_image_result(
    result: dict[str, Any],
    source_name: str,
    image: Any,
) -> None:
    st.session_state.image_result = result
    st.session_state.image_name = source_name
    st.session_state.image_source = image.copy()
    detections = result["detector"]["qualified_detections"]
    decision = result["decision"]
    record_inspection(
        "Image",
        source_name,
        detections,
        decision["status"],
        result["classifier"],
        decision,
    )
    if decision["alert"]:
        classifier_result = result["classifier"]
        detector_detections = result["detector"]["detections"]
        model_confidences = {
            "Custom classifier": decision["classifier_confidence"],
            "YOLO detector": max(
                (item["confidence"] for item in decision["detector_qualified"]),
                default=None,
            ),
        }
        source_names = " + ".join(decision["alert_sources"])
        predicted_defect = ", ".join(
            item["class"] for item in decision["alert_defects"]
        )
        single_model_confidence = (
            model_confidences[decision["alert_sources"][0]]
            if len(decision["alert_sources"]) == 1
            else None
        )
        add_incident(
            source=f"Image: {source_name} · {source_names}",
            predicted_class=predicted_defect,
            confidence=single_model_confidence,
            action="STOP REQUESTED — SIMULATION ONLY",
            reason=decision["reason"],
            classifier_result=classifier_result,
            yolo_detections=detector_detections,
            combined_decision=decision,
            model_confidences=model_confidences,
        )
    elif decision["manual_review_recommended"]:
        request_warning(
            st.session_state.printer_state,
            decision["reason"],
        )


def _show_prediction(result: dict[str, Any], image: Any) -> None:
    classifier = result["classifier"]
    detector = result["detector"]
    decision = result["decision"]
    detections = detector["detections"]
    if detections:
        st.image(annotate_detections(image, detections), width="stretch")
    defects = [item["class"] for item in detections]
    if classifier and classifier["predicted_class"] != "Normal":
        defects.append(classifier["predicted_class"])
    if defects:
        st.error("PRINTER STOPPED · POTENTIAL DEFECT", icon=":material/pause_circle:")
        st.markdown("**Detected:** " + " · ".join(dict.fromkeys(defects)))
    elif result["errors"].get("classifier") and result["errors"].get("detector"):
        st.warning("Inspection unavailable. Please try again.")
    else:
        st.success("No defect detected in this image.")
    with st.expander("Inspection details"):
        if classifier:
            st.write(f"Classifier: {classifier['predicted_class']} ({classifier['confidence']:.0%})")
        if detections:
            st.write("Detector: " + ", ".join(f"{item['class']} ({item['confidence']:.0%})" for item in detections))
        if result["errors"].get("classifier"):
            st.caption("Classifier unavailable")
        if result["errors"].get("detector"):
            st.caption("Object detector unavailable")


def overview_page() -> None:
    page_header("PrintGuard", "3D print defect monitoring")
    state = st.session_state.printer_state
    telemetry = telemetry_snapshot(state)
    history = st.session_state.inspection_history
    defects = [
        detection
        for inspection in history
        for detection in inspection["detections"]
    ]
    latest = st.session_state.last_detection
    state_label = state["status"]
    if state_label == "STOP REQUESTED":
        state_label = "ALERT REQUESTED · SIMULATED"
    elif state_label == "STOPPED (SIMULATED)":
        state_label = "ALERT ACKNOWLEDGED · SIMULATED"
    models_ready = (
        st.session_state.classifier_loaded and st.session_state.detector_loaded
    )
    engine_label = "Both models loaded" if models_ready else "One or more models not loaded"
    engine_class = "pg-dot" if models_ready else "pg-dot pg-dot-idle"

    st.markdown(
        f'<div class="pg-status"><span class="{engine_class}"></span>'
        f'Model status · {engine_label}</div>',
        unsafe_allow_html=True,
    )
    if state["status"] in {"STOP REQUESTED", "STOPPED (SIMULATED)"}:
        st.error("PRINTER STOPPED · DEFECT ALERT", icon=":material/pause_circle:")
    else:
        st.success("PRINTER READY", icon=":material/check_circle:")
    metrics = st.columns(3)
    metrics[0].metric("Printer status", state_label)
    metrics[1].metric("Images inspected", len(history))
    metrics[2].metric("Defects found", len(defects))
    st.caption("Printer stop is simulated for this demonstration.")

    left, right = st.columns([1.25, 1])
    with left:
        with st.container(border=True):
            st.subheader("Recent inspection activity")
            if history:
                st.dataframe(
                    [
                        {
                            "Time": formatted_time(item["timestamp"]),
                            "Source": item["source"],
                            "File": item["filename"],
                            "Result": item["status"],
                        }
                        for item in reversed(history[-8:])
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("Inspection history will appear here after your first analysis.")
        st.button("Inspect an image", type="primary", use_container_width=True,
                  key="quick_image_inspection", on_click=navigate_to,
                  args=("Image Inspection",))
    with right:
        with st.container(border=True):
            st.subheader("Printer telemetry summary")
            values = st.columns(2)
            values[0].metric("Nozzle", f"{telemetry['nozzle_c']:.1f} °C")
            values[1].metric("Bed", f"{telemetry['bed_c']:.1f} °C")
            values[0].metric("Print speed", f"{telemetry['speed_mm_s']:.1f} mm/s")
            values[1].metric(
                "Estimated remaining", f"{telemetry['remaining_minutes']} min"
            )
            st.progress(
                telemetry["progress"] / 100,
                text=f"Progress · {telemetry['progress']:.1f}%",
            )
            st.caption("Simulated values · printer status is not hardware-connected.")
        with st.container(border=True):
            st.subheader("Recent incidents")
            incidents = st.session_state.incident_log
            if incidents:
                st.dataframe(
                    [
                        {
                            "Time": formatted_time(item["timestamp"]),
                            "Defect": item["predicted_defect"],
                            "Confidence": f"{item['confidence']:.1%}",
                            "Source": item["source"],
                        }
                        for item in reversed(incidents[-5:])
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No alert incidents have been raised.")



def image_detection_page() -> None:
    page_header("Image inspection", "Upload a print photo to check for defects.")
    top_actions = st.columns([1, 5])
    if top_actions[0].button(
        "Clear inspection", icon=":material/refresh:", key="clear_image_inspection"
    ):
        st.session_state.image_result = None
        st.session_state.image_name = None
        st.session_state.image_source = None
        st.session_state.image_upload_version += 1
        st.rerun()
    upload = st.file_uploader(
        "Upload a print image",
        type=IMAGE_UPLOAD_TYPES,
        key=f"image_upload_{st.session_state.image_upload_version}",
        help="JPG, JPEG, or PNG · select or drag an image into this area.",
    )
    classifier_threshold = st.session_state.settings_classifier_confidence
    detector_threshold = st.session_state.settings_detector_confidence
    if upload is not None:
        image_bytes = upload.getvalue()
        try:
            image = decode_uploaded_image(image_bytes)
        except ValueError:
            LOGGER.exception("Uploaded image could not be decoded.")
            st.error("This image is invalid or unreadable. Select another image.")
            return

        preview, controls = st.columns([1.25, 1])
        with preview:
            st.image(
                image,
                caption=upload.name,
                use_container_width=True,
            )
        with controls:
            st.info("Any detected defect will trigger a simulated printer stop.")
        if st.button(
            "Analyze image",
            type="primary",
            icon=":material/search:",
            key="analyze_image",
        ):
            detector = get_model()
            classifier_bundle = get_classifier()
            if detector is None and classifier_bundle is None:
                st.error(
                    "Neither model is available. Check the custom checkpoint and "
                    "detector cache, then retry."
                )
                return
            errors: dict[str, str | None] = {
                "classifier": (
                    "Checkpoint unavailable or failed to load."
                    if classifier_bundle is None
                    else None
                ),
                "detector": (
                    "Detector unavailable or failed to load."
                    if detector is None
                    else None
                ),
            }
            classifier_model = None
            class_to_idx = None
            preprocessing = None
            device = None
            classifier_result = None
            detector_result = None
            with st.spinner("Running the available models on this image..."):
                if classifier_bundle is not None:
                    (
                        classifier_model,
                        class_to_idx,
                        preprocessing,
                        device,
                    ) = classifier_bundle
                    try:
                        classifier_result = predict_classifier_image(
                            image,
                            classifier_model,
                            class_to_idx,
                            preprocessing,
                            device,
                        )
                    except (RuntimeError, TypeError, ValueError, OSError, KeyError) as exc:
                        LOGGER.exception("Custom classifier image inference failed.")
                        errors["classifier"] = f"{type(exc).__name__}: {exc}"
                        st.session_state.classifier_loaded = False
                if detector is not None:
                    try:
                        detector_result = predict_detector_image(
                            image, detector, 0.0
                        )
                    except (RuntimeError, TypeError, ValueError, OSError, KeyError) as exc:
                        LOGGER.exception("YOLO image inference failed.")
                        errors["detector"] = f"{type(exc).__name__}: {exc}"
                        st.session_state.detector_loaded = False
            detector_detections = (
                detector_result["detections"] if detector_result is not None else []
            )
            decision = evaluate_combined(
                classifier_result,
                detector_detections if detector_result is not None else None,
                classifier_threshold=0.0,
                detector_threshold=0.0,
                classifier_error=errors["classifier"],
                detector_error=errors["detector"],
            )
            result = {
                "classifier": classifier_result,
                "detector": detector_result or {
                    "detections": [],
                    "qualified_detections": [],
                    "predicted_class": "YOLO unavailable",
                    "confidence": None,
                },
                "decision": decision,
                "errors": errors,
                "model_available": {
                    "classifier": classifier_bundle is not None,
                    "detector": detector is not None,
                },
                "classifier_threshold": classifier_threshold,
                "detector_threshold": detector_threshold,
            }
            record_image_result(result, upload.name, image)
            st.rerun()

    saved_result = st.session_state.image_result
    if saved_result:
        st.divider()
        st.subheader("Inspection result")
        status_col, time_col = st.columns([2, 1])
        status_col.markdown(f"**{saved_result['decision']['status']}**")
        time_col.caption(
            "Analyzed " + formatted_time(st.session_state.last_detection["timestamp"])
        )
        if (
            classifier_threshold != saved_result["classifier_threshold"]
            or detector_threshold != saved_result["detector_threshold"]
        ):
            st.info(
                "The confidence setting has changed. Analyze this image again "
                "to apply the new threshold."
            )
        st.caption(st.session_state.image_name)
        _show_prediction(saved_result, st.session_state.image_source)


def _video_result_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "Time (seconds)": round(frame["timestamp_seconds"], 2),
            "Frame": frame["frame_index"],
            "Classifier": (
                f"{frame['classifier_prediction']} "
                f"({frame['classifier_confidence']:.1%})"
                if frame["classifier_prediction"] is not None
                else "Unavailable"
            ),
            "YOLO detections": "; ".join(
                f"{item['class']} ({item['confidence']:.1%}) "
                f"box={item['bbox_xyxy']}"
                for item in frame["detections"]
            )
            or "No supported objects",
            "Combined assessment": frame["safety_status"],
            "Alert": "STOP REQUESTED" if frame.get("alert_decision") else "No",
            "Detection confidence": (
                round(frame["confidence"], 4)
                if frame["confidence"] is not None
                else ""
            ),
        }
        for frame in result["frames"]
    ]


def video_detection_page() -> None:
    page_header(
        "Video analysis",
        "Sample frames from a print recording and review defect detections over time.",
    )
    st.caption(
        "Both models analyze each sampled frame with separate preprocessing. "
        "This is uploaded-video analysis, not live camera monitoring."
    )
    top_actions = st.columns([1, 5])
    if top_actions[0].button(
        "Clear analysis", icon=":material/refresh:", key="clear_video_analysis"
    ):
        st.session_state.video_result = None
        st.session_state.video_name = None
        st.session_state.video_upload_version += 1
        st.rerun()
    upload = st.file_uploader(
        "Upload a print video",
        type=VIDEO_UPLOAD_TYPES,
        key=f"video_upload_{st.session_state.video_upload_version}",
        help="MP4, AVI, MOV, MKV, M4V, or WebM · maximum 100 MB.",
    )
    interval = st.session_state.settings_interval
    classifier_threshold = st.session_state.settings_classifier_confidence
    detector_threshold = st.session_state.settings_detector_confidence
    consecutive = st.session_state.settings_consecutive
    if upload is not None:
        video_bytes = upload.getvalue()
        if len(video_bytes) > MAX_VIDEO_BYTES:
            st.error("This video exceeds the 100 MB processing limit.")
            return
        left, right = st.columns([1.3, 1])
        with left:
            st.video(video_bytes)
        with right:
            with st.container(border=True):
                st.subheader("Analysis settings")
                st.metric("Frame interval", f"{interval:.2f} s")
                st.metric("Classifier threshold", f"{classifier_threshold:.0%}")
                st.metric("YOLO threshold", f"{detector_threshold:.0%}")
                st.metric("Consecutive frames", consecutive)
                st.caption("Adjust model thresholds from System Settings.")
        max_frames_col, batch_col = st.columns(2)
        with max_frames_col:
            max_frames = st.slider(
                "Maximum sampled frames",
                min_value=1,
                max_value=MAX_SAMPLED_FRAMES,
                value=40,
                step=1,
                key="video_max_frames",
            )
        with batch_col:
            batch_size = st.select_slider(
                "Frames per inference batch",
                options=[1, 2, 4, 8, 16],
                value=8,
                key="video_batch_size",
            )

        if st.button(
            "Analyze video",
            type="primary",
            icon=":material/analytics:",
            key="analyze_video",
        ):
            model = get_model()
            classifier_bundle = get_classifier()
            if model is None and classifier_bundle is None:
                st.error("Neither model is available. Video analysis cannot run.")
                return
            classifier_model = None
            class_to_idx = None
            preprocessing = None
            device = None
            classifier_error = (
                "Custom classifier unavailable or failed to load."
                if classifier_bundle is None
                else None
            )
            detector_error = (
                "YOLO detector unavailable or failed to load."
                if model is None
                else None
            )
            if classifier_bundle is not None:
                (
                    classifier_model,
                    class_to_idx,
                preprocessing,
                device,
                ) = classifier_bundle
            progress = st.progress(0.0, text="Preparing analysis...")
            try:
                with st.spinner("Sampling and analyzing video frames..."):
                    result = process_detector_video(
                        video_bytes,
                        model,
                        sample_interval_seconds=interval,
                        batch_size=batch_size,
                        max_sampled_frames=max_frames,
                        confidence_threshold=detector_threshold,
                        required_consecutive_frames=consecutive,
                        classifier_model=classifier_model,
                        class_to_idx=class_to_idx,
                        preprocessing=preprocessing,
                        device=device,
                        classifier_confidence_threshold=classifier_threshold,
                        detector_confidence_threshold=detector_threshold,
                        classifier_error=classifier_error,
                        detector_error=detector_error,
                        progress_callback=lambda value: progress.progress(
                            min(value, 1.0), text="Analyzing sampled frames..."
                        ),
                        filename_suffix=Path(upload.name).suffix.casefold(),
                    )
            except (OSError, TypeError, ValueError, RuntimeError):
                progress.empty()
                LOGGER.exception("Video analysis failed.")
                st.error(
                    "Video analysis could not be completed. The file may be unreadable "
                    "or use an unsupported codec."
                )
                return
            progress.progress(1.0, text="Analysis complete.")
            st.session_state.video_result = result
            st.session_state.video_name = upload.name
            if result["metadata"]["classifier_error"]:
                st.session_state.classifier_loaded = False
            if result["metadata"]["detector_error"]:
                st.session_state.detector_loaded = False
            confirmed_detections = [
                detection
                for frame in result["frames"]
                for detection in frame["assessment"]["detector_qualified"]
            ]
            has_low_confidence = any(
                frame["safety_status"] == "REVIEW REQUIRED"
                for frame in result["frames"]
            )
            has_frame_alert = any(
                frame["assessment"]["alert"] for frame in result["frames"]
            )
            record_inspection(
                "Video",
                upload.name,
                confirmed_detections,
                "DEFECT DETECTED"
                if result["events"]
                else "POTENTIAL DEFECT — FRAME RULE NOT MET"
                if has_frame_alert
                else "UNCERTAIN — REVIEW REQUIRED"
                if has_low_confidence
                else "NO HIGH-CONFIDENCE DEFECT SIGNAL",
                next(
                    (
                        frame["classifier_result"]
                        for frame in result["frames"]
                        if frame["classifier_result"] is not None
                    ),
                    None,
                ),
                next(
                    (
                        frame["assessment"]
                        for frame in result["frames"]
                        if frame["assessment"]["alert"]
                    ),
                    None,
                ),
            )

            for event in result["events"]:
                event_frames = [
                    frame
                    for frame in result["frames"]
                    if event["start_seconds"]
                    <= frame["timestamp_seconds"]
                    <= event["end_seconds"]
                    and any(
                        item["class"].casefold()
                        == event["predicted_defect"].casefold()
                        for item in frame["assessment"]["alert_defects"]
                    )
                ]
                event_classifier = next(
                    (
                        frame["classifier_result"]
                        for frame in event_frames
                        if frame["classifier_result"] is not None
                    ),
                    None,
                )
                event_detections = [
                    detection
                    for frame in event_frames
                    for detection in frame["detections"]
                ]
                model_confidences = {
                    "Custom classifier": event["classifier_confidence"],
                    "YOLO detector": event["yolo_confidence"],
                }
                add_incident(
                    source=f"Video: {upload.name} · {event['source']}",
                    predicted_class=event["predicted_defect"],
                    confidence=event["confidence"],
                    action="ALERT REQUESTED · SIMULATED",
                    reason=(
                        f"Consecutive-frame rule met for {event['predicted_defect']} "
                        f"using {event['source']} evidence, "
                        f"{event['start_seconds']:.2f}–{event['end_seconds']:.2f}s."
                    ),
                    timestamp_in_video=event["start_seconds"],
                    classifier_result=event_classifier,
                    yolo_detections=event_detections,
                    combined_decision=(
                        event_frames[0]["assessment"] if event_frames else None
                    ),
                    model_confidences=model_confidences,
                )
            uncertain_frames = [
                frame
                for frame in result["frames"]
                if frame["assessment"]["manual_review_recommended"]
                and not frame["assessment"]["alert"]
            ]
            if uncertain_frames:
                request_warning(
                    st.session_state.printer_state,
                    "Low-confidence detections require review.",
                )
            if result["events"]:
                st.rerun()

    result = st.session_state.video_result
    if result:
        metadata = result["metadata"]
        st.divider()
        st.subheader(f"Analysis result · {st.session_state.video_name}")
        if (
            classifier_threshold != metadata["classifier_confidence_threshold"]
            or detector_threshold != metadata["detector_confidence_threshold"]
            or consecutive != metadata["required_consecutive_frames"]
            or interval != metadata["sample_interval_seconds"]
        ):
            st.info(
                "Analysis settings have changed. Reanalyze the video to apply "
                "the current values."
            )
        st.caption(
            f"{metadata['width']} × {metadata['height']} · "
            f"{metadata['fps']:.2f} fps · {metadata['duration_seconds']:.1f} sec"
        )
        if metadata["classifier_error"]:
            st.warning("Custom classifier inference failed; YOLO results remain available.")
        if metadata["detector_error"]:
            st.warning("YOLO inference failed; custom classifier results remain available.")
        frames = result["frames"]
        defect_count = sum(
            len(frame["assessment"]["alert_defects"]) for frame in frames
        )
        detected_frames = sum(
            frame["assessment"]["alert"] for frame in frames
        )
        uncertain_count = sum(
            frame["assessment"]["manual_review_recommended"]
            for frame in frames
        )
        cards = st.columns(4)
        cards[0].metric("Frames analyzed", metadata["sampled_frames"])
        cards[1].metric("Frames with defect evidence", detected_frames)
        cards[2].metric("Qualified model signals", defect_count)
        cards[3].metric("Confirmed events", len(result["events"]))

        timeline_rows = [
            {
                "Time (sec)": frame["timestamp_seconds"],
                "Custom classifier": frame["classifier_confidence"],
                "YOLO detector": frame["confidence"],
            }
            for frame in frames
        ]
        timeline_data = pd.DataFrame(timeline_rows)
        if not timeline_data.empty:
            st.subheader("Detection timeline")
            st.line_chart(
                timeline_data.set_index("Time (sec)"),
                color=["#38bdf8", "#a78bfa"],
            )
            st.dataframe(
                _video_result_rows(result),
                hide_index=True,
                use_container_width=True,
            )
        if result["events"]:
            st.subheader("Detected events")
            st.dataframe(
                [
                    {
                        "Start": f"{event['start_seconds']:.2f} sec",
                        "End": f"{event['end_seconds']:.2f} sec",
                        "Defect": event["predicted_defect"],
                        "Sources": event["source"],
                        "Classifier confidence": (
                            f"{event['classifier_confidence']:.1%}"
                            if event["classifier_confidence"] is not None
                            else "—"
                        ),
                        "YOLO confidence": (
                            f"{event['yolo_confidence']:.1%}"
                            if event["yolo_confidence"] is not None
                            else "—"
                        ),
                        "Frames": event["frames"],
                        "Decision": "Alert requested · simulated",
                    }
                    for event in result["events"]
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.warning(
                "A simulated alert was recorded. No physical printer action was sent."
            )
        elif not defect_count:
            st.info(
                "No high-confidence defect signal met the consecutive-frame settings. "
                "This is not proof that the print is defect-free."
            )
        if uncertain_count:
            st.caption(
                f"{uncertain_count} sampled frame(s) have uncertain or incomplete "
                "model results and may need review."
            )
        with st.expander("Sampled frame previews", expanded=False):
            st.image(
                [frame["thumbnail"] for frame in frames],
                caption=[
                    (
                        f"{frame['timestamp_seconds']:.2f}s · "
                        f"{frame['assessment']['status']} · "
                        f"classifier {frame['classifier_confidence']:.1%} · "
                        f"{len(frame['detections'])} YOLO object(s)"
                        if frame["classifier_confidence"] is not None
                        else f"{frame['timestamp_seconds']:.2f}s · "
                        f"{frame['assessment']['status']} · "
                        f"{len(frame['detections'])} YOLO object(s)"
                    )
                    for frame in frames
                ],
                width=240,
            )
        downloads = st.columns(2)
        downloads[0].download_button(
            "Download frame report",
            data=frames_to_csv(frames),
            file_name="printguard_video_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )
        event_csv = io.StringIO(newline="")
        event_fields = (
            "start_seconds",
            "end_seconds",
            "predicted_defect",
            "source",
            "confidence",
            "classifier_confidence",
            "yolo_confidence",
            "frames",
            "action",
        )
        event_writer = csv.DictWriter(event_csv, fieldnames=event_fields)
        event_writer.writeheader()
        event_writer.writerows(
            {field: event.get(field, "") for field in event_fields}
            for event in result["events"]
        )
        downloads[1].download_button(
            "Download event report",
            data=event_csv.getvalue().encode("utf-8"),
            file_name="printguard_detected_events.csv",
            mime="text/csv",
            use_container_width=True,
        )


@st.fragment(run_every="2s")
def telemetry_simulator_page() -> None:
    page_header(
        "Printer telemetry",
        "Review the current simulated print state and manage the local alert workflow.",
    )
    state = st.session_state.printer_state

    with st.container(horizontal=True, vertical_alignment="bottom"):
        if st.button(
            "Start print simulation",
            disabled=state["status"]
            in {"PRINTING", "WARNING", "STOP REQUESTED"},
            type="primary",
            key="start_print",
        ):
            start_print(state)
        if st.button("Reset print simulation", key="reset_print"):
            reset_print(state)
        if st.button(
            "Acknowledge stop",
            disabled=state["status"] != "STOP REQUESTED",
            key="acknowledge_stop",
        ):
            acknowledge_stop(state)
        if st.button(
            "Acknowledge warning",
            disabled=state["status"] != "WARNING",
            key="acknowledge_warning",
        ):
            acknowledge_warning(state)

    snapshot = telemetry_snapshot(state)
    st.caption("Simulated telemetry · values are not measured from a connected printer.")
    metrics = st.columns(3)
    metrics[0].metric("Printer state", snapshot["printer_status"])
    metrics[1].metric("Nozzle temperature", f"{snapshot['nozzle_c']:.1f} °C")
    metrics[2].metric("Bed temperature", f"{snapshot['bed_c']:.1f} °C")
    metrics[0].metric("Print speed", f"{snapshot['speed_mm_s']:.1f} mm/s")
    metrics[1].metric("Estimated remaining", f"{snapshot['remaining_minutes']} min")
    metrics[2].metric("Extruder", snapshot["extruder_status"])
    st.progress(
        snapshot["progress"] / 100,
        text=f"Print progress · {snapshot['progress']:.1f}%",
    )

    if state["status"] in {"STOP REQUESTED", "STOPPED (SIMULATED)"}:
        st.error(
            f"Simulated alert state · {state.get('stop_reason', '')}",
            icon=":material/warning:",
        )
    elif state["status"] == "WARNING":
        st.warning(
            f"Review recommended · {state.get('warning_reason', '')}",
            icon=":material/warning:",
        )


def incident_log_page() -> None:
    page_header(
        "Incident history",
        "Review confidence-qualified events that raised a simulated alert.",
    )
    incidents = st.session_state.incident_log
    if not incidents:
        st.info("No incidents have been recorded in this session.")
        return
    filters = st.columns(2)
    sources = ["All sources"] + sorted({item["source"].split(":")[0] for item in incidents})
    defects = ["All defects"] + sorted({item["predicted_defect"] for item in incidents})
    source_filter = filters[0].selectbox("Source", sources, key="incident_source_filter")
    defect_filter = filters[1].selectbox("Defect", defects, key="incident_defect_filter")
    filtered = [
        item
        for item in incidents
        if (source_filter == "All sources" or item["source"].startswith(source_filter))
        and (defect_filter == "All defects" or item["predicted_defect"] == defect_filter)
    ]
    st.dataframe(
        [
            {
                "Timestamp": formatted_time(item["timestamp"]),
                "Source": item["source"],
                "Defect": item["predicted_defect"],
                "Classifier confidence": (
                    f"{item['model_confidences'].get('Custom classifier'):.1%}"
                    if item.get("model_confidences", {}).get("Custom classifier")
                    is not None
                    else "—"
                ),
                "YOLO confidence": (
                    f"{item['model_confidences'].get('YOLO detector'):.1%}"
                    if item.get("model_confidences", {}).get("YOLO detector")
                    is not None
                    else "—"
                ),
                "Decision": "Alert requested · simulated",
                "Reason": item["reason"],
            }
            for item in reversed(filtered)
        ],
        hide_index=True,
        use_container_width=True,
    )
    csv_buffer = io.StringIO(newline="")
    fields = (
        "timestamp",
        "source",
        "predicted_defect",
        "confidence",
        "model_confidences",
        "classifier_result",
        "yolo_detections",
        "combined_decision",
        "action",
        "reason",
        "video_timestamp_seconds",
    )
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(
        {
            field: (
                json.dumps(event.get(field), ensure_ascii=False)
                if isinstance(event.get(field), (dict, list))
                else event.get(field, "")
            )
            for field in fields
        }
        for event in filtered
    )
    st.download_button(
        "Download filtered history",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="printguard_incident_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if st.button("Acknowledge alert", key="incident_acknowledge"):
        acknowledge_stop(st.session_state.printer_state)
        st.rerun()


def system_settings_page() -> None:
    page_header(
        "System settings",
        "Tune inspection sensitivity and review detection capabilities.",
    )
    settings_left, settings_right = st.columns([1.2, 1])
    with settings_left:
        with st.container(border=True):
            st.subheader("Detection preferences")
            st.slider(
                "Custom classifier confidence threshold",
                min_value=0.50,
                max_value=0.99,
                step=0.01,
                key="settings_classifier_confidence",
                help="Classifier predictions below this threshold require review and do not independently raise an alert.",
            )
            st.slider(
                "YOLO confidence threshold",
                min_value=0.50,
                max_value=0.99,
                step=0.01,
                key="settings_detector_confidence",
                help="YOLO detections below this threshold are shown for review and do not independently raise an alert.",
            )
            st.slider(
                "Video frame-sampling interval",
                min_value=0.25,
                max_value=10.0,
                step=0.25,
                key="settings_interval",
                help="Analyze one sampled frame for each interval.",
            )
            st.slider(
                "Consecutive detections for video alert",
                min_value=1,
                max_value=10,
                step=1,
                key="settings_consecutive",
                help="Require repeated same-class detections across sampled frames before logging an alert.",
            )
    with settings_right:
        with st.container(border=True):
            st.subheader("Custom classifier")
            if st.session_state.classifier_loaded:
                st.success("Available · loaded for this session")
            elif st.session_state.classifier_load_error:
                st.error("Unavailable · checkpoint could not be loaded")
            else:
                st.info("Not loaded · availability is unverified")
            st.caption("Labels · " + " · ".join(CLASS_NAMES))
            if st.button(
                "Load custom classifier",
                icon=":material/model_training:",
                key="load_classifier_settings",
                use_container_width=True,
            ):
                if get_classifier() is not None:
                    st.success("Custom classifier loaded and ready.")
            st.subheader("YOLO detector")
            if st.session_state.detector_loaded:
                st.success("Available · loaded for this session")
            elif st.session_state.model_load_error:
                st.error("Unavailable · model could not be loaded")
            else:
                st.info("Not loaded · availability is unverified")
            st.caption(
                "Class IDs · "
                + " · ".join(
                    f"{class_id}: {label}"
                    for class_id, label in DETECTOR_CLASS_MAPPING.items()
                )
            )
            if st.button(
                "Load YOLO detector",
                icon=":material/download:",
                key="load_detector_settings",
                use_container_width=True,
            ):
                if get_model() is not None:
                    st.success("YOLO detector loaded and ready.")
            model_download_message()
        with st.container(border=True):
            st.subheader("Telemetry source")
            st.caption(
                "Printer values and alert actions are generated by the local "
                "simulator. No hardware connection is configured."
            )


initialize_state()
with st.sidebar:
    st.markdown(
        '<div class="pg-brand">PrintGuard</div>'
        '<div class="pg-subtitle">AI-Powered 3D Print Quality Monitoring</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Workspace")
    selected_page = st.radio(
        "Application sections",
        APP_PAGES,
        key="app_page",
        label_visibility="collapsed",
        format_func=lambda page: {
            "Overview": "◉  Overview",
            "Image Inspection": "▣  Image Inspection",
            "Video Analysis": "▶  Video Analysis",
            "Printer Telemetry": "◈  Printer Telemetry",
            "Incident History": "≡  Incident History",
            "System Settings": "⚙  System Settings",
        }[page],
    )
    st.divider()
    st.caption(
        "Classifier · "
        + ("Loaded" if st.session_state.classifier_loaded else "Not loaded")
    )
    st.caption(
        "YOLO detector · "
        + ("Loaded" if st.session_state.detector_loaded else "Not loaded")
    )
    st.caption("Telemetry source · Simulated")

if selected_page == "Image Inspection":
    image_detection_page()
elif selected_page == "Video Analysis":
    video_detection_page()
elif selected_page == "Printer Telemetry":
    telemetry_simulator_page()
elif selected_page == "Incident History":
    incident_log_page()
elif selected_page == "System Settings":
    system_settings_page()
else:
    overview_page()
