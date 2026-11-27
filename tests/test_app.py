from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_dashboard_smoke_and_navigation() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception
    assert app.session_state["app_page"] == "Overview"
    app.button(key="quick_image_inspection").click().run()
    assert not app.exception
    assert app.session_state["app_page"] == "Image Inspection"

    app.radio(key="app_page").set_value("Video Analysis").run()
    assert not app.exception
    assert app.session_state["app_page"] == "Video Analysis"

    app.radio(key="app_page").set_value("System Settings").run()
    assert not app.exception
    app.slider(key="settings_classifier_confidence").set_value(0.82).run()
    assert not app.exception
    assert app.session_state["settings_classifier_confidence"] == 0.82
    app.slider(key="settings_detector_confidence").set_value(0.77).run()
    assert not app.exception
    assert app.session_state["settings_detector_confidence"] == 0.77

    app.radio(key="app_page").set_value("Image Inspection").run()
    assert not app.exception
    app.button(key="clear_image_inspection").click().run()
    assert not app.exception
    assert app.session_state["image_result"] is None

    app.radio(key="app_page").set_value("Printer Telemetry").run()
    assert not app.exception
    app.button(key="start_print").click().run()
    assert not app.exception
    assert app.session_state["printer_state"]["status"] == "PRINTING"

    app.radio(key="app_page").set_value("Incident History").run()
    assert not app.exception
