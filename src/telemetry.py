from __future__ import annotations

import math
import time
from typing import Any


PRINT_DURATION_SECONDS = 300


def initial_state() -> dict[str, Any]:
    return {
        "status": "IDLE",
        "started_at": None,
        "progress": 0.0,
        "stop_reason": "",
    }


def start_print(state: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    state.update(status="PRINTING", started_at=now, progress=0.0, stop_reason="")
    state.pop("status_before_warning", None)
    state.pop("warning_reason", None)
    return state


def request_stop(state: dict[str, Any], reason: str) -> dict[str, Any]:
    if state.get("status") not in {"STOP REQUESTED", "STOPPED (SIMULATED)"}:
        state["status"] = "STOP REQUESTED"
        state["stop_reason"] = reason
        state.pop("status_before_warning", None)
        state.pop("warning_reason", None)
    return state


def request_warning(state: dict[str, Any], reason: str) -> dict[str, Any]:
    if state.get("status") in {"STOP REQUESTED", "STOPPED (SIMULATED)"}:
        return state
    if state.get("status") != "WARNING":
        state["status_before_warning"] = state.get("status", "IDLE")
    state["status"] = "WARNING"
    state["warning_reason"] = reason
    return state


def acknowledge_warning(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("status") == "WARNING":
        state["status"] = state.pop("status_before_warning", "IDLE")
        state.pop("warning_reason", None)
    return state


def acknowledge_stop(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("status") == "STOP REQUESTED":
        state["status"] = "STOPPED (SIMULATED)"
    return state


def reset_print(state: dict[str, Any]) -> dict[str, Any]:
    state.clear()
    state.update(initial_state())
    return state


def telemetry_snapshot(
    state: dict[str, Any], now: float | None = None
) -> dict[str, float | int | str | bool]:
    now = time.time() if now is None else now
    started_at = state.get("started_at")
    status = state.get("status", "IDLE")
    underlying_status = (
        state.get("status_before_warning", "IDLE")
        if status == "WARNING"
        else status
    )
    active = underlying_status == "PRINTING" and started_at is not None
    elapsed = max(0.0, now - started_at) if active else 0.0
    progress = min(100.0, elapsed / PRINT_DURATION_SECONDS * 100) if active else float(
        state.get("progress", 0.0)
    )
    if active and progress >= 100:
        if status == "WARNING":
            state["status_before_warning"] = "IDLE"
        else:
            state["status"] = "IDLE"
        state.update(started_at=None, progress=100.0)
        active = False
    elif active:
        state["progress"] = progress

    phase = min(1.0, progress / 100)
    nozzle = 24.0 + 196.0 * min(1.0, phase * 7 + 0.15)
    bed = 23.0 + 42.0 * min(1.0, phase * 5 + 0.12)
    speed = 0.0 if not active else 45.0 + 5.0 * math.sin(elapsed / 11)
    remaining = max(0, round((100 - progress) / 100 * PRINT_DURATION_SECONDS / 60))

    return {
        "nozzle_c": round(nozzle, 1),
        "bed_c": round(bed, 1),
        "progress": round(progress, 1),
        "speed_mm_s": round(speed, 1),
        "extruder_status": "ACTIVE" if active else "IDLE",
        "remaining_minutes": int(remaining),
        "printer_status": str(state.get("status", "IDLE")),
        "simulated": True,
    }
