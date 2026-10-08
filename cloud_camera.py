"""Continuous camera frames over Streamlit's existing connection, without ICE/TURN."""
import base64
from pathlib import Path
import time

import cv2
import numpy as np
import streamlit as st
from streamlit.components.v1 import declare_component


def predict_frame(event, detector, confidence):
    """Return a reply even on inference failure so the browser can recover."""
    reply = {"id": event.get("id"), "error": ""}
    try:
        data = event.get("image", "")
        if not isinstance(data, str) or not data.startswith("data:image/jpeg;base64,") or len(data) > 1_000_000:
            raise ValueError("Invalid camera frame")
        raw = base64.b64decode(data.split(",", 1)[1], validate=True)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or max(image.shape[:2]) > 1920:
            raise ValueError("Unreadable camera frame")
        started = time.monotonic()
        annotated, counts = detector.predict(image, confidence)
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("Could not encode prediction")
        reply.update(image="data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
                     fire=counts["fire"], smoke=counts["smoke"],
                     latency_ms=(time.monotonic() - started) * 1000)
    except Exception as exc:
        reply["error"] = str(exc)
    return reply


@st.fragment(run_every=0.3)
def render_cloud_camera(detector, confidence):
    # Register inside the script context, including after test/import-only contexts.
    component = declare_component("firewatch_camera", path=str(Path(__file__).parent / "camera_frontend"))
    state = st.session_state.setdefault("cloud_camera_state", {
        "session": None, "reply": None, "frames": 0, "last_prediction": 0,
    })
    event = component(reply=state["reply"], key="cloud-camera", default=None)
    if isinstance(event, dict):
        if event.get("session") != state["session"]:
            state.update(session=event.get("session"), reply=None, frames=0, last_prediction=0)
        if event.get("kind") == "frame" and event.get("id"):
            if not state["reply"] or event["id"] != state["reply"]["id"]:
                state["reply"] = predict_frame(event, detector, confidence)
                if not state["reply"]["error"]:
                    state["frames"] += 1
                    state["last_prediction"] = time.monotonic()
        elif event.get("kind") in ("stopped", "error"):
            state.update(reply=None, frames=0, last_prediction=0)
    reply = state["reply"]
    fresh = bool(reply and not reply["error"] and time.monotonic() - state["last_prediction"] < 5)
    if event and event.get("kind") == "error":
        st.error(event.get("message", "Camera unavailable"))
    elif reply and reply["error"]:
        st.error(f"Prediction failed: {reply['error']}")
    elif fresh and (reply["fire"] or reply["smoke"]):
        st.error(f"Detection alert · {reply['fire']} fire / {reply['smoke']} smoke in the latest frame")
    elif fresh:
        st.success("Live · predicting continuously · no fire or smoke detected in the latest frame")
    else:
        st.info("Start the camera above and allow browser access. Predictions appear beside the live preview.")
    columns = st.columns(4)
    values = [reply["fire"] if fresh else "—", reply["smoke"] if fresh else "—",
              state["frames"], f"{reply['latency_ms']:.0f} ms" if fresh else "—"]
    for column, label, value in zip(columns, ["Fire in frame", "Smoke in frame", "Frames analyzed", "Inference time"], values):
        column.metric(label, value)
