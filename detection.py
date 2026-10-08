"""Shared inference and thread-safe live camera statistics."""
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
import logging
import time
import av
import cv2
from streamlit_webrtc import VideoProcessorBase
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "best_new.pt"
logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Required model weights are missing: {model_path}")
        self.model = YOLO(str(model_path))
        self.lock = Lock()

    def predict(self, image, confidence=0.4):
        # The cached model is shared by camera sessions and uploaded videos.
        with self.lock:
            result = self.model.predict(image, conf=confidence, imgsz=640, verbose=False)[0]
            annotated = result.plot()
            counts = {"fire": 0, "smoke": 0}
            for class_id in result.boxes.cls.tolist():
                name = result.names[int(class_id)].lower()
                for label in counts:
                    if label in name:
                        counts[label] += 1
            return annotated, counts


@dataclass
class LiveStats:
    frames: int = 0
    fire: int = 0
    smoke: int = 0
    fps: float = 0.0
    latency_ms: float = 0.0
    last_prediction: float = 0.0
    error: str = ""


class LiveProcessor(VideoProcessorBase):
    def __init__(self, detector, confidence=0.4):
        self.detector = detector
        self.lock = Lock()
        self.confidence = confidence
        self.stats = LiveStats()
        self.started = None

    def set_confidence(self, confidence):
        with self.lock:
            self.confidence = confidence

    def snapshot(self):
        with self.lock:
            return replace(self.stats)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        started = time.monotonic()
        with self.lock:
            confidence = self.confidence
        try:
            annotated, counts = self.detector.predict(image, confidence)
            finished = time.monotonic()
            with self.lock:
                if self.started is None:
                    self.started = started
                self.stats.frames += 1
                self.stats.fire = counts["fire"]
                self.stats.smoke = counts["smoke"]
                self.stats.latency_ms = (finished - started) * 1000
                self.stats.fps = self.stats.frames / max(finished - self.started, 0.001)
                self.stats.last_prediction = finished
                self.stats.error = ""
            label = f"LIVE | Fire: {counts['fire']} | Smoke: {counts['smoke']}"
            color = (60, 60, 235) if any(counts.values()) else (100, 210, 90)
        except Exception as exc:
            logger.exception("Live inference failed")
            with self.lock:
                self.stats.error = str(exc)
                self.stats.fire = self.stats.smoke = 0
            annotated = image.copy()
            label, color = "INFERENCE ERROR - see status panel", (60, 60, 235)
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (24, 30, 42), -1)
        cv2.putText(annotated, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
        output = av.VideoFrame.from_ndarray(annotated, format="bgr24")
        output.pts, output.time_base = frame.pts, frame.time_base
        return output
