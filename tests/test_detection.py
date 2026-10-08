import asyncio
from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import av
import numpy as np

from detection import Detector, LiveProcessor


def frame(value=0):
    result = av.VideoFrame.from_ndarray(np.full((120, 160, 3), value, dtype=np.uint8), format="bgr24")
    result.pts = value
    result.time_base = Fraction(1, 30)
    return result


class LiveDetectionTests(unittest.TestCase):
    def test_continuous_frames_confidence_update_and_timestamps(self):
        detector = Mock()
        detector.predict.side_effect = lambda image, conf: (image.copy(), {"fire": 1, "smoke": 2})
        processor = LiveProcessor(detector)
        for index in range(20):
            result = processor.recv(frame(index))
            self.assertEqual(result.pts, index)
            self.assertEqual(result.time_base, Fraction(1, 30))
        processor.set_confidence(0.75)
        processor.recv(frame())
        self.assertEqual(detector.predict.call_args.args[1], 0.75)
        stats = processor.snapshot()
        self.assertEqual(stats.frames, 21)
        self.assertEqual((stats.fire, stats.smoke), (1, 2))
        self.assertGreater(stats.fps, 0)
        stats.frames = -1
        self.assertEqual(processor.snapshot().frames, 21)

    def test_latest_frame_wins_when_inference_falls_behind(self):
        detector = Mock()
        detector.predict.side_effect = lambda image, conf: (image.copy(), {"fire": 0, "smoke": 0})
        processor = LiveProcessor(detector)
        results = asyncio.run(processor.recv_queued([frame(1), frame(2), frame(3)]))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].pts, 3)
        self.assertEqual(detector.predict.call_count, 1)

    def test_error_is_visible_and_next_frame_recovers(self):
        detector = Mock()
        detector.predict.side_effect = RuntimeError("inference unavailable")
        processor = LiveProcessor(detector)
        with self.assertLogs("detection", level="ERROR"):
            output = processor.recv(frame())
        self.assertIsInstance(output, av.VideoFrame)
        self.assertEqual(processor.snapshot().error, "inference unavailable")
        self.assertEqual(processor.snapshot().frames, 0)
        detector.predict.side_effect = lambda image, conf: (image.copy(), {"fire": 0, "smoke": 0})
        processor.recv(frame())
        self.assertEqual(processor.snapshot().error, "")
        self.assertEqual(processor.snapshot().frames, 1)

    def test_classes_follow_model_names_not_assumed_indices(self):
        result = SimpleNamespace(
            boxes=SimpleNamespace(cls=np.array([0, 0, 1])),
            names={0: "smoke", 1: "fire"},
            plot=lambda: np.zeros((120, 160, 3), dtype=np.uint8),
        )
        with patch("detection.YOLO") as yolo:
            yolo.return_value.predict.return_value = [result]
            detector = Detector()
            _, counts = detector.predict(np.zeros((120, 160, 3), dtype=np.uint8))
        self.assertEqual(counts, {"fire": 1, "smoke": 2})

    def test_missing_new_weights_never_falls_back(self):
        with self.assertRaises(FileNotFoundError):
            Detector("missing-best-new.pt")


if __name__ == "__main__":
    unittest.main()
