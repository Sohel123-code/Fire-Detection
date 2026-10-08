"""Test the UI states; WebRTC transport itself requires a real browser."""
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock, patch
from streamlit.testing.v1 import AppTest
from detection import LiveStats


class AppTests(unittest.TestCase):
    def render(self, playing=False, stats=None):
        processor = Mock()
        processor.snapshot.return_value = stats or LiveStats()
        context = SimpleNamespace(state=SimpleNamespace(playing=playing), video_processor=processor)
        with patch("streamlit_webrtc.webrtc_streamer", return_value=context):
            app = AppTest.from_file("app.py").run(timeout=60)
        self.assertFalse(app.exception)
        return app, context

    def test_waiting_and_off_do_not_claim_live_predictions(self):
        app, context = self.render()
        self.assertIn("Waiting for camera", app.info[0].value)
        with patch("streamlit_webrtc.webrtc_streamer", return_value=context):
            app.toggle[0].set_value(False).run()
        self.assertFalse(app.exception)
        self.assertIn("Camera off", app.info[0].value)

    def test_fresh_detection_alert_and_counts(self):
        app, _ = self.render(True, LiveStats(frames=10, fire=2, smoke=1, last_prediction=time.monotonic()))
        self.assertIn("Detection alert", app.error[0].value)
        self.assertEqual(app.metric[0].value, "2")

    def test_stale_predictions_are_not_shown_as_clear(self):
        app, _ = self.render(True, LiveStats(frames=10, last_prediction=time.monotonic() - 60))
        self.assertIn("waiting for predictions", app.warning[0].value)
        self.assertEqual(app.metric[0].value, "—")

    def test_prediction_error_and_upload_navigation(self):
        app, _ = self.render(True, LiveStats(error="test failure"))
        self.assertIn("test failure", app.error[0].value)
        app.radio[0].set_value("Analyze video").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.subheader[0].value, "Analyze a video")


if __name__ == "__main__":
    unittest.main()
