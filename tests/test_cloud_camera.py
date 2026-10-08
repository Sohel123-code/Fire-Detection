import base64
from unittest.mock import Mock, patch
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest
from cloud_camera import predict_frame


class CloudCameraTests(unittest.TestCase):
    def test_real_jpeg_roundtrip_and_confidence(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', image)
        detector = Mock()
        detector.predict.return_value = (image, {"fire": 1, "smoke": 2})
        event = {"id": "session:1", "image": "data:image/jpeg;base64," + base64.b64encode(encoded).decode()}
        result = predict_frame(event, detector, .75)
        self.assertEqual(result['id'], event['id'])
        self.assertFalse(result['error'])
        self.assertEqual((result['fire'], result['smoke']), (1, 2))
        self.assertEqual(detector.predict.call_args.args[1], .75)
        self.assertTrue(result['image'].startswith('data:image/jpeg;base64,'))

    def test_failed_frame_gets_error_acknowledgement(self):
        detector = Mock()
        for image in ['bad', 'data:image/jpeg;base64,!!!!', 'data:image/jpeg;base64,' + 'a' * 1_000_000]:
            result = predict_frame({"id": "x:2", "image": image}, detector, .4)
            self.assertEqual(result['id'], 'x:2')
            self.assertTrue(result['error'])
        detector.predict.assert_not_called()

    def test_cloud_camera_is_default_and_loads_without_webrtc(self):
        detector = Mock(model=SimpleNamespace(names={0: "smoke", 1: "fire"}))
        with patch('detection.Detector', return_value=detector), patch('cloud_camera.declare_component', return_value=Mock(return_value=None)):
            app = AppTest.from_file('app.py').run(timeout=60)
        self.assertFalse(app.exception)
        connection = next(r for r in app.radio if r.label == 'Camera connection')
        self.assertEqual(connection.value, 'Cloud camera')
        self.assertIn('Start the camera', app.info[0].value)


if __name__ == '__main__':
    unittest.main()
