# FireWatch — Fire & Smoke Detection

A Streamlit app using **best_new.pt** for continuous browser camera predictions and uploaded video analysis. Both the app and `detect.py` require these weights; neither silently falls back to `best.pt`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. In **Live camera → Cloud camera**, click **Start camera** and allow camera access. The preview stays live and predictions update continuously beside it. Click **Stop camera** to release the camera. Audio is disabled.

- Cloud camera sends JPEG frames through the existing Streamlit connection. It works without a separate WebRTC connection or TURN relay. Only one frame is in flight, so slow inference cannot build a queue.
- Boxes and labels appear in **Latest prediction**, next to the smooth live preview. Current fire/smoke counts, frame count, and inference time update beneath it. Prediction frequency depends on server speed and network latency.
- **WebRTC (advanced)** remains available for deployments with working ICE/TURN connectivity, with annotations directly on the returned video and prediction FPS statistics.
- Changing confidence applies to the running processor. Errors and stalled predictions are displayed explicitly.
- Shared model access is serialized to protect inference across sessions. CPU speed and concurrent users affect prediction FPS.
- **Analyze video** processes uploaded MP4, AVI, MOV, or MKV files and exports H.264 MP4 without audio. Counts include repeated detections across frames.

## Hosted camera access

Camera capture requires **HTTPS** on a hosted app, or localhost during development. Deployment must include `best_new.pt`, `requirements.txt`, and `packages.txt`. On Streamlit Community Cloud, select this repository, branch `master`, and entrypoint `app.py`.

The default **Cloud camera** mode requires no relay credentials. If you choose **WebRTC (advanced)**, some cloud/firewall/mobile networks require a TURN relay. That mode discovers supported provider environment credentials automatically, or you can configure ICE servers in Streamlit's deployment secrets or local `.streamlit/secrets.toml` (ignored by Git):

```toml
[[webrtc.ice_servers]]
urls = ["stun:stun.l.google.com:19302"]

[[webrtc.ice_servers]]
urls = ["turn:YOUR_TURN_HOST:3478", "turns:YOUR_TURN_HOST:5349"]
username = "YOUR_TURN_USERNAME"
credential = "YOUR_TURN_CREDENTIAL"
```

Use valid credentials from your relay provider. ICE credentials are delivered to the browser as part of WebRTC negotiation; use scoped, short-lived credentials where available. See the [streamlit-webrtc deployment guide](https://whitphx.github.io/streamlit-webrtc/deployment/).

## Validation

```bash
python -m unittest discover -s tests -v
node --test --test-isolation=none tests/camera_frontend.test.cjs
```

Tests cover repeated processing, JPEG transport, latest-frame behavior, confidence changes, class mapping, error recovery, and UI states. JavaScript tests exercise continuous frame acknowledgements, permission denial, stop during startup, and track cleanup. UI tests mock browser component mounting because AppTest does not provide a real browser session. An end-to-end deployment check still requires allowing a physical camera and confirming the frame count increases, confidence updates, and camera stop/restart work on the target network.

## Files

- `app.py`: camera dashboard and video analysis
- `detection.py`: required model loading, serialized inference, live processor and statistics
- `cloud_camera.py`, `camera_frontend/`: browser camera and continuous prediction over the Streamlit connection
- `best_new.pt`: active trained weights (smoke and fire)
- `detect.py`: standalone local video detection script
- `tests/`: processor and Streamlit UI regression tests

## License

MIT
