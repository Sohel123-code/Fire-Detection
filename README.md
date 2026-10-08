# FireWatch — Fire & Smoke Detection

A Streamlit app using **best_new.pt** for continuous browser camera predictions and uploaded video analysis. Both the app and `detect.py` require these weights; neither silently falls back to `best.pt`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. The **Live camera** workspace requests camera access automatically. Allow access, select your camera if needed, and leave **Camera on** enabled. Turn it off to stop. Audio is disabled.

- Predictions run continuously in a WebRTC worker; slow inference drops queued older frames to avoid growing delay.
- Boxes and labels appear on the returned video. Current fire/smoke counts, prediction FPS, frame count, and inference time update beneath it.
- Changing confidence applies to the running processor. Errors and stalled predictions are displayed explicitly.
- Shared model access is serialized to protect inference across sessions. CPU speed and concurrent users affect prediction FPS.
- **Analyze video** processes uploaded MP4, AVI, MOV, or MKV files and exports H.264 MP4 without audio. Counts include repeated detections across frames.

## Hosted camera access

Camera capture requires **HTTPS** on a hosted app, or localhost during development. Deployment must include `best_new.pt`, `requirements.txt`, and `packages.txt`. On Streamlit Community Cloud, select this repository, branch `master`, and entrypoint `app.py`.

STUN is configured by default. Some cloud/firewall/mobile networks also require a TURN relay. Configure your provider's ICE servers in Streamlit's deployment secrets or local `.streamlit/secrets.toml` (ignored by Git):

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
```

Tests cover repeated processing, latest-frame behavior, confidence changes, class mapping, error recovery, and UI states. UI tests mock the WebRTC component because Streamlit's AppTest does not provide a real browser session. An end-to-end deployment check still requires allowing a physical camera and confirming the frame count increases, confidence updates, and camera stop/restart work on the target network.

## Files

- `app.py`: camera dashboard and video analysis
- `detection.py`: required model loading, serialized inference, live processor and statistics
- `best_new.pt`: active trained weights (smoke and fire)
- `detect.py`: standalone local video detection script
- `tests/`: processor and Streamlit UI regression tests

## License

MIT
