"""FireWatch: continuous browser camera inference and video analysis."""
from fractions import Fraction
from pathlib import Path
import hashlib
import tempfile
import time
import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from detection import Detector, LiveProcessor, MODEL_PATH

st.set_page_config(page_title="FireWatch | Fire & Smoke Detection", page_icon="🔥",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp { background: #f5f7fb; }
.block-container { max-width: 1440px; padding-top: 2rem; }
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e4e9f1; }
.hero { background: linear-gradient(115deg,#10213a,#203c58); padding: 2rem 2.4rem;
        border-radius: 20px; color: white; margin-bottom: 1.6rem; }
.eyebrow { color: #ffbd8b; font-size: .75rem; letter-spacing: .18em; font-weight: 700; }
.hero h1 { color: white; font-size: 2.5rem; padding: .4rem 0; letter-spacing: -.04em; }
.hero p { color: #c7d6e6; margin: 0; max-width: 680px; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #e4e9f1;
                         border-radius: 14px; padding: 1rem; }
[data-testid="stMetricLabel"] { color: #65738a; }
[data-testid="stMetricValue"] { color: #14263e; }
.footnote { color: #768399; font-size: .8rem; padding-top: 1.5rem; }
@media(max-width:640px) {
  .hero { padding: 1.4rem; } .hero h1 { font-size: 2rem; }
  .block-container { padding-top: 1rem; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector(weights_version):
    # Replacing the weights invalidates the resource cache.
    return Detector()


st.markdown("""
<div class="hero">
  <div class="eyebrow">FIRE & SMOKE MONITORING</div>
  <h1>FireWatch</h1>
  <p>Keep an eye on what matters. Live camera predictions and video analysis,
     with fire and smoke highlighted as they appear.</p>
</div>
""", unsafe_allow_html=True)

try:
    with st.spinner("Loading best_new.pt…"):
        detector = load_detector((MODEL_PATH.stat().st_mtime_ns, MODEL_PATH.stat().st_size))
except Exception as exc:
    st.error(f"Could not load best_new.pt: {exc}")
    st.info("Place the trained best_new.pt weights beside app.py and restart the app.")
    st.stop()

with st.sidebar:
    st.title("🔥 FireWatch")
    st.caption("YOUR DETECTION WORKSPACE")
    page = st.radio("Workspace", ["Live camera", "Analyze video"], label_visibility="collapsed")
    st.divider()
    st.subheader("Detection settings")
    confidence = st.slider("Confidence threshold", 0.10, 0.95, 0.40, 0.05,
                           help="Lower values show more predictions. Higher values require more confidence.")
    st.caption("Changes apply to the running camera without restarting it.")
    st.divider()
    st.caption("ACTIVE MODEL")
    st.code("best_new.pt", language=None)
    st.success("Model loaded")
    st.caption("Classes: " + ", ".join(detector.model.names.values()))


def rtc_configuration():
    servers = [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
    try:
        configured = st.secrets.get("webrtc", {}).get("ice_servers")
        if configured:
            servers = [dict(server) for server in configured]
    except FileNotFoundError:
        pass
    return {"iceServers": servers}


if page == "Live camera":
    heading, badge = st.columns([3, 1])
    with heading:
        st.subheader("Live camera")
        st.caption("Continuous predictions · annotated video · microphone off")
    with badge:
        camera_on = st.toggle("Camera on", value=True, key="camera_on")

    video_column, help_column = st.columns([3, 1])
    with video_column:
        with st.container(border=True):
            ctx = webrtc_streamer(
                key="firewatch-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration(),
                desired_playing_state=camera_on,
                video_processor_factory=lambda: LiveProcessor(detector, confidence),
                media_stream_constraints={
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                              "frameRate": {"ideal": 15, "max": 20}},
                    "audio": False,
                },
                async_processing=True,
                video_html_attrs={"autoPlay": True, "controls": False,
                                  "muted": True, "playsInline": True},
            )
            if ctx.video_processor:
                ctx.video_processor.set_confidence(confidence)
    with help_column:
        st.markdown("#### Ready when you are")
        st.write("Allow camera access in your browser. Predictions continue while the camera is on.")
        st.markdown("**Fire** · detection alert\n\n**Smoke** · detection alert")
        st.caption("Counts show objects in the latest predicted frame, not unique incidents.")
        with st.expander("Camera help"):
            st.write("Use localhost or an HTTPS address. Allow camera permission, close other apps using it, and select the correct camera in the video controls.")
            st.write("If the connection stalls, switch the camera off and on. Restricted networks may need a TURN relay configured by the app owner.")

    @st.fragment(run_every=0.5)
    def live_status():
        processor = ctx.video_processor
        stats = processor.snapshot() if processor else None
        fresh = bool(stats and stats.last_prediction and time.monotonic() - stats.last_prediction < 5)
        if not ctx.state.playing:
            if camera_on:
                st.info("Waiting for camera connection. Allow browser access or use START in the video controls.")
            else:
                st.info("Camera off. Switch Camera on to begin continuous detection.")
        elif stats and stats.error:
            st.error(f"Prediction failed: {stats.error}. Switch the camera off and on to retry.")
        elif not fresh:
            st.warning("Camera connected · waiting for predictions. If this persists, restart the camera.")
        elif stats.fire or stats.smoke:
            st.error(f"Detection alert · {stats.fire} fire / {stats.smoke} smoke in the current frame")
        else:
            st.success("Live · predicting continuously · no fire or smoke detected in the current frame")
        valid = bool(ctx.state.playing and fresh and not stats.error)
        columns = st.columns(4)
        for column, label, value in zip(columns, ["Fire in frame", "Smoke in frame", "Prediction FPS", "Frames analyzed"],
                                       [stats.fire if valid else "—", stats.smoke if valid else "—",
                                        f"{stats.fps:.1f}" if valid else "—", stats.frames if stats else 0]):
            column.metric(label, value)
        if valid:
            st.caption(f"Latest inference: {stats.latency_ms:.0f} ms · model: best_new.pt · confidence: {confidence:.0%}")

    live_status()

else:
    st.subheader("Analyze a video")
    st.caption("Upload a recording to annotate each frame and download the results.")
    uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        video_data = uploaded.getvalue()
        signature = hashlib.sha256(video_data).hexdigest()
        if st.session_state.get("upload_signature") != signature:
            st.session_state.pop("video_result", None)
            st.session_state["upload_signature"] = signature
        run = st.button("Analyze video", type="primary", use_container_width=True)
        if run:
            st.session_state.pop("video_result", None)
            progress = st.progress(0, text="Preparing video…")
            preview = st.empty()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / ("input" + Path(uploaded.name).suffix)
                    output = Path(directory) / "detected.mp4"
                    source.write_bytes(video_data)
                    cap = cv2.VideoCapture(str(source))
                    try:
                        if not cap.isOpened():
                            raise ValueError("Could not read this video. Try another MP4 file.")
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        fps = fps if 0 < fps <= 240 else 25.0
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        frames, fire, smoke = 0, 0, 0
                        with av.open(str(output), mode="w") as container:
                            stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
                            stream.width = max(2, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2)
                            stream.height = max(2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2)
                            stream.pix_fmt = "yuv420p"
                            stream.options = {"preset": "veryfast", "crf": "23"}
                            while True:
                                ok, frame = cap.read()
                                if not ok:
                                    break
                                annotated, counts = detector.predict(frame, confidence)
                                frames += 1
                                fire += counts["fire"]
                                smoke += counts["smoke"]
                                encoded_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
                                for packet in stream.encode(encoded_frame):
                                    container.mux(packet)
                                if frames == 1 or frames % 5 == 0:
                                    preview.image(annotated, channels="BGR", use_container_width=True)
                                    progress.progress(min(frames / total, 1.0) if total > 0 else 0,
                                                      text=f"Analyzed {frames:,} frames")
                            for packet in stream.encode():
                                container.mux(packet)
                        if not frames:
                            raise ValueError("The video contains no readable frames.")
                    finally:
                        cap.release()
                    st.session_state["video_result"] = {
                        "data": output.read_bytes(), "frames": frames, "fire": fire,
                        "smoke": smoke, "confidence": confidence,
                    }
                progress.progress(1.0, text="Analysis complete")
                preview.empty()
            except Exception as exc:
                progress.empty()
                st.error(f"Video analysis failed: {exc}")
        result = st.session_state.get("video_result")
        if result:
            st.success("Analysis complete")
            cols = st.columns(3)
            for col, label, key in zip(cols, ["Frames analyzed", "Fire detections", "Smoke detections"], ["frames", "fire", "smoke"]):
                col.metric(label, result[key])
            st.caption(f"Counts include repeated detections across frames · confidence {result['confidence']:.0%} · output has no audio")
            st.video(result["data"])
            st.download_button("Download annotated video", result["data"],
                               "firewatch_detected.mp4", "video/mp4", use_container_width=True)

st.markdown('<div class="footnote">FireWatch · Fire & smoke detection powered by your trained model</div>',
            unsafe_allow_html=True)
