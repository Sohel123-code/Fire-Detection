import streamlit as st
import cv2
import tempfile
import os
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="🔥 Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global styles - White background, dark blue text */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff !important;
        color: #0f3460 !important;
    }

    /* Override Streamlit default dark backgrounds */
    .stApp > header {
        background-color: #ffffff !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #f0f4f8 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #0f3460 !important;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #1a1a2e 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(15, 52, 96, 0.2);
        box-shadow: 0 10px 40px rgba(15, 52, 96, 0.15);
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.08) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #a0c4e8;
        font-size: 1.05rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    /* Stat cards */
    .stat-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(15, 52, 96, 0.08);
    }

    .stat-card:hover {
        border-color: #0f3460;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(15, 52, 96, 0.15);
    }

    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f3460;
        margin-bottom: 0.25rem;
    }

    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Upload area */
    .upload-zone {
        background: #f8fafc;
        border: 2px dashed #0f3460;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 1.5rem 0;
    }

    .upload-zone:hover {
        border-color: #16213e;
        background: #eef2f7;
    }

    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .upload-text {
        color: #0f3460;
        font-size: 1.1rem;
        font-weight: 500;
    }

    .upload-hint {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .status-ready {
        background: rgba(34, 197, 94, 0.1);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .status-processing {
        background: rgba(15, 52, 96, 0.1);
        color: #0f3460;
        border: 1px solid rgba(15, 52, 96, 0.3);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .status-done {
        background: rgba(15, 52, 96, 0.1);
        color: #0f3460;
        border: 1px solid rgba(15, 52, 96, 0.3);
    }

    /* Results section */
    .results-header {
        background: #f0f4f8;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        border: 1px solid #d1dce6;
    }

    .results-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f3460;
        margin-bottom: 0.3rem;
    }

    /* Detection label */
    .detection-label {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    .label-fire {
        background: rgba(239, 68, 68, 0.1);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .label-smoke {
        background: rgba(100, 116, 139, 0.1);
        color: #475569;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }

    /* Sidebar styling */
    .sidebar-section {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15, 52, 96, 0.06);
    }

    .sidebar-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #0f3460;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
    }

    /* Camera live badge */
    .camera-live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 700;
        background: rgba(239, 68, 68, 0.1);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
        animation: livePulse 1.5s infinite;
        letter-spacing: 1px;
    }

    @keyframes livePulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { opacity: 0.85; box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    }

    /* Camera panel */
    .camera-panel {
        background: #f8fafc;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(15, 52, 96, 0.08);
    }

    .camera-panel-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #0f3460;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #64748b;
        font-size: 0.8rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Progress bar custom */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0f3460, #1a5276);
        border-radius: 10px;
    }

    /* Divider styling */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #0f3460, transparent);
        margin: 2rem 0;
        border: none;
        opacity: 0.2;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f0f4f8;
        border-radius: 12px;
        padding: 0.4rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.6rem 1.5rem;
        color: #64748b;
    }

    .stTabs [aria-selected="true"] {
        background: #0f3460 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL (cached)
# ==============================

@st.cache_resource
def load_model():
    for name in ["best_new.pt", "best.pt"]:
        model_path = Path(__file__).parent / name
        if model_path.exists():
            return YOLO(str(model_path)), name
    return None, None

model, loaded_model_name = load_model()

# ==============================
# HEADER
# ==============================

st.markdown("""
<div class="hero-header">
    <div class="hero-title">🔥 Fire & Smoke Detection</div>
    <div class="hero-subtitle">Powered by YOLO11 — Real-time fire and smoke detection with deep learning</div>
</div>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">🛡️</span>
        <h2 style="color: #0f3460; margin: 0.5rem 0 0; font-size: 1.2rem; font-weight: 700;">Detection Settings</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Confidence threshold
    st.markdown('<div class="sidebar-title">⚡ Confidence Threshold</div>', unsafe_allow_html=True)
    confidence = st.slider(
        "Minimum confidence for detection",
        min_value=0.10,
        max_value=0.95,
        value=0.40,
        step=0.05,
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Model info
    st.markdown('<div class="sidebar-title">🧠 Model Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-section">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #64748b; font-size: 0.85rem;">Model</span>
            <span style="color: #0f3460; font-weight: 600; font-size: 0.85rem;">YOLO11</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #64748b; font-size: 0.85rem;">Weights</span>
            <span style="color: #0f3460; font-weight: 600; font-size: 0.85rem;">{loaded_model_name or "Not Found"}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #64748b; font-size: 0.85rem;">Confidence</span>
            <span style="color: #0f3460; font-weight: 600; font-size: 0.85rem;">{confidence:.0%}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #64748b; font-size: 0.85rem;">Classes</span>
            <span style="color: #0f3460; font-weight: 600; font-size: 0.85rem;">Fire, Smoke</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Detection classes legend
    st.markdown('<div class="sidebar-title">🏷️ Detection Classes</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        <span class="detection-label label-fire">🔥 Fire</span>
        <span class="detection-label label-smoke">💨 Smoke</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Status
    if model:
        st.markdown("""
        <div style="text-align: center;">
            <span class="status-badge status-ready">● Model Ready</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Model weights (`best_new.pt` or `best.pt`) not found in project folder!")

# ==============================
# MAIN CONTENT
# ==============================

if not model:
    st.error("❌ Model file (`best_new.pt` or `best.pt`) not found. Please place it in the project directory.")
    st.stop()

# ==============================
# TABS
# ==============================

tab1, tab2 = st.tabs(["📹 Upload Video", "📷 Live Camera"])

# ==============================
# TAB 1: UPLOAD VIDEO
# ==============================

with tab1:
    # Upload section
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">📹</div>
        <div class="upload-text">Upload a video for fire & smoke detection</div>
        <div class="upload-hint">Supports MP4, AVI, MOV, MKV • Max 500MB</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Save uploaded file to temp location
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()

        # Open video to get info
        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("❌ Could not open the uploaded video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Video stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{width}×{height}</div>
                <div class="stat-label">Resolution</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{fps:.0f}</div>
                <div class="stat-label">FPS</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{total_frames:,}</div>
                <div class="stat-label">Total Frames</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{duration:.1f}s</div>
                <div class="stat-label">Duration</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Run detection button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            run_detection = st.button(
                "🚀 Run Detection",
                use_container_width=True,
                type="primary",
            )

        if run_detection:
            st.markdown("""
            <div style="text-align: center; margin: 1rem 0;">
                <span class="status-badge status-processing">◉ Processing Video...</span>
            </div>
            """, unsafe_allow_html=True)

            # Output file
            output_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ).name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Reset video capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Progress
            progress_bar = st.progress(0, text="Analyzing frames...")
            frame_display = st.empty()
            stats_display = st.empty()

            fire_count = 0
            smoke_count = 0
            frames_with_detections = 0

            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO detection
                results = model(frame, conf=confidence, verbose=False)

                # Count detections
                boxes = results[0].boxes
                if len(boxes) > 0:
                    frames_with_detections += 1
                    for box in boxes:
                        cls = int(box.cls[0])
                        cls_name = results[0].names[cls].lower()
                        if "fire" in cls_name:
                            fire_count += 1
                        elif "smoke" in cls_name:
                            smoke_count += 1

                # Annotate frame
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

                # Update progress
                progress = (frame_idx + 1) / total_frames
                progress_bar.progress(
                    progress,
                    text=f"Processing frame {frame_idx + 1}/{total_frames} "
                         f"({progress:.0%})"
                )

                # Show live preview every 10 frames
                if frame_idx % 10 == 0:
                    preview = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_display.image(preview, channels="RGB", use_container_width=True)

            # Cleanup
            cap.release()
            out.release()

            progress_bar.progress(1.0, text="✅ Detection complete!")

            # Clear live preview
            frame_display.empty()

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ==============================
            # RESULTS
            # ==============================

            st.markdown("""
            <div class="results-header">
                <div class="results-title">📊 Detection Results</div>
            </div>
            """, unsafe_allow_html=True)

            # Result stats
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value" style="color: #dc2626;">🔥 {fire_count}</div>
                    <div class="stat-label">Fire Detections</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value" style="color: #475569;">💨 {smoke_count}</div>
                    <div class="stat-label">Smoke Detections</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                pct = (frames_with_detections / total_frames * 100) if total_frames > 0 else 0
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value" style="color: #d97706;">{pct:.1f}%</div>
                    <div class="stat-label">Frames with Alerts</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value" style="color: #16a34a;">{total_frames:,}</div>
                    <div class="stat-label">Frames Processed</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Re-encode to H.264 for full browser playback (seekable, compatible)
            h264_output = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ).name

            try:
                import subprocess
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", output_path,
                    "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-acodec", "aac",
                    "-strict", "experimental",
                    h264_output
                ], capture_output=True, check=True)
                final_output = h264_output
            except Exception:
                # Fallback to mp4v output if ffmpeg not available
                final_output = output_path

            # Read video bytes into memory BEFORE cleanup
            # This ensures the video stays fully playable even after temp files are removed
            with open(final_output, "rb") as f:
                video_bytes = f.read()

            # Store in session state so the video persists across reruns
            st.session_state["output_video_bytes"] = video_bytes

            # Cleanup temp files now that bytes are in memory
            try:
                os.unlink(tfile.name)
                os.unlink(output_path)
                if final_output != output_path:
                    os.unlink(h264_output)
            except Exception:
                pass

            # Show output video from bytes (fully playable: play, pause, seek, replay)
            st.markdown("### 🎬 Output Video")
            st.video(video_bytes, format="video/mp4")

            # Download button
            col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
            with col_dl2:
                st.download_button(
                    label="⬇️ Download Detected Video",
                    data=video_bytes,
                    file_name="fire_smoke_detected.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    type="primary",
                )

            st.markdown("""
            <div style="text-align: center; margin-top: 1rem;">
                <span class="status-badge status-done">✅ Analysis Complete</span>
            </div>
            """, unsafe_allow_html=True)

        elif "output_video_bytes" in st.session_state:
            # Re-display previously processed video on page rerun
            video_bytes = st.session_state["output_video_bytes"]

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="results-header">
                <div class="results-title">📊 Previous Detection Results</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🎬 Output Video")
            st.video(video_bytes, format="video/mp4")

            col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
            with col_dl2:
                st.download_button(
                    label="⬇️ Download Detected Video",
                    data=video_bytes,
                    file_name="fire_smoke_detected.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    type="primary",
                )

            st.markdown("""
            <div style="text-align: center; margin-top: 1rem;">
                <span class="status-badge status-done">✅ Analysis Complete</span>
            </div>
            """, unsafe_allow_html=True)

# ==============================
# TAB 2: LIVE CAMERA
# ==============================

with tab2:
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">📷</div>
        <div class="upload-text">Live Camera Fire & Smoke Detection</div>
        <div class="upload-hint">Real-time detection using your browser webcam or local camera</div>
    </div>
    """, unsafe_allow_html=True)

    cam_mode = st.radio(
        "Camera Mode",
        options=["🌐 Browser Camera (Cloud & Web Compatible)", "🖥️ Direct Camera Stream (Local Only)"],
        index=0,
        horizontal=True,
        help="Use Browser Camera for Streamlit Cloud deployment or mobile/laptop web browsers. Use Direct Camera for local desktop OpenCV execution."
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if "Browser Camera" in cam_mode:
        # Browser Camera Mode using st.camera_input (Works on Streamlit Cloud & Browsers)
        cam_col1, cam_col2 = st.columns([3, 1])

        with cam_col2:
            st.markdown('<div class="camera-panel">', unsafe_allow_html=True)
            st.markdown('<div class="camera-panel-title">🎛️ Camera Status</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="text-align: center; margin: 0.8rem 0;">
                <span class="camera-live-badge">🌐 BROWSER WEBCAM</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="camera-panel-title">📊 Frame Detections</div>', unsafe_allow_html=True)
            fire_metric = st.empty()
            smoke_metric = st.empty()
            status_metric = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        with cam_col1:
            camera_file = st.camera_input("📷 Capture Frame from Webcam", key="browser_cam_input")

            if camera_file:
                # Read image bytes
                bytes_data = camera_file.getvalue()
                file_bytes = np.frombuffer(bytes_data, np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Run YOLO detection
                    results = model(frame, conf=confidence, verbose=False)
                    annotated_frame = results[0].plot()

                    # Count detections
                    fire_count = 0
                    smoke_count = 0
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        for box in boxes:
                            cls = int(box.cls[0])
                            cls_name = results[0].names[cls].lower()
                            if "fire" in cls_name:
                                fire_count += 1
                            elif "smoke" in cls_name:
                                smoke_count += 1

                    # Update metrics
                    fire_metric.metric("🔥 Fire Detected", fire_count, delta="ACTIVE" if fire_count > 0 else None, delta_color="inverse")
                    smoke_metric.metric("💨 Smoke Detected", smoke_count, delta="ACTIVE" if smoke_count > 0 else None, delta_color="inverse")

                    if fire_count > 0 and smoke_count > 0:
                        status_metric.markdown('<span class="status-badge" style="background: #fee2e2; color: #991b1b;">⚠️ FIRE & SMOKE DETECTED!</span>', unsafe_allow_html=True)
                    elif fire_count > 0:
                        status_metric.markdown('<span class="status-badge" style="background: #fee2e2; color: #991b1b;">🔥 FIRE DETECTED!</span>', unsafe_allow_html=True)
                    elif smoke_count > 0:
                        status_metric.markdown('<span class="status-badge" style="background: #fef3c7; color: #92400e;">💨 SMOKE DETECTED!</span>', unsafe_allow_html=True)
                    else:
                        status_metric.markdown('<span class="status-badge status-ready">✅ ALL CLEAR</span>', unsafe_allow_html=True)

                    # Display annotated output image
                    display_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st.markdown("### 🎯 Detection Result")
                    st.image(display_rgb, caption="Analyzed Webcam Snapshot", use_container_width=True)

    else:
        # Direct OpenCV Camera Stream Mode (For local execution)
        cam_col1, cam_col2 = st.columns([3, 1])

        with cam_col2:
            st.markdown('<div class="camera-panel">', unsafe_allow_html=True)
            st.markdown('<div class="camera-panel-title">🎛️ Camera Controls</div>', unsafe_allow_html=True)

            camera_index = st.selectbox(
                "Camera Source Index",
                options=[0, 1, 2],
                format_func=lambda x: f"Camera {x}",
                key="camera_source",
            )

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            camera_active = st.toggle("🎥 Start Live Stream", value=False, key="camera_toggle")

            if camera_active:
                st.markdown("""
                <div style="text-align: center; margin: 0.8rem 0;">
                    <span class="camera-live-badge">🔴 LIVE STREAM</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; margin: 0.8rem 0;">
                    <span class="status-badge status-ready">⏸️ Standby</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            st.markdown('<div class="camera-panel-title">📊 Live Stats</div>', unsafe_allow_html=True)
            fire_metric = st.empty()
            smoke_metric = st.empty()
            frame_metric = st.empty()
            fps_metric = st.empty()

            st.markdown('</div>', unsafe_allow_html=True)

        with cam_col1:
            stframe = st.empty()

            if not camera_active:
                stframe.markdown("""
                <div style="
                    background: #f0f4f8;
                    border-radius: 16px;
                    padding: 5rem 2rem;
                    text-align: center;
                    border: 1px solid #e2e8f0;
                ">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">📷</div>
                    <div style="color: #64748b; font-size: 1.1rem; font-weight: 500;">
                        Direct camera stream will appear here
                    </div>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">
                        Note: Direct stream works on local machine. For Streamlit Cloud, switch to "Browser Camera".
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if camera_active:
            try:
                cap = cv2.VideoCapture(camera_index)

                if not cap.isOpened():
                    stframe.error("❌ Could not open local camera device. If running on Streamlit Cloud, please switch to 'Browser Camera' mode above.")
                else:
                    fire_total = 0
                    smoke_total = 0
                    frame_count = 0
                    start_time = time.time()

                    while cap.isOpened() and st.session_state.get("camera_toggle", False):
                        ret, frame = cap.read()
                        if not ret:
                            stframe.warning("⚠️ Camera feed lost. Please toggle off and on to restart.")
                            break

                        frame_count += 1
                        results = model(frame, conf=confidence, verbose=False)

                        frame_fire = 0
                        frame_smoke = 0
                        boxes = results[0].boxes
                        if len(boxes) > 0:
                            for box in boxes:
                                cls = int(box.cls[0])
                                cls_name = results[0].names[cls].lower()
                                if "fire" in cls_name:
                                    fire_total += 1
                                    frame_fire += 1
                                elif "smoke" in cls_name:
                                    smoke_total += 1
                                    frame_smoke += 1

                        annotated = results[0].plot()
                        display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        stframe.image(display_frame, use_container_width=True)

                        if frame_count % 5 == 0:
                            elapsed = time.time() - start_time
                            current_fps = frame_count / elapsed if elapsed > 0 else 0
                            fire_metric.metric("🔥 Fire", fire_total,
                                               delta=f"+{frame_fire}" if frame_fire > 0 else None,
                                               delta_color="inverse")
                            smoke_metric.metric("💨 Smoke", smoke_total,
                                                delta=f"+{frame_smoke}" if frame_smoke > 0 else None,
                                                delta_color="inverse")
                            frame_metric.metric("🖼️ Frames", f"{frame_count:,}")
                            fps_metric.metric("⚡ FPS", f"{current_fps:.1f}")

                    cap.release()
            except Exception as e:
                stframe.error(f"❌ Camera error: {e}. Please switch to 'Browser Camera' mode.")

# ==============================
# FOOTER
# ==============================

st.markdown("""
<div class="footer">
    <p>🔥 Fire & Smoke Detection System • Built with YOLO11 + Streamlit</p>
</div>
""", unsafe_allow_html=True)
