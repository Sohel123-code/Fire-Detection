import streamlit as st
import cv2
import tempfile
import os
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
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL (cached)
# ==============================

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "best.pt"
    if not model_path.exists():
        return None
    return YOLO(str(model_path))

model = load_model()

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
            <span style="color: #0f3460; font-weight: 600; font-size: 0.85rem;">best.pt</span>
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
        st.error("⚠️ `best.pt` not found in project folder!")

# ==============================
# MAIN CONTENT
# ==============================

if not model:
    st.error("❌ Model file `best.pt` not found. Please place it in the project directory.")
    st.stop()

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
# FOOTER
# ==============================

st.markdown("""
<div class="footer">
    <p>🔥 Fire & Smoke Detection System • Built with YOLO11 + Streamlit</p>
</div>
""", unsafe_allow_html=True)
