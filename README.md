# 🔥 Fire & Smoke Detection

Real-time fire and smoke detection powered by **YOLO11** and **Streamlit**.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## Features

- 📹 Upload video files (MP4, AVI, MOV, MKV)
- 🔍 Real-time YOLO11 fire & smoke detection
- 📊 Detection statistics and analytics
- 🎬 Fully playable output video with annotations
- ⬇️ Download processed video
- ⚡ Adjustable confidence threshold

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | YOLO11 (Ultralytics) |
| Frontend | Streamlit |
| Vision | OpenCV |
| Encoding | FFmpeg (H.264) |

## Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/fire-detection.git
cd fire-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repo → branch `main` → file `app.py`
5. Click **Deploy**

> The app auto-installs system packages from `packages.txt` and Python packages from `requirements.txt`.

## Project Structure

```
fire-detection/
├── app.py                 # Streamlit web app
├── detect.py              # Standalone detection script
├── best_new.pt            # YOLO11 trained weights (primary model)
├── best.pt                # YOLO11 trained weights (fallback model)
├── requirements.txt       # Python dependencies
├── packages.txt           # System dependencies (apt)
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── .gitignore
└── README.md
```

## License

MIT
