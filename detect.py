import cv2
from ultralytics import YOLO

# ==============================
# FILE PATHS
# ==============================

MODEL_PATH = "best.pt"
VIDEO_PATH = "video.mp4"
OUTPUT_PATH = "output.mp4"

# ==============================
# LOAD MODEL
# ==============================

model = YOLO(MODEL_PATH)

print("✅ Model loaded successfully")

# ==============================
# OPEN VIDEO
# ==============================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Could not open video.mp4")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Original video: {width} x {height}")
print(f"FPS: {fps}")

# ==============================
# OUTPUT VIDEO
# ==============================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

# ==============================
# CREATE RESIZABLE WINDOW
# ==============================

WINDOW_NAME = "Fire & Smoke Detection - YOLO11"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

# Fit window to screen
cv2.resizeWindow(
    WINDOW_NAME,
    1000,
    700
)

# ==============================
# DETECTION
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    results = model(
        frame,
        conf=0.40,
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # ==============================
    # DISPLAY FULL VIDEO
    # ==============================

    cv2.imshow(
        WINDOW_NAME,
        annotated_frame
    )

    # ==============================
    # SAVE ORIGINAL RESOLUTION
    # ==============================

    out.write(annotated_frame)

    # Press Q to stop
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n================================")
print("✅ Detection completed!")
print("📁 Output saved:", OUTPUT_PATH)
print("================================")