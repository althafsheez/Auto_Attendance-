# ================================
# Optimized YOLOv8 + InsightFace + Frame Skipping (Every 3 Frames)
# ================================
import os
import time
import cv2
import pickle
import numpy as np
from numpy.linalg import norm
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from duration_tracker import DurationTracker


# -------------------------------
# STEP 1: Load Models
# -------------------------------
print("🔄 Loading YOLOv8n-face model (GPU)...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-face.pt")

yolo_model = YOLO(YOLO_MODEL_PATH).to("cpu")

print("✅ YOLO initialized on:", yolo_model.device)

print("🔄 Loading InsightFace model (buffalo_s)...")
face_app = FaceAnalysis(name='buffalo_s')
face_app.prepare(ctx_id=0, det_size=(640, 640))

for name, sub_model in face_app.models.items():
    try:
        sub_model.session.set_providers(['CUDAExecutionProvider'])
        print(f"✅ '{name}' on GPU")
    except Exception as e:
        print(f"⚠️ GPU not set for '{name}':", e)

# -------------------------------
# STEP 2: Load Face Database
# -------------------------------
db_path = os.path.join(os.path.dirname(__file__), "students_opt.pkl")
print(f"📂 Loading database: {db_path}")
with open(db_path, "rb") as f:
    face_db = pickle.load(f)
print(f"✅ Loaded {len(face_db)} student embeddings\n")

all_students = list(face_db.keys())
attendance = {name: 0 for name in all_students}

# -------------------------------
# STEP 3: Recognition Function
# -------------------------------
def recognize_face(face_crop):
    # -------------------------------
    # Step A: Lighting Normalization
    # -------------------------------
    try:
        # Convert to YCrCb color space (better for brightness correction)
        ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)  # histogram equalization on luminance channel
        ycrcb_eq = cv2.merge((y_eq, cr, cb))
        face_crop = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

        # Optional small brightness normalization
        face_crop = cv2.convertScaleAbs(face_crop, alpha=1.1, beta=10)
    except Exception as e:
        print(f"⚠️ Lighting normalization failed: {e}")

    # -------------------------------
    # Step B: Recognition (unchanged)
    # -------------------------------
    faces = face_app.get(face_crop)
    if not faces:
        return "Unknown", 0.0

    emb = faces[0].embedding
    best_name, best_score = "Unknown", -1

    for name, db_emb in face_db.items():
        score = np.dot(emb, db_emb) / (norm(emb) * norm(db_emb))
        if score > best_score:
            best_score, best_name = score, name

    if best_score > 0.28:
        attendance[best_name] = 1
        return best_name, best_score
    else:
        return "Unknown", best_score


# -------------------------------
# STEP 4: Start Live Camera (Every 3rd Frame Detection)
# -------------------------------
cap = cv2.VideoCapture("http://192.168.0.148:8080/video")
if not cap.isOpened():
    print("❌ Failed to open video stream")
    exit()

print("🚀 Starting live recognition (every 3 frames, press 'q' to quit)...\n")

prev_time = time.time()
frame_count = 0
cached_boxes = []  # reuse detections for skipped frames

tracker = DurationTracker()


while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed.")
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # Run YOLO detection every 3rd frame only
    if frame_count % 3 == 0:
        results = yolo_model(frame, conf=0.3, imgsz=320, verbose=False)
        cached_boxes = results[0].boxes.xyxy.cpu().numpy()  # store detections
    else:
        # Skip detection — reuse last boxes
        results = None

    # Use cached boxes for recognition
    for box in cached_boxes:
        x1, y1, x2, y2 = map(int, box)
        h, w, _ = frame.shape
        box_w, box_h = x2 - x1, y2 - y1
        pad_x, pad_y = int(0.3 * box_w), int(0.3 * box_h)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = frame[y1:y2, x1:x2]
        name, conf = recognize_face(face_crop)
        
        if name != "Unknown":
            tracker.update(name)
        
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"{name} ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # FPS Counter
    # FPS Counter (safe version)
    curr_time = time.time()
    time_diff = curr_time - prev_time if curr_time - prev_time > 0 else 1e-6  # avoid zero division
    fps = 1.0 / time_diff
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    

    cv2.imshow("🎥 Optimized Face Recognition (Every 3 Frames)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------
# STEP 5: Attendance Summary
# -------------------------------
print("\n📊 Attendance Summary:")
for name, status in attendance.items():
    print(f"{name}: {'Present ✅' if status == 1 else 'Absent ❌'}")

present_count = sum(attendance.values())
absent_count = len(attendance) - present_count

print(f"\n✅ Total Present: {present_count}")
print(f"❌ Total Absent: {absent_count}")
print("📅 Attendance recording complete.") 

# -------------------------------
# STEP 5A: Duration Summary
# -------------------------------
print("\n🕒 Duration per Student:")
durations = tracker.get_durations()
for name, duration in durations.items():
    print(f"{name}: {duration} seconds in frame")

