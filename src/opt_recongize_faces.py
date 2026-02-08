# ================================
# Optimized YOLOv8 + InsightFace Live Recognition
# ================================
import os
import time
import cv2
import pickle
import numpy as np
from numpy.linalg import norm
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis

# -------------------------------
# STEP 1: Load Models
# -------------------------------
print("🔄 Loading YOLOv8n-face model (GPU)...")
yolo_model = YOLO("yolov8n-face.pt").to("cuda")
print("✅ YOLO initialized on:", yolo_model.device)

print("🔄 Loading InsightFace model (buffalo_s)...")
face_app = FaceAnalysis(name='buffalo_s')  # 'buffalo_s' = medium, faster than 'buffalo_l'
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Force GPU usage for all sub-models
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

# -------------------------------
# STEP 3: Recognition Function
# -------------------------------
def recognize_face(face_crop):
    faces = face_app.get(face_crop)
    if not faces:
        return "Unknown", 0.0

    emb = faces[0].embedding
    best_name, best_score = "Unknown", -1

    for name, db_emb in face_db.items():
        score = np.dot(emb, db_emb) / (norm(emb) * norm(db_emb))
        if score > best_score:
            best_score, best_name = score, name

    return (best_name, best_score) if best_score > 0.28 else ("Unknown", best_score)

# -------------------------------
# STEP 4: Start Live Camera
# -------------------------------
cap = cv2.VideoCapture("http://172.16.197.156:8080/video")
if not cap.isOpened():
    print("❌ Failed to open video stream")
    exit()

print("🚀 Starting live recognition (press 'q' to quit)...\n")
prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed.")
        break

    frame_count += 1

    # Resize frame (speed boost)
    frame = cv2.resize(frame, (640, 480))

    # Run YOLO every frame, recognition every 2nd frame
    results = yolo_model(frame, conf=0.3, imgsz=480, verbose=False)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)

        h, w, _ = frame.shape
        box_w, box_h = x2 - x1, y2 - y1
        pad_x, pad_y = int(0.3 * box_w), int(0.3 * box_h)

        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = frame[y1:y2, x1:x2]

        # Run recognition only every 2nd frame
        name, conf = recognize_face(face_crop)
        # Draw results
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"{name} ({conf:.2f})" if name != "Processing" else "..."
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("🎥 Face Recognition (Optimized)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Recognition ended.")
