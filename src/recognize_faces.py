# ================================
# YOLOv8 + InsightFace Recognition (Live)
# ================================
import os
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis
import cv2
import pickle
import numpy as np
from numpy.linalg import norm
import time


# -------------------------------
# STEP 1: Load Models
# -------------------------------
# -------------------------------
# STEP 1: Load Models
# -------------------------------
print("🔄 Loading YOLOv8 face model...")
yolo_model = YOLO("yolov8n-face.pt").to("cuda")
print("✅ YOLO device:", yolo_model.device)

print("🔄 Loading InsightFace (buffalo_l)...")
face_app = insightface.app.FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Force InsightFace sub-models to use GPU
for name, sub_model in face_app.models.items():
    try:
        sub_model.session.set_providers(['CUDAExecutionProvider'])
        print(f"✅ Model '{name}' using GPU provider")
    except Exception as e:
        print(f"⚠️ Could not set GPU provider for '{name}':", e)

# Verify
for name, sub_model in face_app.models.items():
    print(f"🔹 {name} providers:", sub_model.session.get_providers())


# -------------------------------
# STEP 2: Load Student Database
# -------------------------------
print("🔄 Loading face database...")
db_path = os.path.join(os.path.dirname(__file__), "students.pkl")
print("📂 Loading database from:", db_path)

with open(db_path, "rb") as f:
     face_db = pickle.load(f)
print(f"✅ Loaded {len(face_db)} students")

# -------------------------------
# STEP 3: Recognition Function
# -------------------------------
def recognize_face(face_crop):
    """Compare detected face with saved embeddings."""
    faces = face_app.get(face_crop)
    print(f"🧩 InsightFace detected {len(faces)} faces in crop")

    if len(faces) == 0:
        print("⚠️ No face detected in YOLO crop")
        return "Unknown", 0.0

    emb = faces[0].embedding
    best_match = None
    best_score = -1

    # Compare current embedding with all stored ones
    for name, db_emb in face_db.items():
        score = np.dot(emb, db_emb) / (norm(emb) * norm(db_emb))
        if score > best_score:
            best_score = score
            best_match = name

    if best_score > 0.25:  # threshold (adjust between 0.3–0.45)
        return best_match, best_score
    else:
        return "Unknown", best_score

# -------------------------------
# STEP 4: Start Webcam
# -------------------------------
cap = cv2.VideoCapture("http://172.16.197.99:8080/video")

print("🚀 Starting live recognition... Press 'q' to quit.")
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Could not read frame.")
        break

    # Detect faces with YOLOv8
    results = yolo_model(frame, conf=0.3)
    print(f"🔍 Detected {len(results[0].boxes)} faces in frame")


    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        # Expand box a little for InsightFace
        # Expand and clip box safely
        h, w, _ = frame.shape
        box_w = x2 - x1
        box_h = y2 - y1

        # padding = 30% of face size (you can tune 0.3 → 0.2 or 0.4)
        pad_x = int(0.35 * box_w)
        pad_y = int(0.35 * box_h)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y) 


        # Ensure valid crop
        if x2 <= x1 or y2 <= y1:
            print("⚠️ Invalid crop dimensions, skipping frame.")
            continue

        face_crop = frame[y1:y2, x1:x2]



        # Recognize face
        name, conf = recognize_face(face_crop)

        # Draw bounding box and name
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    

    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show live video
    cv2.imshow("🎥 Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Recognition ended.")
