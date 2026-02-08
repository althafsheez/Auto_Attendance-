import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

# -------------------------------
# STEP 1: Setup YOLOv8 Face Detector
# -------------------------------
print("🔄 Loading YOLOv8n-face model for capturing...")
face_detector = YOLO("yolov8n-face.pt")

# -------------------------------
# STEP 2: Create Folder Structure
# -------------------------------
roll_number = input("🧾 Enter Roll Number (e.g., 2022BCS001): ").strip()
name = input("👤 Enter Full Name: ").strip()

save_dir = os.path.join("..", "data", roll_number)
os.makedirs(save_dir, exist_ok=True)
print(f"📂 Folder created at: {save_dir}")

# -------------------------------
# STEP 3: Initialize Camera
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("\n📸 Instructions:")
print("➡ Keep your face inside the green box.")
print("➡ Look straight, left, right, up, and down when asked.")
print("⚠ Avoid other faces in frame.\n")

angles = ["Center", "Left", "Right", "Down", "Up"]
current_angle = 0
photo_id = 0
frame_count = 0

# -------------------------------
# STEP 4: Capture Loop
# -------------------------------
last_capture_time = 0  # to space out captures
capture_delay = 1.0    # seconds between photos

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Could not read frame.")
        break

    frame = cv2.flip(frame, 1)
    results = face_detector(frame, conf=0.5, imgsz=320, verbose=False)

    if len(results[0].boxes) > 0:
        # Select biggest face
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        biggest_idx = np.argmax(areas)
        x1, y1, x2, y2 = map(int, boxes[biggest_idx])

        # Expand crop for full head (50% padding)
        pad_w, pad_h = int(0.7 * (x2 - x1)), int(0.5 * (y2 - y1))
        x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
        x2, y2 = min(frame.shape[1], x2 + pad_w), min(frame.shape[0], y2 + pad_h)

        # Draw guide box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Capturing: {angles[current_angle]}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        face_crop = frame[y1:y2, x1:x2]

        # Lighting normalization (YCrCb)
        ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = cv2.equalizeHist(y)
        clean_face = cv2.merge([y, cr, cb])
        clean_face = cv2.cvtColor(clean_face, cv2.COLOR_YCrCb2BGR)
        clean_face = cv2.resize(clean_face, (224, 224))

        # Delay-based photo capture
        current_time = time.time()
        if current_time - last_capture_time > capture_delay:
            photo_id += 1
            save_path = os.path.join(save_dir, f"{name}_{photo_id}.jpg")
            cv2.imwrite(save_path, clean_face)
            last_capture_time = current_time
            print(f"✅ Saved {save_path}")

            if photo_id % 5 == 0:
                current_angle += 1
                if current_angle >= len(angles):
                    print("\n📸 Capture complete!")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                print(f"\n➡ Now please face: {angles[current_angle]}")
                time.sleep(2)  # small break for next pose

    frame_count += 1
    cv2.putText(frame, f"Photos: {photo_id}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("📷 Dataset Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("❌ Stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()
