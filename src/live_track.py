# ================================
# YOLOv8 Face Detection (Live)
# ================================

from ultralytics import YOLO
import cv2
import time
from ultralytics.utils import LOGGER
LOGGER.setLevel("ERROR")


# ------------------------
# STEP 1: Load your model
# ------------------------
# Make sure yolov8n-face.pt is in the same folder
model = YOLO("yolov8n-face.pt").to("cuda")
print("✅ YOLO running on:", model.device)

# ------------------------
# STEP 2: Choose video source
# ------------------------
# 0 = laptop webcam
# or use your phone IP Webcam stream:
# VIDEO_URL = "http://192.168.1.5:8080/video"
cap = cv2.VideoCapture("http://172.16.197.99:8080/video")

# ------------------------
# STEP 3: Live loop
# ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Could not read frame. Check your camera.")
        break

    # ------------------------
    # STEP 4: Run YOLO on the frame
    # ------------------------
   
    start = time.time()
    results = model(frame)
    end = time.time()
    fps = 1 / (end - start)
    # ------------------------
    # STEP 5: Draw boxes on faces
    # ------------------------
    for box, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        confidence = float(conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the result
    cv2.imshow("YOLOv8n-Face (GPU, FP16)", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------
# STEP 7: Clean up
# ------------------------
cap.release()
cv2.destroyAllWindows()
print("✅ Done!")
