import os
import cv2
import insightface
import numpy as np
import pickle
from numpy.linalg import norm

# -----------------------------
# Initialize InsightFace
# -----------------------------
print("🔄 Initializing InsightFace model (buffalo_l)...")
face_app = insightface.app.FaceAnalysis(name='buffalo_s')
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ Model ready.\n")

# -----------------------------
# Paths
# -----------------------------
base_folder = "../data"   # main folder containing student subfolders
output_db = "students_opt.pkl"

# -----------------------------
# Load existing database if any
# -----------------------------
if os.path.exists(output_db):
    with open(output_db, "rb") as f:
        db = pickle.load(f)
    print(f"📂 Loaded existing database with {len(db)} students.\n")
else:
    db = {}
    print("🆕 Starting new student database.\n")

# -----------------------------
# Loop through each student folder
# -----------------------------
for student_id in os.listdir(base_folder):
    folder = os.path.join(base_folder, student_id)
    if not os.path.isdir(folder):
        continue  # skip files

    print(f"👤 Processing student: {student_id}")
    embeddings = []

    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, file)
            img = cv2.imread(path)

            if img is None:
                print(f"⚠️ Could not read {path}")
                continue

            faces = face_app.get(img)
            if len(faces) > 0:
                emb = faces[0].embedding
                embeddings.append(emb)
                print(f"   ✅ Processed {file}")
            else:
                print(f"   ⚠️ No face found in {file}")

    if embeddings:
        avg_emb = np.mean(embeddings, axis=0)
        db[student_id] = avg_emb
        print(f"✅ Saved embedding for {student_id} ({len(embeddings)} images)\n")
    else:
        print(f"❌ No valid faces for {student_id}\n")

# -----------------------------
# Save all embeddings
# -----------------------------
with open(output_db, "wb") as f:
    pickle.dump(db, f)

print(f"🎉 Done! Total students saved: {len(db)}")
print(f"📦 Database file: {output_db}")
