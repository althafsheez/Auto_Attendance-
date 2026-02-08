import pickle
import pprint

# Load the students database (since it's in the same folder)
with open("students.pkl", "rb") as f:
    db = pickle.load(f)

print("\n✅ Loaded entries:")
for name, emb in db.items():
    print(f"\n🎓 Student ID/Name: {name}")
    print(f"📏 Embedding length: {len(emb)}")
    print("📊 Embedding vector:\n")
    pprint.pprint(emb.tolist(), width=120)
