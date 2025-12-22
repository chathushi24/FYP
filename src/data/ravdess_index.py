import os
import csv

# ======================================================
# CONFIG – DEFINE PATHS FIRST (VERY IMPORTANT)
# ======================================================
DATASET_PATH = "/content/drive/MyDrive/My_RAVDESS_Dataset"
OUTPUT_CSV = "/content/drive/MyDrive/ravdess_index.csv"

# ======================================================
# DEBUG CHECKS (RUNS BEFORE PROCESSING)
# ======================================================
print("=== DEBUG START ===")
print("DATASET_PATH:", DATASET_PATH)
print("Exists?", os.path.exists(DATASET_PATH))

if os.path.exists(DATASET_PATH):
    print("Top-level folders:", os.listdir(DATASET_PATH)[:10])
else:
    raise FileNotFoundError("❌ Dataset path not found. Check Google Drive mount.")

print("=== DEBUG END ===\n")

# ======================================================
# EMOTION MAP (RAVDESS STANDARD)
# ======================================================
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ======================================================
# INDEX BUILDER
# ======================================================
def build_ravdess_index(dataset_path, output_csv):
    rows = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if not file.lower().endswith(".mp4"):
                continue

            parts = file.split("-")
            if len(parts) < 7:
                continue  # safety check

            emotion_code = parts[2]
            actor_id = parts[-1].replace(".mp4", "")
            emotion = emotion_map.get(emotion_code, "unknown")

            root_lower = root.lower()
            if "speech" in root_lower:
                modality = "speech"
            elif "song" in root_lower:
                modality = "song"
            else:
                modality = "unknown"

            rows.append({
                "video_path": os.path.join(root, file),
                "emotion": emotion,
                "modality": modality,
                "actor_id": actor_id
            })

    # ==================================================
    # WRITE CSV
    # ==================================================
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "emotion", "modality", "actor_id"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("✅ Index created:", output_csv)
    print("✅ Total samples indexed:", len(rows))


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    build_ravdess_index(DATASET_PATH, OUTPUT_CSV)
