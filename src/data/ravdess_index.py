import os
import csv
import argparse

from src.utils import config

print("CONFIG LOADED FROM:", config.__file__)
print("HAS DATASET_ROOT?:", hasattr(config, "DATASET_ROOT"))
if hasattr(config, "DATASET_ROOT"):
    print("DATASET_ROOT =", config.DATASET_ROOT)


EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

def infer_modality_from_root(root: str) -> str:
    r = root.lower()
    # your folders include: Video_Speech_..., Video_Song_..., Audio_Speech_..., Audio_Song_...
    if "speech" in r:
        return "speech"
    if "song" in r:
        return "song"
    return "unknown"

def build_ravdess_index(dataset_path: str, output_csv: str, exts=(".mp4",), verbose=True):
    if verbose:
        print("=== DEBUG START ===")
        print("DATASET_PATH:", dataset_path)
        print("Exists?:", os.path.exists(dataset_path))
        if os.path.exists(dataset_path):
            print("Top-level folders:", os.listdir(dataset_path)[:20])
        print("OUTPUT_CSV:", output_csv)
        print("=== DEBUG END ===\n")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    rows = []
    total_files_seen = 0

    for root, _, files in os.walk(dataset_path):
        for file in files:
            total_files_seen += 1
            low = file.lower()
            if not any(low.endswith(e) for e in exts):
                continue

            parts = file.split("-")
            # RAVDESS filenames are like: 01-01-02-02-01-01-24.mp4
            if len(parts) < 7:
                continue

            emotion_code = parts[2]
            actor_id = parts[-1].split(".")[0]  # "24"
            emotion = EMOTION_MAP.get(emotion_code, "unknown")
            modality = infer_modality_from_root(root)

            rows.append({
                "path": os.path.join(root, file),
                "emotion": emotion,
                "modality": modality,
                "actor_id": actor_id,
            })

    # ensure output folder exists
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "emotion", "modality", "actor_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Index created: {output_csv}")
    print(f"✅ Total indexed samples: {len(rows)}")
    print(f"ℹ️ Total files scanned (all types): {total_files_seen}")
    speech = sum(1 for r in rows if r["modality"] == "speech")
    song = sum(1 for r in rows if r["modality"] == "song")
    unk = sum(1 for r in rows if r["modality"] == "unknown")
    print(f"ℹ️ modality counts -> speech={speech}, song={song}, unknown={unk}")

def main():
    parser = argparse.ArgumentParser()
    # ✅ Use config defaults (Windows local paths)
    parser.add_argument("--dataset", default=config.DATASET_ROOT)
    parser.add_argument("--out", default=config.RAVDESS_INDEX_CSV)
    parser.add_argument("--include_audio", action="store_true", help="Also index .wav files if present")
    args = parser.parse_args()

    exts = (".mp4",)
    if args.include_audio:
        exts = (".mp4", ".wav")

    build_ravdess_index(args.dataset, args.out, exts=exts, verbose=True)

if __name__ == "__main__":
    main()
