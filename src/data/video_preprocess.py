import os
import cv2
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.utils import config


# --- Face detector (OpenCV Haar) ---
# This is the easiest to install. Good enough for prototype.
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def video_id_from_path(video_path: str) -> str:
    # e.g., 01-01-02-02-01-01-24.mp4 -> 01-01-02-02-01-01-24
    return os.path.splitext(os.path.basename(video_path))[0]


def extract_faces_from_video(video_path: str, out_dir: str, every_n_frames: int, min_size: int):
    """
    Extract face crops from a video and save as jpg.
    Returns list of saved jpg paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    saved_paths = []
    vid_id = video_id_from_path(video_path)
    frame_idx = 0
    face_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n_frames != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # pick the largest face if multiple
        if len(faces) > 0:
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            x, y, w, h = faces[0]

            if w >= min_size and h >= min_size:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (config.IMG_SIZE, config.IMG_SIZE))

                out_path = os.path.join(out_dir, f"{vid_id}_f{face_idx:04d}.jpg")
                cv2.imwrite(out_path, face)
                saved_paths.append(out_path)
                face_idx += 1

        frame_idx += 1

    cap.release()
    return saved_paths


def build_faces_labeled(video_index_csv: str, out_faces_csv: str, out_faces_dir: str):
    """
    Creates faces_labeled.csv with columns:
      face_path, emotion, video_path, video_id
    """
    ensure_dir(out_faces_dir)

    df = pd.read_csv(video_index_csv)
    # Keep only mp4 rows
    df = df[df["path"].str.lower().str.endswith(".mp4")].copy()

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extracting faces"):
        video_path = r["path"]
        emotion = r["emotion"]
        vid_id = video_id_from_path(video_path)

        face_paths = extract_faces_from_video(
            video_path=video_path,
            out_dir=out_faces_dir,
            every_n_frames=config.EVERY_N_FRAMES,
            min_size=config.FACE_MIN_SIZE
        )

        for fp in face_paths:
            rows.append({
                "face_path": fp,
                "emotion": emotion,
                "video_path": video_path,
                "video_id": vid_id
            })

    out_df = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(out_faces_csv))
    out_df.to_csv(out_faces_csv, index=False)
    print(f"✅ Saved: {out_faces_csv}  rows={len(out_df)}")
    return out_df


def split_faces_by_video_id(faces_csv: str, out_train: str, out_val: str, out_test: str):
    """
    Split by video_id to avoid leakage.
    """
    df = pd.read_csv(faces_csv)
    video_ids = df["video_id"].unique()

    # 70/15/15 split on video ids
    train_vids, temp_vids = train_test_split(
        video_ids,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED,
        shuffle=True
    )

    # split temp into val/test
    val_size = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_vids, test_vids = train_test_split(
        temp_vids,
        test_size=(1 - val_size),
        random_state=config.RANDOM_SEED,
        shuffle=True
    )

    train_df = df[df["video_id"].isin(train_vids)].copy()
    val_df   = df[df["video_id"].isin(val_vids)].copy()
    test_df  = df[df["video_id"].isin(test_vids)].copy()

    ensure_dir(os.path.dirname(out_train))
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)

    print("✅ Split saved:")
    print("  ", out_train, "rows=", len(train_df), "videos=", len(train_vids))
    print("  ", out_val,   "rows=", len(val_df),   "videos=", len(val_vids))
    print("  ", out_test,  "rows=", len(test_df),  "videos=", len(test_vids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", default=config.RAVDESS_INDEX_CSV)
    parser.add_argument("--faces_csv", default=config.FACES_LABELED_CSV)
    parser.add_argument("--faces_dir", default=config.VIDEO_FACES_DIR)
    args = parser.parse_args()

    # 1) build faces_labeled.csv
    build_faces_labeled(args.index_csv, args.faces_csv, args.faces_dir)

    # 2) split into train/val/test
    split_faces_by_video_id(
        faces_csv=args.faces_csv,
        out_train=config.FACES_TRAIN_CSV,
        out_val=config.FACES_VAL_CSV,
        out_test=config.FACES_TEST_CSV
    )


if __name__ == "__main__":
    main()
