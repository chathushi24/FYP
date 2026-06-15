import argparse
import os
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.utils.config import EMOTION_MAP


CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)


def parse_ravdess_filename(file_path: str) -> dict:
    """
    RAVDESS filename format:
    01-01-03-01-01-01-01.mp4

    parts:
    modality, vocal_channel, emotion, intensity, statement, repetition, actor
    """
    path = Path(file_path)
    sample_id = path.stem
    parts = sample_id.split("-")

    if len(parts) != 7:
        return {
            "sample_id": sample_id,
            "emotion": "unknown",
            "modality_type": "unknown",
        }

    emotion_code = parts[2]
    modality_code = parts[0]

    emotion = EMOTION_MAP.get(emotion_code, "unknown")
    modality_type = "speech" if modality_code == "01" else "song"

    return {
        "sample_id": sample_id,
        "emotion": emotion,
        "modality_type": modality_type,
    }


def find_mp4_files(ravdess_root: str) -> list[str]:
    root = Path(ravdess_root)
    return [str(p) for p in root.rglob("*.mp4")]


def extract_faces_from_video(
    video_path: str,
    output_dir: str,
    sample_fps: int = 1,
    max_faces: int = 5,
    image_size: int = 224,
) -> list[str]:
    """
    Extract face crops from a video file.
    Saves up to max_faces face images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    video_id = Path(video_path).stem
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 25

    step = max(int(fps / sample_fps), 1)

    saved_paths = []
    frame_idx = 0
    face_idx = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(faces) > 0:
            # Choose largest face
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

            face = frame[y : y + h, x : x + w]
            face = cv2.resize(face, (image_size, image_size))

            out_path = output_dir / f"{video_id}_face_{face_idx}.jpg"
            cv2.imwrite(str(out_path), face)

            saved_paths.append(str(out_path))
            face_idx += 1

            if face_idx >= max_faces:
                break

        frame_idx += 1

    cap.release()
    return saved_paths


def build_video_face_index(
    ravdess_root: str,
    output_dir: str,
    index_csv_path: str,
    sample_fps: int = 1,
    max_faces: int = 5,
):
    mp4_files = find_mp4_files(ravdess_root)

    rows = []

    for video_path in tqdm(mp4_files, desc="Extracting video faces"):
        metadata = parse_ravdess_filename(video_path)

        face_paths = extract_faces_from_video(
            video_path=video_path,
            output_dir=output_dir,
            sample_fps=sample_fps,
            max_faces=max_faces,
        )

        for face_path in face_paths:
            rows.append(
                {
                    "sample_id": metadata["sample_id"],
                    "video_path": video_path,
                    "face_path": face_path,
                    "emotion": metadata["emotion"],
                    "modality_type": metadata["modality_type"],
                }
            )

    df = pd.DataFrame(rows)
    Path(index_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(index_csv_path, index=False)

    print("Saved video face index:", index_csv_path)
    print("Rows:", len(df))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ravdess-root", required=True)
    parser.add_argument("--output-dir", default="data/processed/video_faces")
    parser.add_argument("--index-csv", default="data/processed/video_faces_index.csv")
    parser.add_argument("--sample-fps", type=int, default=1)
    parser.add_argument("--max-faces", type=int, default=5)

    args = parser.parse_args()

    build_video_face_index(
        ravdess_root=args.ravdess_root,
        output_dir=args.output_dir,
        index_csv_path=args.index_csv,
        sample_fps=args.sample_fps,
        max_faces=args.max_faces,
    )


if __name__ == "__main__":
    main()