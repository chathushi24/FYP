import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.config import EMOTION_MAP


def parse_ravdess_filename(file_path: str) -> dict:
    path = Path(file_path)
    sample_id = path.stem
    parts = sample_id.split("-")

    if len(parts) != 7:
        return {
            "sample_id": sample_id,
            "emotion": "unknown",
            "modality_type": "unknown",
        }

    modality_code = parts[0]
    emotion_code = parts[2]

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


def extract_wav_from_mp4(
    video_path: str,
    output_wav_path: str,
    sample_rate: int = 16000,
):
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_wav_path),
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def build_audio_from_video_index(
    ravdess_root: str,
    output_audio_dir: str,
    index_csv_path: str,
    speech_only: bool = True,
):
    mp4_files = find_mp4_files(ravdess_root)

    rows = []

    for video_path in tqdm(mp4_files, desc="Extracting audio from MP4"):
        metadata = parse_ravdess_filename(video_path)

        if speech_only and metadata["modality_type"] != "speech":
            continue

        output_wav = Path(output_audio_dir) / f"{metadata['sample_id']}.wav"

        try:
            extract_wav_from_mp4(
                video_path=video_path,
                output_wav_path=str(output_wav),
            )

            rows.append(
                {
                    "sample_id": metadata["sample_id"],
                    "video_path": video_path,
                    "audio_path": str(output_wav),
                    "emotion": metadata["emotion"],
                    "modality_type": metadata["modality_type"],
                }
            )

        except Exception as error:
            print("Failed:", video_path, "|", error)

    df = pd.DataFrame(rows)
    Path(index_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(index_csv_path, index=False)

    print("Saved audio index:", index_csv_path)
    print("Rows:", len(df))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ravdess-root", required=True)
    parser.add_argument("--output-audio-dir", default="data/processed/speech_wavs_16k")
    parser.add_argument("--index-csv", default="data/processed/extracted_audio_from_mp4_index.csv")
    parser.add_argument("--include-song", action="store_true")

    args = parser.parse_args()

    build_audio_from_video_index(
        ravdess_root=args.ravdess_root,
        output_audio_dir=args.output_audio_dir,
        index_csv_path=args.index_csv,
        speech_only=not args.include_song,
    )


if __name__ == "__main__":
    main()