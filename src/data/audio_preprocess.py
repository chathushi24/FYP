import os
import argparse
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def extract_wav_from_mp4(mp4_path: str, wav_path: str, sr: int = 16000) -> None:
    """
    Extract mono 16kHz PCM wav from an mp4 using ffmpeg.
    """
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", mp4_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-acodec", "pcm_s16le",
        wav_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for: {mp4_path}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Error:\n{res.stderr[:2000]}"
        )


def build_audio_csvs(
    input_csv: str,
    out_dir: str,
    wav_dir: str,
    sr: int = 16000,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    video_col: str = "video_path",
    label_col: str = "label",
) -> None:
    """
    Input CSV must contain:
      - video_path (mp4 path)
      - label (emotion label or engagement label, whatever you trained)

    Outputs:
      - audio_train.csv, audio_val.csv, audio_test.csv
    with columns:
      - audio_path, label
    """
    out_dir = Path(out_dir)
    wav_dir = Path(wav_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    assert video_col in df.columns, f"Missing column '{video_col}' in {input_csv}"
    assert label_col in df.columns, f"Missing column '{label_col}' in {input_csv}"

    # extract wavs
    audio_paths = []
    for i, row in df.iterrows():
        mp4_path = str(row[video_col])
        if not os.path.exists(mp4_path):
            raise FileNotFoundError(f"Missing mp4: {mp4_path}")

        wav_name = Path(mp4_path).stem + "_fixed.wav"
        wav_path = str(wav_dir / wav_name)

        if not os.path.exists(wav_path):
            extract_wav_from_mp4(mp4_path, wav_path, sr=sr)

        audio_paths.append(wav_path)

    out_df = pd.DataFrame({
        "audio_path": audio_paths,
        "label": df[label_col].astype(str).tolist()
    })

    # Split: train / val / test (stratified)
    train_df, test_df = train_test_split(
        out_df,
        test_size=test_size,
        random_state=seed,
        stratify=out_df["label"]
    )

    # val_size is fraction of TOTAL; convert to fraction of remaining train_df
    val_ratio = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_df["label"]
    )

    train_path = out_dir / "audio_train.csv"
    val_path = out_dir / "audio_val.csv"
    test_path = out_dir / "audio_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("✅ Saved:")
    print(" -", train_path)
    print(" -", val_path)
    print(" -", test_path)
    print("Counts:", len(train_df), len(val_df), len(test_df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="CSV with video_path,label")
    ap.add_argument("--out_dir", required=True, help="Where to save audio_train/val/test CSVs")
    ap.add_argument("--wav_dir", required=True, help="Where to save extracted wav files")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--video_col", default="video_path")
    ap.add_argument("--label_col", default="label")
    args = ap.parse_args()

    build_audio_csvs(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        wav_dir=args.wav_dir,
        sr=args.sr,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        video_col=args.video_col,
        label_col=args.label_col,
    )


if __name__ == "__main__":
    main()
