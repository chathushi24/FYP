import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor


SAMPLE_RATE = 16000


def load_fixed_audio(
    audio_path: str,
    sample_rate: int = SAMPLE_RATE,
    target_seconds: float = 3.0,
):
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Trim silence
    waveform, _ = librosa.effects.trim(waveform, top_db=25)

    target_len = int(sample_rate * target_seconds)

    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]

    return waveform.astype(np.float32)


def load_hubert(device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/hubert-base-ls960"
    )

    model = HubertModel.from_pretrained(
        "facebook/hubert-base-ls960"
    )

    model.to(device)
    model.eval()

    return feature_extractor, model, device


def extract_hubert_embedding(
    waveform: np.ndarray,
    feature_extractor,
    model,
    device: str,
):
    inputs = feature_extractor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state

    # Mean pooling over time
    mean_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()

    return mean_embedding


def extract_audio_features(
    audio_csv: str,
    output_x: str,
    output_y: str,
):
    df = pd.read_csv(audio_csv)

    if "audio_path" not in df.columns:
        raise ValueError("audio_csv must contain audio_path column.")

    if "emotion" not in df.columns:
        raise ValueError("audio_csv must contain emotion column.")

    feature_extractor, model, device = load_hubert()

    features = []
    labels = []
    sample_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting HuBERT features"):
        try:
            waveform = load_fixed_audio(row["audio_path"])
            embedding = extract_hubert_embedding(
                waveform=waveform,
                feature_extractor=feature_extractor,
                model=model,
                device=device,
            )

            features.append(embedding)
            labels.append(row["emotion"])
            sample_ids.append(row["sample_id"])

        except Exception as error:
            print("Skipped:", row["audio_path"], "|", error)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    Path(output_x).parent.mkdir(parents=True, exist_ok=True)

    np.save(output_x, X)
    np.save(output_y, y)

    sample_id_path = str(Path(output_x).with_name(Path(output_x).stem + "_sample_ids.csv"))
    pd.DataFrame({"sample_id": sample_ids}).to_csv(sample_id_path, index=False)

    print("Saved X:", output_x, X.shape)
    print("Saved y:", output_y, y.shape)
    print("Saved sample IDs:", sample_id_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio-csv", default="data/processed/extracted_audio_from_mp4_index.csv")
    parser.add_argument("--output-x", default="data/processed/X_audio_hubert.npy")
    parser.add_argument("--output-y", default="data/processed/y_audio_labels.npy")

    args = parser.parse_args()

    extract_audio_features(
        audio_csv=args.audio_csv,
        output_x=args.output_x,
        output_y=args.output_y,
    )


if __name__ == "__main__":
    main()