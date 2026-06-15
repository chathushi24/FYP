import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.config import LABEL_ORDER


def load_audio_model(model_path: str):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Audio model not found: {model_path}")

    return joblib.load(model_path)


def predict_audio_probabilities(
    model,
    X: np.ndarray,
    sample_ids: list[str],
    audio_paths: list[str],
    emotions: list[str],
    output_csv: str,
    split_name: str = "demo",
):
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    model_classes = list(model.classes_)

    output = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "audio_path": audio_paths,
            "emotion": emotions,
        }
    )

    for emotion in LABEL_ORDER:
        if emotion in model_classes:
            idx = model_classes.index(emotion)
            output[f"audio_prob_{emotion}"] = probabilities[:, idx]
        else:
            output[f"audio_prob_{emotion}"] = 0.0

    output["audio_predicted_emotion"] = predictions
    output["audio_model"] = "final_selected_audio_model"
    output["split"] = split_name

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    print("Saved audio probabilities:", output_csv)
    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--x-path", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="demo")

    args = parser.parse_args()

    model = load_audio_model(args.model_path)
    X = np.load(args.x_path)
    metadata = pd.read_csv(args.metadata_csv)

    required_cols = ["sample_id", "audio_path", "emotion"]
    missing = [c for c in required_cols if c not in metadata.columns]

    if missing:
        raise ValueError(f"metadata_csv is missing columns: {missing}")

    predict_audio_probabilities(
        model=model,
        X=X,
        sample_ids=metadata["sample_id"].astype(str).tolist(),
        audio_paths=metadata["audio_path"].astype(str).tolist(),
        emotions=metadata["emotion"].astype(str).tolist(),
        output_csv=args.output_csv,
        split_name=args.split,
    )


if __name__ == "__main__":
    main()