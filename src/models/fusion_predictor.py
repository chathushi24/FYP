import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.models.engagement_mapping import (
    map_emotion_to_engagement,
    get_feedback_for_engagement,
)
from src.utils.config import (
    LABEL_ORDER,
    VIDEO_PROB_PATH,
    FINAL_AUDIO_PROB_DIR,
    FINAL_FUSION_MODEL_PATH,
)


def prepare_video_probs(video_df: pd.DataFrame) -> pd.DataFrame:
    df = video_df.copy()

    if "sample_id" not in df.columns:
        raise ValueError("Video probability file must contain sample_id.")

    rename_map = {}

    for emotion in LABEL_ORDER:
        possible_cols = [
            f"video_prob_{emotion}",
            f"video_{emotion}",
            f"prob_{emotion}",
            emotion,
        ]

        for col in possible_cols:
            if col in df.columns:
                rename_map[col] = f"video_prob_{emotion}"
                break

    df = df.rename(columns=rename_map)

    required_cols = ["sample_id"] + [f"video_prob_{e}" for e in LABEL_ORDER]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Video probability file missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df[required_cols].groupby("sample_id", as_index=False).mean()


def prepare_audio_probs(audio_df: pd.DataFrame) -> pd.DataFrame:
    df = audio_df.copy()

    if "sample_id" not in df.columns:
        raise ValueError("Audio probability file must contain sample_id.")

    if "emotion" not in df.columns:
        if "label" in df.columns:
            df = df.rename(columns={"label": "emotion"})
        else:
            raise ValueError("Audio probability file must contain emotion or label.")

    rename_map = {}

    for emotion in LABEL_ORDER:
        possible_cols = [
            f"audio_prob_{emotion}",
            f"prob_{emotion}",
            emotion,
        ]

        for col in possible_cols:
            if col in df.columns:
                rename_map[col] = f"audio_prob_{emotion}"
                break

    df = df.rename(columns=rename_map)

    required_cols = ["sample_id", "emotion"] + [f"audio_prob_{e}" for e in LABEL_ORDER]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Audio probability file missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df[required_cols]


def load_fusion_model(model_path: str | Path = FINAL_FUSION_MODEL_PATH):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Fusion model not found: {model_path}")

    package = joblib.load(model_path)

    if isinstance(package, dict):
        model = package["model"]
        feature_columns = package["feature_columns"]
        label_order = package.get("label_order", LABEL_ORDER)
        method = package.get("method", "stacking_logistic_regression")
    else:
        model = package
        feature_columns = (
            [f"video_prob_{e}" for e in LABEL_ORDER]
            + [f"audio_prob_{e}" for e in LABEL_ORDER]
        )
        label_order = LABEL_ORDER
        method = "stacking_logistic_regression"

    return {
        "model": model,
        "feature_columns": feature_columns,
        "label_order": label_order,
        "method": method,
    }


def load_default_probability_files(split: str = "test"):
    video_df = pd.read_csv(VIDEO_PROB_PATH)

    audio_path = FINAL_AUDIO_PROB_DIR / f"final_audio_{split}_probabilities.csv"

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio probability file not found: {audio_path}")

    audio_df = pd.read_csv(audio_path)

    video_clean = prepare_video_probs(video_df)
    audio_clean = prepare_audio_probs(audio_df)

    merged = audio_clean.merge(video_clean, on="sample_id", how="inner")

    return merged


def predict_with_fusion(merged_df: pd.DataFrame, model_package: dict):
    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

    missing = [c for c in feature_columns if c not in merged_df.columns]

    if missing:
        raise ValueError(f"Merged dataframe missing feature columns: {missing}")

    X = merged_df[feature_columns].values

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    model_classes = list(model.classes_)

    output = merged_df.copy()
    output["final_predicted_emotion"] = predictions

    if "emotion" in output.columns:
        output["is_correct_final_emotion"] = (
            output["emotion"] == output["final_predicted_emotion"]
        )

    for emotion in LABEL_ORDER:
        if emotion in model_classes:
            idx = model_classes.index(emotion)
            output[f"stack_prob_{emotion}"] = probabilities[:, idx]
        else:
            output[f"stack_prob_{emotion}"] = 0.0

    output["predicted_engagement_level_final"] = output[
        "final_predicted_emotion"
    ].apply(map_emotion_to_engagement)

    output["logic_based_feedback_final"] = output[
        "predicted_engagement_level_final"
    ].apply(get_feedback_for_engagement)

    return output


def predict_split(split: str = "test"):
    model_package = load_fusion_model()
    merged = load_default_probability_files(split=split)
    predictions = predict_with_fusion(merged, model_package)
    return predictions


def predict_one_sample(sample_id: str, split: str = "test"):
    predictions = predict_split(split=split)

    sample = predictions[predictions["sample_id"].astype(str) == str(sample_id)]

    if sample.empty:
        raise ValueError(f"Sample ID not found: {sample_id}")

    return sample.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--output-csv", default="data/processed/vscode_fusion_predictions.csv")

    args = parser.parse_args()

    predictions = predict_split(split=args.split)

    if args.sample_id:
        sample = predictions[predictions["sample_id"].astype(str) == str(args.sample_id)]

        if sample.empty:
            print("Sample not found:", args.sample_id)
        else:
            print(sample.T)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_csv, index=False)

    print("Saved predictions:", args.output_csv)
    print("Rows:", len(predictions))


if __name__ == "__main__":
    main()