from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.models.fusion_predictor import (
    load_fusion_model,
    load_default_probability_files,
)
from src.utils.config import LABEL_ORDER


def _get_feature_display_name(feature_name: str) -> tuple[str, str]:
    """
    Converts feature name into modality and emotion.
    Example:
    video_prob_happy -> Video, happy
    audio_prob_sad -> Audio, sad
    """
    if feature_name.startswith("video_prob_"):
        return "Video", feature_name.replace("video_prob_", "")

    if feature_name.startswith("audio_prob_"):
        return "Audio", feature_name.replace("audio_prob_", "")

    return "Unknown", feature_name


def _extract_class_shap_values(shap_values, predicted_class_index: int):
    """
    Handles different SHAP output formats safely.
    Multiclass LogisticRegression may return:
    - list[class] of arrays
    - ndarray with shape (samples, features, classes)
    - ndarray with shape (samples, features)
    """
    if isinstance(shap_values, list):
        return shap_values[predicted_class_index][0]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        # shape can be (samples, features, classes)
        return shap_values[0, :, predicted_class_index]

    if shap_values.ndim == 2:
        return shap_values[0]

    raise ValueError(f"Unsupported SHAP values shape: {shap_values.shape}")


def explain_sample_with_shap(sample_id: str, split: str = "test", background_size: int = 500):
    """
    Explains one sample prediction using SHAP.

    SHAP explains the stacking logistic regression emotion prediction.
    Engagement is then explained using the rule-based emotion-to-engagement mapping.
    """
    model_package = load_fusion_model()
    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

    # Use training split as SHAP background because model learned from training data.
    train_df = load_default_probability_files(split="train")
    selected_df = load_default_probability_files(split=split)

    selected_row_df = selected_df[
        selected_df["sample_id"].astype(str) == str(sample_id)
    ].copy()

    if selected_row_df.empty:
        raise ValueError(f"Sample ID not found in {split} split: {sample_id}")

    # Use a sample of training data as background for speed
    background_df = train_df.sample(
        n=min(background_size, len(train_df)),
        random_state=42
    )

    X_background = background_df[feature_columns]
    X_sample = selected_row_df[feature_columns]

    predicted_emotion = model.predict(X_sample)[0]
    predicted_probs = model.predict_proba(X_sample)[0]
    model_classes = list(model.classes_)
    predicted_class_index = model_classes.index(predicted_emotion)

    explainer = shap.LinearExplainer(model, X_background)
    explanation = explainer(X_sample)

    shap_values_for_class = _extract_class_shap_values(
        explanation.values,
        predicted_class_index
    )

    feature_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "shap_value": shap_values_for_class,
            "abs_shap_value": np.abs(shap_values_for_class),
            "feature_value": X_sample.iloc[0].values,
        }
    )

    feature_df[["modality", "emotion"]] = feature_df["feature"].apply(
        lambda x: pd.Series(_get_feature_display_name(x))
    )

    total_abs = feature_df["abs_shap_value"].sum()

    if total_abs == 0:
        feature_df["contribution_percentage"] = 0.0
    else:
        feature_df["contribution_percentage"] = (
            feature_df["abs_shap_value"] / total_abs * 100
        )

    feature_df["effect"] = feature_df["shap_value"].apply(
        lambda x: "Supports prediction" if x > 0 else "Reduces prediction"
    )

    feature_df = feature_df.sort_values(
        by="abs_shap_value",
        ascending=False
    ).reset_index(drop=True)

    # Group by emotion across audio + video
    emotion_df = (
        feature_df.groupby("emotion", as_index=False)["abs_shap_value"]
        .sum()
        .sort_values(by="abs_shap_value", ascending=False)
    )

    emotion_total = emotion_df["abs_shap_value"].sum()

    if emotion_total == 0:
        emotion_df["contribution_percentage"] = 0.0
    else:
        emotion_df["contribution_percentage"] = (
            emotion_df["abs_shap_value"] / emotion_total * 100
        )

    # Group by modality
    modality_df = (
        feature_df.groupby("modality", as_index=False)["abs_shap_value"]
        .sum()
        .sort_values(by="abs_shap_value", ascending=False)
    )

    modality_total = modality_df["abs_shap_value"].sum()

    if modality_total == 0:
        modality_df["contribution_percentage"] = 0.0
    else:
        modality_df["contribution_percentage"] = (
            modality_df["abs_shap_value"] / modality_total * 100
        )

    probability_df = pd.DataFrame(
        {
            "emotion": model_classes,
            "final_model_probability": predicted_probs,
        }
    ).sort_values(by="final_model_probability", ascending=False)

    return {
        "sample_id": sample_id,
        "split": split,
        "predicted_emotion": predicted_emotion,
        "probability_df": probability_df,
        "feature_contributions": feature_df,
        "emotion_contributions": emotion_df,
        "modality_contributions": modality_df,
    }