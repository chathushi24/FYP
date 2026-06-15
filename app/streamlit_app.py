import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow app/streamlit_app.py to import from src/
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.models.fusion_predictor import predict_split
from src.utils.config import LABEL_ORDER, UPLOAD_DIR
from src.explainability.shap_explain import explain_sample_with_shap
from src.inference.video_upload_pipeline import process_classroom_video


st.set_page_config(
    page_title="Multimodal Student Engagement Detection",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Explainable Multimodal AI for Student Engagement Detection")
st.caption(
    "Audio + Video decision-level fusion using Stacking Logistic Regression, "
    "SHAP explainability, and rule-based teaching feedback."
)

st.sidebar.header("System Mode")

mode = st.sidebar.radio(
    "Choose demo mode",
    [
        "Dataset Demo",
        "Upload Classroom Video",
    ],
)

st.sidebar.markdown("---")


@st.cache_data
def load_predictions(selected_split):
    return predict_split(split=selected_split)


def show_prediction_header(sample):
    st.subheader("1. Prediction Result")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("True Emotion", sample.get("emotion", "N/A"))

    with col2:
        st.metric("Predicted Emotion", sample["final_predicted_emotion"])

    with col3:
        st.metric("Engagement Level", sample["predicted_engagement_level_final"])

    stack_cols = [f"stack_prob_{e}" for e in LABEL_ORDER]
    confidence = max([sample[col] for col in stack_cols])

    with col4:
        st.metric("Confidence Score", f"{confidence * 100:.2f}%")


def show_probability_outputs(sample):
    st.subheader("2. Modality Probability Outputs")

    video_cols = [f"video_prob_{e}" for e in LABEL_ORDER]
    audio_cols = [f"audio_prob_{e}" for e in LABEL_ORDER]
    stack_cols = [f"stack_prob_{e}" for e in LABEL_ORDER]

    probability_table = pd.DataFrame(
        {
            "emotion": LABEL_ORDER,
            "video_probability": [sample[col] for col in video_cols],
            "audio_probability": [sample[col] for col in audio_cols],
            "final_fusion_probability": [sample[col] for col in stack_cols],
        }
    )

    st.dataframe(probability_table, width="stretch")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Video Emotion Probabilities")
        st.bar_chart(probability_table.set_index("emotion")[["video_probability"]])

    with col2:
        st.markdown("#### Audio Emotion Probabilities")
        st.bar_chart(probability_table.set_index("emotion")[["audio_probability"]])

    with col3:
        st.markdown("#### Final Fused Emotion Probabilities")
        st.bar_chart(
            probability_table.set_index("emotion")[["final_fusion_probability"]]
        )


def show_shap_section(selected_sample_id, split):
    st.subheader("3. XAI Explanation using SHAP")

    show_shap = st.checkbox("Show SHAP explanation for this sample", value=True)

    if not show_shap:
        return

    try:
        shap_result = explain_sample_with_shap(
            sample_id=selected_sample_id,
            split=split,
            background_size=500,
        )

        st.write(
            f"SHAP explains why the fusion model predicted "
            f"**{shap_result['predicted_emotion']}** for this sample."
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Modality Contribution")
            modality_df = shap_result["modality_contributions"].copy()
            modality_df["contribution_percentage"] = modality_df[
                "contribution_percentage"
            ].round(2)

            st.dataframe(modality_df, width="stretch")
            st.bar_chart(
                modality_df.set_index("modality")["contribution_percentage"]
            )

        with col_b:
            st.markdown("#### Emotion Contribution")
            emotion_df = shap_result["emotion_contributions"].copy()
            emotion_df["contribution_percentage"] = emotion_df[
                "contribution_percentage"
            ].round(2)

            st.dataframe(emotion_df, width="stretch")
            st.bar_chart(
                emotion_df.set_index("emotion")["contribution_percentage"]
            )

        st.markdown("#### Top 5 Contributing Features")

        feature_df = shap_result["feature_contributions"].copy()

        feature_df["shap_value"] = feature_df["shap_value"].round(4)
        feature_df["feature_value"] = feature_df["feature_value"].round(4)
        feature_df["contribution_percentage"] = feature_df[
            "contribution_percentage"
        ].round(2)

        st.dataframe(
            feature_df[
                [
                    "feature",
                    "modality",
                    "emotion",
                    "feature_value",
                    "shap_value",
                    "contribution_percentage",
                    "effect",
                ]
            ].head(5),
            width="stretch",
        )

        st.info(
            "SHAP explains the final emotion prediction made by the stacking "
            "logistic regression fusion model. The engagement level is then assigned "
            "using the transparent rule-based emotion-to-engagement mapping."
        )

    except Exception as error:
        st.warning("SHAP explanation could not be generated for this sample.")
        st.exception(error)


def show_feedback_section(sample):
    st.subheader("4. Lecturer Feedback")

    engagement_level = sample["predicted_engagement_level_final"]
    feedback = sample["logic_based_feedback_final"]

    if engagement_level == "Engaged":
        st.success(feedback)
    elif engagement_level == "Moderately Engaged":
        st.warning(feedback)
    elif engagement_level == "Disengaged":
        st.error(feedback)
    else:
        st.info(feedback)


def show_engagement_summary(predictions):
    st.subheader("5. Engagement Summary")

    summary = (
        predictions["predicted_engagement_level_final"]
        .value_counts()
        .reset_index()
    )

    summary.columns = ["engagement_level", "count"]
    summary["percentage"] = (
        summary["count"] / summary["count"].sum() * 100
    ).round(2)

    st.dataframe(summary, width="stretch")

    st.bar_chart(summary.set_index("engagement_level")["percentage"])


def show_all_predictions(predictions):
    with st.expander("View All Predictions"):
        st.dataframe(
            predictions[
                [
                    "sample_id",
                    "emotion",
                    "final_predicted_emotion",
                    "predicted_engagement_level_final",
                    "logic_based_feedback_final",
                ]
            ],
            width="stretch",
        )


def get_overall_feedback(engagement_level):
    if engagement_level == "Engaged":
        return (
            "Overall classroom engagement is strong. Continue the current teaching flow, "
            "maintain discussion-based interaction, and use follow-up legal reasoning "
            "questions to sustain participation."
        )

    if engagement_level == "Moderately Engaged":
        return (
            "Overall classroom engagement is moderate. Introduce an interactive activity "
            "such as a short debate prompt, case-law question, quick student response, "
            "or practical legal scenario to increase participation."
        )

    if engagement_level == "Disengaged":
        return (
            "Overall classroom engagement appears low. Shift to a more active learning "
            "strategy, such as role-play, a moot-court style prompt, or a real-world "
            "legal problem to regain attention."
        )

    return (
        "Engagement could not be clearly determined. Review the classroom recording "
        "quality and consider using a clearer audio/video sample."
    )


def show_overall_feedback(engagement_level):
    st.subheader("3. Overall Lecturer Feedback")

    feedback = get_overall_feedback(engagement_level)

    if engagement_level == "Engaged":
        st.success(feedback)
    elif engagement_level == "Moderately Engaged":
        st.warning(feedback)
    elif engagement_level == "Disengaged":
        st.error(feedback)
    else:
        st.info(feedback)


if mode == "Dataset Demo":
    st.header("Dataset Demo Mode")

    st.write(
        "This mode uses saved audio/video probability outputs from the RAVDESS-based "
        "evaluation pipeline and applies the final stacking fusion model."
    )

    split = st.sidebar.selectbox(
        "Select dataset split",
        ["test", "validation", "train"],
        index=0,
    )

    try:
        predictions = load_predictions(split)

        st.success(
            f"Loaded {len(predictions)} fused predictions from the {split} split."
        )

        sample_ids = predictions["sample_id"].astype(str).tolist()

        selected_sample_id = st.sidebar.selectbox(
            "Select sample ID",
            sample_ids,
        )

        sample = predictions[
            predictions["sample_id"].astype(str) == selected_sample_id
        ].iloc[0]

        show_prediction_header(sample)
        show_probability_outputs(sample)
        show_shap_section(selected_sample_id, split)
        show_feedback_section(sample)
        show_engagement_summary(predictions)
        show_all_predictions(predictions)

    except Exception as error:
        st.error("The dataset demo could not load the model/prediction files.")
        st.exception(error)


elif mode == "Upload Classroom Video":
    st.header("Upload Classroom Video Mode")

    st.write(
        "Upload an online law classroom video. The system will split the video into "
        "short time windows, analyse available audio and visual cues, predict emotion, "
        "map the emotion into an engagement level, and generate lecturer feedback."
    )

    uploaded_video = st.file_uploader(
        "Upload a classroom video file",
        type=["mp4", "mov", "avi", "mkv"],
    )

    segment_seconds = st.sidebar.slider(
        "Segment length for analysis",
        min_value=5,
        max_value=30,
        value=30,
        step=5,
    )

    st.info(
        "Recommended first test: use a short 20–30 second video. "
        "For longer videos, use 30-second segments to reduce processing time."
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Analyse Uploaded Video"):
            try:
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

                saved_video_path = UPLOAD_DIR / uploaded_video.name

                with open(saved_video_path, "wb") as file:
                    file.write(uploaded_video.getbuffer())

                with st.spinner(
                    "Processing video. This may take a few minutes, "
                    "especially during the first run..."
                ):
                    segment_results, engagement_summary = process_classroom_video(
                        video_path=saved_video_path,
                        segment_seconds=segment_seconds,
                    )

                st.success("Video analysis completed.")

                st.subheader("1. Overall Engagement Summary")

                st.dataframe(engagement_summary, width="stretch")

                st.bar_chart(
                    engagement_summary.set_index("engagement_level")["percentage"]
                )

                dominant_engagement = engagement_summary.iloc[0]["engagement_level"]
                dominant_percentage = engagement_summary.iloc[0]["percentage"]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Dominant Engagement", dominant_engagement)

                with col2:
                    st.metric("Dominant Percentage", f"{dominant_percentage:.2f}%")

                with col3:
                    st.metric("Analysed Segments", len(segment_results))

                st.info(
                    "The dominant engagement level represents the overall classroom "
                    "engagement across all analysed video segments."
                )

                st.subheader("2. Engagement Timeline")

                timeline_view = segment_results[
                    [
                        "segment_id",
                        "start_time",
                        "end_time",
                        "video_status",
                        "audio_status",
                        "prediction_mode",
                        "final_predicted_emotion",
                        "predicted_engagement_level_final",
                        "confidence",
                    ]
                ].copy()

                timeline_view["confidence"] = (
                    timeline_view["confidence"] * 100
                ).round(2)

                st.dataframe(timeline_view, width="stretch")

                show_overall_feedback(dominant_engagement)

                with st.expander("View technical segment prediction table"):
                    st.dataframe(segment_results, width="stretch")

            except Exception as error:
                st.error("Uploaded video analysis failed.")
                st.exception(error)