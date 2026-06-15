import sys
from pathlib import Path

import pandas as pd
import streamlit as st

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


@st.cache_data(show_spinner=False)
def run_uploaded_video_analysis_cached(
    video_path_str,
    segment_seconds,
    file_size,
    file_mtime,
):
    return process_classroom_video(
        video_path=Path(video_path_str),
        segment_seconds=segment_seconds,
    )


def get_overall_feedback(engagement_level):
    if engagement_level == "Engaged":
        return (
            "Overall classroom engagement is strong. Students appear attentive or actively involved. "
            "Continue discussion-based teaching and use follow-up legal reasoning questions "
            "to sustain participation."
        )

    if engagement_level == "Moderately Engaged":
        return (
            "Overall classroom engagement is moderate. Introduce an interactive activity such as "
            "a short debate prompt, case-law question, quick student response, or practical legal "
            "scenario to increase participation."
        )

    if engagement_level == "Disengaged":
        return (
            "Overall classroom engagement appears low. Shift to a more active learning strategy, "
            "such as role-play, a moot-court style prompt, or a real-world legal problem to regain attention."
        )

    return (
        "Engagement could not be clearly determined. Review the classroom recording quality "
        "and repeat the analysis if needed."
    )


def build_engagement_summary(predictions):
    summary = (
        predictions["predicted_engagement_level_final"]
        .value_counts()
        .reset_index()
    )

    summary.columns = ["engagement_level", "count"]
    summary["percentage"] = (
        summary["count"] / summary["count"].sum() * 100
    ).round(2)

    return summary


def show_overall_predicted_engagement_summary(
    predictions,
    count_label="Analysed Samples",
):
    st.subheader("1. Overall Predicted Engagement Summary")

    summary = build_engagement_summary(predictions)

    st.dataframe(summary, width="stretch")
    st.bar_chart(summary.set_index("engagement_level")["percentage"])

    dominant_engagement = summary.iloc[0]["engagement_level"]
    dominant_percentage = summary.iloc[0]["percentage"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dominant Engagement", dominant_engagement)

    with col2:
        st.metric("Dominant Percentage", f"{dominant_percentage:.2f}%")

    with col3:
        st.metric(count_label, len(predictions))

    st.info(
        "The dominant engagement level represents the overall predicted classroom "
        "engagement across all analysed samples or video segments."
    )

    return summary, dominant_engagement, dominant_percentage


def show_overall_feedback(engagement_level):
    st.subheader("2. Overall Lecturer Feedback")

    feedback = get_overall_feedback(engagement_level)

    if engagement_level == "Engaged":
        st.success(feedback)
    elif engagement_level == "Moderately Engaged":
        st.warning(feedback)
    elif engagement_level == "Disengaged":
        st.error(feedback)
    else:
        st.info(feedback)


def show_dataset_shap_section(selected_sample_id, split):
    st.subheader("3. XAI Explanation using SHAP")

    show_shap = st.checkbox("Show SHAP explanation", value=True)

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
            f"**{shap_result['predicted_emotion']}** for the selected representative sample."
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
        st.warning("SHAP explanation could not be generated.")
        st.exception(error)


def show_upload_xai_section(segment_results):
    st.subheader("3. XAI Explanation using SHAP")

    st.info(
        "For uploaded classroom videos, the explanation is summarised at classroom level "
        "using the final fused emotion probabilities generated for each 5-second segment. "
        "The dataset demo provides the detailed SHAP breakdown for saved evaluation samples."
    )

    final_prob_cols = [f"final_prob_{emotion}" for emotion in LABEL_ORDER]
    available_final_cols = [col for col in final_prob_cols if col in segment_results.columns]

    video_prob_cols = [f"video_prob_{emotion}" for emotion in LABEL_ORDER]
    audio_prob_cols = [f"audio_prob_{emotion}" for emotion in LABEL_ORDER]

    available_video_cols = [col for col in video_prob_cols if col in segment_results.columns]
    available_audio_cols = [col for col in audio_prob_cols if col in segment_results.columns]

    if available_final_cols:
        emotion_summary = pd.DataFrame(
            {
                "emotion": LABEL_ORDER,
                "average_final_probability": [
                    segment_results[f"final_prob_{emotion}"].mean()
                    if f"final_prob_{emotion}" in segment_results.columns
                    else 0.0
                    for emotion in LABEL_ORDER
                ],
            }
        )

        emotion_summary["average_final_probability"] = (
            emotion_summary["average_final_probability"] * 100
        ).round(2)

        st.markdown("#### Overall Emotion Contribution")
        st.dataframe(emotion_summary, width="stretch")
        st.bar_chart(
            emotion_summary.set_index("emotion")["average_final_probability"]
        )

    if available_video_cols and available_audio_cols:
        video_strength = segment_results[available_video_cols].max(axis=1).mean()
        audio_strength = segment_results[available_audio_cols].max(axis=1).mean()

        modality_df = pd.DataFrame(
            {
                "modality": ["Video", "Audio"],
                "average_confidence_strength": [
                    round(video_strength * 100, 2),
                    round(audio_strength * 100, 2),
                ],
            }
        )

        st.markdown("#### Modality Contribution")
        st.dataframe(modality_df, width="stretch")
        st.bar_chart(
            modality_df.set_index("modality")["average_confidence_strength"]
        )

    top_emotions = (
        segment_results["final_predicted_emotion"]
        .value_counts()
        .reset_index()
    )
    top_emotions.columns = ["predicted_emotion", "segment_count"]

    st.markdown("#### Most Frequent Predicted Emotions")
    st.dataframe(top_emotions, width="stretch")


def show_dataset_modality_probability_outputs(sample):
    st.subheader("4. Modality Probability Outputs")

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


def show_upload_modality_probability_outputs(segment_results):
    st.subheader("4. Modality Probability Outputs")

    video_prob_cols = [f"video_prob_{emotion}" for emotion in LABEL_ORDER]
    audio_prob_cols = [f"audio_prob_{emotion}" for emotion in LABEL_ORDER]
    final_prob_cols = [f"final_prob_{emotion}" for emotion in LABEL_ORDER]

    probability_table = pd.DataFrame(
        {
            "emotion": LABEL_ORDER,
            "average_video_probability": [
                segment_results[f"video_prob_{emotion}"].mean()
                if f"video_prob_{emotion}" in segment_results.columns
                else 0.0
                for emotion in LABEL_ORDER
            ],
            "average_audio_probability": [
                segment_results[f"audio_prob_{emotion}"].mean()
                if f"audio_prob_{emotion}" in segment_results.columns
                else 0.0
                for emotion in LABEL_ORDER
            ],
            "average_final_probability": [
                segment_results[f"final_prob_{emotion}"].mean()
                if f"final_prob_{emotion}" in segment_results.columns
                else 0.0
                for emotion in LABEL_ORDER
            ],
        }
    )

    probability_table[
        [
            "average_video_probability",
            "average_audio_probability",
            "average_final_probability",
        ]
    ] = probability_table[
        [
            "average_video_probability",
            "average_audio_probability",
            "average_final_probability",
        ]
    ].round(4)

    st.dataframe(probability_table, width="stretch")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Video")
        st.bar_chart(
            probability_table.set_index("emotion")[["average_video_probability"]]
        )

    with col2:
        st.markdown("#### Audio")
        st.bar_chart(
            probability_table.set_index("emotion")[["average_audio_probability"]]
        )

    with col3:
        st.markdown("#### Final")
        st.bar_chart(
            probability_table.set_index("emotion")[["average_final_probability"]]
        )


def show_all_predictions(predictions):
    with st.expander("View full prediction table"):
        st.dataframe(predictions, width="stretch")


def show_loading_animation():
    return st.empty()


def render_loading_animation(container):
    container.markdown(
        """
        <style>
        .analysis-wrapper {
            background-color: #111827;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 22px;
            margin-top: 18px;
            margin-bottom: 18px;
        }

        .analysis-loader {
            border: 6px solid #334155;
            border-top: 6px solid #38bdf8;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            animation: spin 1s linear infinite;
            display: inline-block;
            vertical-align: middle;
            margin-right: 18px;
        }

        .analysis-text {
            display: inline-block;
            vertical-align: middle;
            color: #e5e7eb;
            font-size: 19px;
            font-weight: 600;
        }

        .analysis-subtext {
            color: #9ca3af;
            font-size: 14px;
            margin-top: 12px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>

        <div class="analysis-wrapper">
            <div class="analysis-loader"></div>
            <div class="analysis-text">
                Analysing classroom engagement using 5-second audio-video segments...
            </div>
            <div class="analysis-subtext">
                The system is processing visual cues, audio cues, fusion prediction, and lecturer feedback.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

        summary, dominant_engagement, dominant_percentage = (
            show_overall_predicted_engagement_summary(
                predictions,
                count_label="Analysed Samples",
            )
        )

        show_overall_feedback(dominant_engagement)

        sample = predictions.iloc[0]
        selected_sample_id = str(sample["sample_id"])

        show_dataset_shap_section(selected_sample_id, split)

        show_dataset_modality_probability_outputs(sample)

        show_all_predictions(predictions)

    except Exception as error:
        st.error("The dataset demo could not load the model/prediction files.")
        st.exception(error)


elif mode == "Upload Classroom Video":
    st.header("Upload Classroom Video Mode")

    st.write(
        "Upload an online law classroom video. The system will split the video into "
        "5-second time windows, analyse available audio and visual cues, predict emotion, "
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
        value=5,
        step=5,
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Analyse Uploaded Video"):
            try:
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

                saved_video_path = UPLOAD_DIR / uploaded_video.name

                with open(saved_video_path, "wb") as file:
                    file.write(uploaded_video.getbuffer())

                file_size = saved_video_path.stat().st_size
                file_mtime = saved_video_path.stat().st_mtime

                loading_container = show_loading_animation()
                render_loading_animation(loading_container)

                progress_bar = st.progress(
                    0,
                    text="Starting classroom engagement analysis...",
                )

                with st.status(
                    "Running multimodal engagement analysis...",
                    expanded=True,
                ) as status:
                    st.write("Step 1/4: Uploaded classroom video saved.")
                    progress_bar.progress(
                        15,
                        text="Video saved successfully.",
                    )

                    st.write("Step 2/4: Loading trained video, audio, and fusion models.")
                    progress_bar.progress(
                        35,
                        text="Loading model pipeline.",
                    )

                    st.write("Step 3/4: Processing 5-second audio-video segments.")
                    progress_bar.progress(
                        55,
                        text="Analysing segment-level engagement cues.",
                    )

                    segment_results, engagement_summary = run_uploaded_video_analysis_cached(
                        str(saved_video_path),
                        segment_seconds,
                        file_size,
                        file_mtime,
                    )

                    st.write("Step 4/4: Generating engagement summary and lecturer feedback.")
                    progress_bar.progress(
                        90,
                        text="Preparing dashboard outputs.",
                    )

                    status.update(
                        label="Multimodal engagement analysis completed.",
                        state="complete",
                        expanded=False,
                    )

                progress_bar.progress(
                    100,
                    text="Analysis completed.",
                )

                loading_container.empty()

                st.success("Video analysis completed.")

                summary, dominant_engagement, dominant_percentage = (
                    show_overall_predicted_engagement_summary(
                        segment_results,
                        count_label="Analysed Segments",
                    )
                )

                show_overall_feedback(dominant_engagement)

                show_upload_xai_section(segment_results)

                show_upload_modality_probability_outputs(segment_results)

                show_all_predictions(segment_results)

            except Exception as error:
                st.error("Uploaded video analysis failed.")
                st.exception(error)