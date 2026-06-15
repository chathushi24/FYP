from pathlib import Path

# Root project folder: FYP/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

NOTEBOOK_DIR = DATA_DIR / "notebooks"

VIDEO_PROB_PATH = PROCESSED_DIR / "video_sample_probabilities_all.csv"

FINAL_AUDIO_PROB_DIR = PROCESSED_DIR / "final_audio_probabilities"

FUSION_EXPERIMENT_DIR = PROCESSED_DIR / "fusion_experiments"
FINAL_FUSION_MODEL_DIR = FUSION_EXPERIMENT_DIR / "final_stacking_fusion_model"

FINAL_FUSION_MODEL_PATH = (
    FINAL_FUSION_MODEL_DIR / "stacking_logistic_regression_fusion_model.joblib"
)

FINAL_ENGAGEMENT_TEST_CSV = (
    FINAL_FUSION_MODEL_DIR / "final_engagement_predictions_literature_aligned_test.csv"
)

LABEL_ORDER = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

VIDEO_MODEL_PATH = PROCESSED_DIR / "rf_video_model.pkl"

FINAL_AUDIO_MODEL_DIR = PROCESSED_DIR / "final_audio_model"
FINAL_AUDIO_MODEL_PATH = FINAL_AUDIO_MODEL_DIR / "final_selected_audio_model.joblib"
FINAL_AUDIO_METADATA_PATH = FINAL_AUDIO_MODEL_DIR / "final_selected_audio_model_metadata.json"

UPLOAD_DIR = DATA_DIR / "uploads"
TEMP_SEGMENT_DIR = PROCESSED_DIR / "uploaded_video_segments"