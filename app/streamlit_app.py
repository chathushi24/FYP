import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
st.title("✅ Streamlit Loaded")
st.write("If you see this, the script is running.")

import os
import numpy as np
import streamlit as st
import joblib

from src.models.fusion_random_forest import predict_fused
from src.explainability.shap_explain import explain_rf

DEFAULT_LABELS = ["Engaged", "Moderately Engaged", "Disengaged"]

PROCESSED = os.path.join("src", "data", "processed")

VIDEO_MODEL = os.path.join(PROCESSED, "rf_video_model.pkl")
AUDIO_MODEL = os.path.join(PROCESSED, "rf_audio_model.pkl")

X_TEST_VIDEO = os.path.join(PROCESSED, "X_test_video.npy")


# If you want HuBERT test features, use your actual file name here:
# Example from your screenshot:
# hubert_v2_noaug_20260106_095112_X_test.npy
HUBERT_X_TEST = os.path.join(PROCESSED, "hubert_v2_noaug_20260106_095112_X_test.npy")


st.set_page_config(page_title="PoC - Engagement Detection", layout="wide")
st.title("✅ Proof of Concept (PoC): Multimodal Engagement Detection")

# Load models
video_model = joblib.load(VIDEO_MODEL)
audio_model = joblib.load(AUDIO_MODEL)

st.success("Models loaded successfully ✅")
st.write("Video model:", VIDEO_MODEL)
st.write("Audio model:", AUDIO_MODEL)

# Load features
Xv = np.load(X_TEST_VIDEO)
Xa = np.load(HUBERT_X_TEST) if os.path.exists(HUBERT_X_TEST) else None

if Xa is None:
    st.warning("HuBERT X_test not found. Using video-only demo.")
    st.stop()

idx = st.slider("Pick a test sample", 0, min(len(Xv), len(Xa)) - 1, 0)

Xv1 = Xv[idx].reshape(1, -1)
Xa1 = Xa[idx].reshape(1, -1)

w_video = st.slider("Video weight", 0.0, 1.0, 0.5, 0.05)

pred_label, fused_probs, v_probs, a_probs = predict_fused(video_model, audio_model, Xv1, Xa1, w_video=w_video)

st.subheader("📌 Prediction")
st.metric("Fused Engagement", pred_label)

st.write("Video probs:", dict(zip(DEFAULT_LABELS, v_probs)))
st.write("Audio probs:", dict(zip(DEFAULT_LABELS, a_probs)))
st.write("Fused probs:", dict(zip(DEFAULT_LABELS, fused_probs)))

st.subheader("🔍 Explainability (Video RF)")
exp = explain_rf(video_model, Xv1, top_k=15)
st.write("Method:", exp["method"])
st.write("Top feature indices:", exp["top_features"])
st.write("Importance/SHAP values:", exp["values"])

st.subheader("🧠 Feedback")
if pred_label == "Engaged":
    st.success("✅ Students are engaged — continue deeper questioning & case-based discussion.")
elif pred_label == "Moderately Engaged":
    st.warning("⚠️ Moderate engagement — add a short debate question / interactive activity.")
else:
    st.error("❌ Disengaged — switch to role-play / direct questioning to regain attention.")
