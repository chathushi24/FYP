import os
import uuid
from pathlib import Path

import cv2
import joblib
import librosa
import numpy as np
import pandas as pd
import torch

from src.models.engagement_mapping import (
    map_emotion_to_engagement,
    get_feedback_for_engagement,
)
from src.models.fusion_predictor import load_fusion_model
from src.utils.config import (
    LABEL_ORDER,
    VIDEO_MODEL_PATH,
    FINAL_AUDIO_MODEL_PATH,
    UPLOAD_DIR,
    TEMP_SEGMENT_DIR,
)


SAMPLE_RATE = 16000
AUDIO_WINDOW_SECONDS = 1.0
AUDIO_HOP_SECONDS = 0.5

_video_feature_extractor = None
_video_model = None
_audio_model = None
_fusion_package = None
_hubert_feature_extractor = None
_hubert_model = None
_device = None


def _ensure_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_SEGMENT_DIR.mkdir(parents=True, exist_ok=True)


def _get_video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps <= 0:
        fps = 25

    duration = frame_count / fps
    return duration


def _make_segments(duration_seconds, segment_seconds):
    segments = []

    start = 0.0
    segment_index = 1

    while start < duration_seconds:
        end = min(start + segment_seconds, duration_seconds)

        if end - start >= 2:
            segments.append(
                {
                    "segment_id": f"segment_{segment_index:03d}",
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                }
            )
            segment_index += 1

        start += segment_seconds

    return segments


def _load_video_feature_extractor():
    global _video_feature_extractor

    if _video_feature_extractor is not None:
        return _video_feature_extractor

    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Model

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    pooled_output = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

    _video_feature_extractor = Model(
        inputs=base_model.input,
        outputs=pooled_output,
    )

    return _video_feature_extractor


def _load_video_model():
    global _video_model

    if _video_model is not None:
        return _video_model

    if not Path(VIDEO_MODEL_PATH).exists():
        raise FileNotFoundError(f"Video model not found: {VIDEO_MODEL_PATH}")

    _video_model = joblib.load(VIDEO_MODEL_PATH)
    return _video_model


def _load_audio_model():
    global _audio_model

    if _audio_model is not None:
        return _audio_model

    if not Path(FINAL_AUDIO_MODEL_PATH).exists():
        raise FileNotFoundError(f"Audio model not found: {FINAL_AUDIO_MODEL_PATH}")

    _audio_model = joblib.load(FINAL_AUDIO_MODEL_PATH)
    return _audio_model


def _load_fusion_package():
    global _fusion_package

    if _fusion_package is not None:
        return _fusion_package

    _fusion_package = load_fusion_model()
    return _fusion_package


def _load_hubert():
    global _hubert_feature_extractor
    global _hubert_model
    global _device

    if _hubert_feature_extractor is not None and _hubert_model is not None:
        return _hubert_feature_extractor, _hubert_model, _device

    from transformers import Wav2Vec2FeatureExtractor, HubertModel

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "facebook/hubert-base-ls960"

    _hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    _hubert_model = HubertModel.from_pretrained(model_name).to(_device)
    _hubert_model.eval()

    return _hubert_feature_extractor, _hubert_model, _device


def _probabilities_to_label_order(model, probabilities):
    model_classes = list(model.classes_)

    output = {}

    for emotion in LABEL_ORDER:
        if emotion in model_classes:
            emotion_index = model_classes.index(emotion)
            output[emotion] = float(probabilities[emotion_index])
        else:
            output[emotion] = 0.0

    return output


def _detect_largest_face(frame):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return x, y, w, h


def _extract_video_probabilities(video_path, start_time, end_time, max_faces=5):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    video_model = _load_video_model()
    feature_extractor = _load_video_feature_extractor()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, "video_unavailable"

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    frame_step = max(int(fps), 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    face_images = []
    current_frame = start_frame

    while current_frame <= end_frame and len(face_images) < max_faces:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            break

        face_box = _detect_largest_face(frame)

        if face_box is not None:
            x, y, w, h = face_box
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = preprocess_input(face.astype(np.float32))
            face_images.append(face)

        current_frame += frame_step

    cap.release()

    if len(face_images) == 0:
        return None, "no_face_detected"

    face_batch = np.array(face_images)

    features = feature_extractor.predict(face_batch, verbose=0)

    frame_probabilities = video_model.predict_proba(features)

    averaged_probabilities = frame_probabilities.mean(axis=0)

    video_probs = _probabilities_to_label_order(video_model, averaged_probabilities)

    return video_probs, "video_available"


def _get_moviepy_video_clip():
    try:
        from moviepy import VideoFileClip
        return VideoFileClip
    except Exception:
        from moviepy.editor import VideoFileClip
        return VideoFileClip


def _extract_audio_segment(video_path, start_time, end_time, output_wav_path):
    VideoFileClip = _get_moviepy_video_clip()

    clip = VideoFileClip(str(video_path))

    try:
        if clip.audio is None:
            clip.close()
            return False

        if hasattr(clip, "subclipped"):
            subclip = clip.subclipped(start_time, end_time)
        else:
            subclip = clip.subclip(start_time, end_time)

        if subclip.audio is None:
            subclip.close()
            clip.close()
            return False

        subclip.audio.write_audiofile(
            str(output_wav_path),
            fps=SAMPLE_RATE,
            nbytes=2,
            codec="pcm_s16le",
            logger=None,
        )

        subclip.close()
        clip.close()

        return Path(output_wav_path).exists() and Path(output_wav_path).stat().st_size > 0

    except Exception:
        try:
            clip.close()
        except Exception:
            pass

        return False


@torch.no_grad()
def _extract_hubert_feature(audio_path):
    hubert_feature_extractor, hubert_model, device = _load_hubert()

    waveform, sample_rate = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True,
    )

    if waveform is None or len(waveform) == 0:
        return None

    rms = float(np.sqrt(np.mean(waveform ** 2)))

    if rms < 0.001:
        return None

    window_length = int(AUDIO_WINDOW_SECONDS * SAMPLE_RATE)
    hop_length = int(AUDIO_HOP_SECONDS * SAMPLE_RATE)

    if len(waveform) < window_length:
        waveform = np.pad(waveform, (0, window_length - len(waveform)))

    windows = []

    for start in range(0, len(waveform) - window_length + 1, hop_length):
        window = waveform[start:start + window_length]
        windows.append(window)

    if len(windows) == 0:
        return None

    inputs = hubert_feature_extractor(
        windows,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs["input_values"].to(device)

    outputs = hubert_model(input_values)

    hidden_state = outputs.last_hidden_state

    window_embeddings = hidden_state.mean(dim=1).cpu().numpy()

    mean_vector = window_embeddings.mean(axis=0)
    std_vector = window_embeddings.std(axis=0)

    final_vector = np.concatenate([mean_vector, std_vector])

    return final_vector.astype(np.float32)


def _extract_audio_probabilities(video_path, start_time, end_time, segment_temp_dir):
    audio_model = _load_audio_model()

    wav_path = segment_temp_dir / f"audio_{start_time:.2f}_{end_time:.2f}.wav"

    audio_created = _extract_audio_segment(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        output_wav_path=wav_path,
    )

    if not audio_created:
        return None, "audio_unavailable"

    audio_feature = _extract_hubert_feature(wav_path)

    if audio_feature is None:
        return None, "silent_or_invalid_audio"

    probabilities = audio_model.predict_proba([audio_feature])[0]

    audio_probs = _probabilities_to_label_order(audio_model, probabilities)

    return audio_probs, "audio_available"


def _predict_with_available_modalities(video_probs, audio_probs):
    fusion_package = _load_fusion_package()
    fusion_model = fusion_package["model"]
    feature_columns = fusion_package["feature_columns"]

    if video_probs is not None and audio_probs is not None:
        row = {}

        for emotion in LABEL_ORDER:
            row[f"video_prob_{emotion}"] = video_probs[emotion]
            row[f"audio_prob_{emotion}"] = audio_probs[emotion]

        fusion_input = pd.DataFrame([row])[feature_columns]

        predicted_emotion = fusion_model.predict(fusion_input)[0]
        probabilities = fusion_model.predict_proba(fusion_input)[0]

        model_classes = list(fusion_model.classes_)

        final_probs = {}

        for emotion in LABEL_ORDER:
            if emotion in model_classes:
                emotion_index = model_classes.index(emotion)
                final_probs[emotion] = float(probabilities[emotion_index])
            else:
                final_probs[emotion] = 0.0

        prediction_mode = "audio_video_fusion"

    elif video_probs is not None:
        final_probs = video_probs
        predicted_emotion = max(final_probs, key=final_probs.get)
        prediction_mode = "video_only_fallback"

    elif audio_probs is not None:
        final_probs = audio_probs
        predicted_emotion = max(final_probs, key=final_probs.get)
        prediction_mode = "audio_only_fallback"

    else:
        final_probs = {emotion: 0.0 for emotion in LABEL_ORDER}
        predicted_emotion = "insufficient_evidence"
        prediction_mode = "insufficient_evidence"

    if predicted_emotion == "insufficient_evidence":
        engagement_level = "Insufficient Evidence"
        feedback = (
            "The system could not detect enough usable audio or visual evidence "
            "for this segment."
        )
        confidence = 0.0
    else:
        engagement_level = map_emotion_to_engagement(predicted_emotion)
        feedback = get_feedback_for_engagement(engagement_level)
        confidence = max(final_probs.values())

    return predicted_emotion, engagement_level, feedback, confidence, final_probs, prediction_mode


def process_classroom_video(video_path, segment_seconds=10):
    """
    Main uploaded classroom video pipeline.

    Returns:
        segment_results_df
        engagement_summary_df
    """
    _ensure_dirs()

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Uploaded video not found: {video_path}")

    duration = _get_video_duration(video_path)

    segments = _make_segments(
        duration_seconds=duration,
        segment_seconds=segment_seconds,
    )

    if len(segments) == 0:
        raise ValueError("Video is too short to analyse.")

    run_id = uuid.uuid4().hex[:8]
    segment_temp_dir = TEMP_SEGMENT_DIR / run_id
    segment_temp_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for segment in segments:
        start_time = segment["start_time"]
        end_time = segment["end_time"]

        video_probs, video_status = _extract_video_probabilities(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
        )

        audio_probs, audio_status = _extract_audio_probabilities(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            segment_temp_dir=segment_temp_dir,
        )

        (
            predicted_emotion,
            engagement_level,
            feedback,
            confidence,
            final_probs,
            prediction_mode,
        ) = _predict_with_available_modalities(video_probs, audio_probs)

        row = {
            "segment_id": segment["segment_id"],
            "start_time": start_time,
            "end_time": end_time,
            "video_status": video_status,
            "audio_status": audio_status,
            "prediction_mode": prediction_mode,
            "final_predicted_emotion": predicted_emotion,
            "predicted_engagement_level_final": engagement_level,
            "confidence": confidence,
            "logic_based_feedback_final": feedback,
        }

        for emotion in LABEL_ORDER:
            row[f"video_prob_{emotion}"] = (
                video_probs[emotion] if video_probs is not None else 0.0
            )

            row[f"audio_prob_{emotion}"] = (
                audio_probs[emotion] if audio_probs is not None else 0.0
            )

            row[f"final_prob_{emotion}"] = final_probs[emotion]

        rows.append(row)

    segment_results_df = pd.DataFrame(rows)

    engagement_summary_df = (
        segment_results_df["predicted_engagement_level_final"]
        .value_counts()
        .reset_index()
    )

    engagement_summary_df.columns = ["engagement_level", "count"]

    engagement_summary_df["percentage"] = (
        engagement_summary_df["count"] / engagement_summary_df["count"].sum() * 100
    ).round(2)

    return segment_results_df, engagement_summary_df