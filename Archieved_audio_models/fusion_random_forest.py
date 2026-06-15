import numpy as np


def _safe_predict_proba(model, X):
    """
    Returns predict_proba if available, otherwise makes a pseudo-probability
    from predicted labels.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    # fallback: one-hot based on predictions
    preds = model.predict(X)
    classes = np.unique(preds)
    proba = np.zeros((len(preds), len(classes)), dtype=np.float32)
    for i, p in enumerate(preds):
        proba[i, np.where(classes == p)[0][0]] = 1.0
    return proba


def predict_fused(video_model, audio_model, Xv, Xa, w_video=0.5):
    """
    Late fusion (soft voting) between video and audio probabilities.

    Args:
        video_model: sklearn model with predict_proba
        audio_model: sklearn model with predict_proba
        Xv: (N, Dv) video feature vectors
        Xa: (N, Da) audio feature vectors
        w_video: weight for video [0..1]. audio weight = 1 - w_video

    Returns:
        pred_label (int)
        fused_probs (1D array)
        v_probs (1D array)
        a_probs (1D array)
    """
    # ensure 2D
    Xv = np.asarray(Xv)
    Xa = np.asarray(Xa)
    if Xv.ndim == 1:
        Xv = Xv.reshape(1, -1)
    if Xa.ndim == 1:
        Xa = Xa.reshape(1, -1)

    v_probs = _safe_predict_proba(video_model, Xv)[0]
    a_probs = _safe_predict_proba(audio_model, Xa)[0]

    # If shapes differ, something is wrong with label ordering
    if v_probs.shape != a_probs.shape:
        raise ValueError(
            f"Video probs shape {v_probs.shape} != Audio probs shape {a_probs.shape}. "
            "This usually means different class ordering or different number of classes."
        )

    w_video = float(w_video)
    w_audio = 1.0 - w_video

    fused_probs = (w_video * v_probs) + (w_audio * a_probs)
    pred_label = int(np.argmax(fused_probs))

    return pred_label, fused_probs, v_probs, a_probs


