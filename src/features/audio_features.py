import numpy as np
from src.utils.config import (
    X_TRAIN_AUDIO_NPY,
    Y_TRAIN_AUDIO_NPY,
    X_VAL_AUDIO_NPY,
    Y_VAL_AUDIO_NPY,
    X_TEST_AUDIO_NPY,
    Y_TEST_AUDIO_NPY,
)

def load_audio_features(split="train"):
    if split == "train":
        X = np.load(X_TRAIN_AUDIO_NPY)
        y = np.load(Y_TRAIN_AUDIO_NPY)
    elif split == "val":
        X = np.load(X_VAL_AUDIO_NPY)
        y = np.load(Y_VAL_AUDIO_NPY)
    elif split == "test":
        X = np.load(X_TEST_AUDIO_NPY)
        y = np.load(Y_TEST_AUDIO_NPY)
    else:
        raise ValueError("Invalid split")

    return X, y
