import os


DATASET_ROOT = r"C:\Users\user\Desktop\dataset"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Processed outputs inside your repo
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
VIDEO_FACES_DIR = os.path.join(PROCESSED_DIR, "video_faces")

# CSVs
RAVDESS_INDEX_CSV = os.path.join(PROCESSED_DIR, "ravdess_index.csv")
FACES_LABELED_CSV = os.path.join(VIDEO_FACES_DIR, "faces_labeled.csv")
FACES_TRAIN_CSV   = os.path.join(VIDEO_FACES_DIR, "faces_train.csv")
FACES_VAL_CSV     = os.path.join(VIDEO_FACES_DIR, "faces_val.csv")
FACES_TEST_CSV    = os.path.join(VIDEO_FACES_DIR, "faces_test.csv")

# Feature outputs (.npy)
X_TRAIN_VIDEO_NPY = os.path.join(PROCESSED_DIR, "X_train_video.npy")
Y_TRAIN_VIDEO_NPY = os.path.join(PROCESSED_DIR, "y_train_video.npy")
X_VAL_VIDEO_NPY   = os.path.join(PROCESSED_DIR, "X_val_video.npy")
Y_VAL_VIDEO_NPY   = os.path.join(PROCESSED_DIR, "y_val_video.npy")
X_TEST_VIDEO_NPY  = os.path.join(PROCESSED_DIR, "X_test_video.npy")
Y_TEST_VIDEO_NPY  = os.path.join(PROCESSED_DIR, "y_test_video.npy")

# Image/face extraction settings
IMG_SIZE = 224
EVERY_N_FRAMES = 5      # extract 1 face every 5 frames
FACE_MIN_SIZE = 60      # ignore tiny detections

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


# Audio feature paths
X_TRAIN_AUDIO_NPY = os.path.join(PROCESSED_DIR, "X_train_audio.npy")
Y_TRAIN_AUDIO_NPY = os.path.join(PROCESSED_DIR, "y_train_audio.npy")

X_VAL_AUDIO_NPY = os.path.join(PROCESSED_DIR, "X_val_audio.npy")
Y_VAL_AUDIO_NPY = os.path.join(PROCESSED_DIR, "y_val_audio.npy")

X_TEST_AUDIO_NPY = os.path.join(PROCESSED_DIR, "X_test_audio.npy")
Y_TEST_AUDIO_NPY = os.path.join(PROCESSED_DIR, "y_test_audio.npy")
