import os
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.utils import config


def build_feature_extractor():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    return model


def load_and_preprocess(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = preprocess_input(img.astype(np.float32))
    return img


def extract_from_csv(csv_path: str, out_x: str, out_y: str, batch_size: int = 32):
    df = pd.read_csv(csv_path)
    extractor = build_feature_extractor()

    X = []
    y = []

    batch_imgs = []
    batch_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {os.path.basename(csv_path)}"):
        img = load_and_preprocess(row["face_path"])
        batch_imgs.append(img)
        batch_labels.append(row["emotion"])

        if len(batch_imgs) == batch_size:
            batch_arr = np.array(batch_imgs)
            feats = extractor.predict(batch_arr, verbose=0)
            X.append(feats)
            y.extend(batch_labels)
            batch_imgs, batch_labels = [], []

    # last batch
    if len(batch_imgs) > 0:
        batch_arr = np.array(batch_imgs)
        feats = extractor.predict(batch_arr, verbose=0)
        X.append(feats)
        y.extend(batch_labels)

    X = np.vstack(X)  # (N, 1280)
    y = np.array(y)

    os.makedirs(os.path.dirname(out_x), exist_ok=True)
    np.save(out_x, X)
    np.save(out_y, y)

    print("✅ Saved:", out_x, out_y)
    print("Shapes:", X.shape, y.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.split in ["train", "all"]:
        extract_from_csv(config.FACES_TRAIN_CSV, config.X_TRAIN_VIDEO_NPY, config.Y_TRAIN_VIDEO_NPY, args.batch_size)
    if args.split in ["val", "all"]:
        extract_from_csv(config.FACES_VAL_CSV, config.X_VAL_VIDEO_NPY, config.Y_VAL_VIDEO_NPY, args.batch_size)
    if args.split in ["test", "all"]:
        extract_from_csv(config.FACES_TEST_CSV, config.X_TEST_VIDEO_NPY, config.Y_TEST_VIDEO_NPY, args.batch_size)


if __name__ == "__main__":
    main()
