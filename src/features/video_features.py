import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tqdm import tqdm


def build_feature_extractor(image_size: int = 224):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )

    pooled = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=pooled,
    )

    return model


def load_and_preprocess_image(image_path: str, image_size: int = 224):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32)

    image = preprocess_input(image)

    return image


def extract_video_features(
    faces_csv: str,
    output_x: str,
    output_y: str,
    batch_size: int = 32,
    image_size: int = 224,
):
    df = pd.read_csv(faces_csv)

    if "face_path" not in df.columns:
        raise ValueError("faces_csv must contain a face_path column.")

    if "emotion" not in df.columns:
        raise ValueError("faces_csv must contain an emotion column.")

    extractor = build_feature_extractor(image_size=image_size)

    images = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading face images"):
        try:
            image = load_and_preprocess_image(row["face_path"], image_size=image_size)
            images.append(image)
            labels.append(row["emotion"])
        except Exception as error:
            print("Skipped:", row["face_path"], "|", error)

    X_images = np.array(images, dtype=np.float32)

    features = extractor.predict(
        X_images,
        batch_size=batch_size,
        verbose=1,
    )

    y = np.array(labels)

    Path(output_x).parent.mkdir(parents=True, exist_ok=True)

    np.save(output_x, features)
    np.save(output_y, y)

    print("Saved X:", output_x, features.shape)
    print("Saved y:", output_y, y.shape)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--faces-csv", default="data/processed/video_faces_index.csv")
    parser.add_argument("--output-x", default="data/processed/X_video_features.npy")
    parser.add_argument("--output-y", default="data/processed/y_video_labels.npy")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    extract_video_features(
        faces_csv=args.faces_csv,
        output_x=args.output_x,
        output_y=args.output_y,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()