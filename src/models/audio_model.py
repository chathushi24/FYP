import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def load_labels(labels_txt: str):
    labels = Path(labels_txt).read_text(encoding="utf-8").splitlines()
    labels = [x.strip() for x in labels if x.strip()]
    return labels


def evaluate(model, name, X, y, labels):
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    print(f"\n===== {name} Accuracy: {acc:.4f} =====")
    print(classification_report(y, pred, target_names=labels))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True, help="Folder containing X_train.npy, y_train.npy, ... and labels.txt")
    ap.add_argument("--out_dir", required=True, help="Where to save model + metadata")

    ap.add_argument("--C", type=float, default=20.0)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(feat_dir / "X_train.npy")
    y_train = np.load(feat_dir / "y_train.npy")
    X_val = np.load(feat_dir / "X_val.npy")
    y_val = np.load(feat_dir / "y_val.npy")
    X_test = np.load(feat_dir / "X_test.npy")
    y_test = np.load(feat_dir / "y_test.npy")

    labels = load_labels(str(feat_dir / "labels.txt"))

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=args.C,
            gamma=args.gamma,
            class_weight="balanced",
            probability=True,
            random_state=args.seed
        ))
    ])

    svm.fit(X_train, y_train)

    val_acc = evaluate(svm, "VAL", X_val, y_val, labels)
    test_acc = evaluate(svm, "TEST", X_test, y_test, labels)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"hubert_svm_{timestamp}"

    model_path = out_dir / f"{prefix}.joblib"
    joblib.dump(svm, model_path)

    meta = {
        "model_type": "HuBERT + SVM",
        "timestamp": timestamp,
        "C": args.C,
        "gamma": args.gamma,
        "seed": args.seed,
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "labels_order": labels,
        "feature_dim": int(X_train.shape[1]),
        "notes": "Training pipeline matches audio_training (2).ipynb: HuBERT mean+std windows -> StandardScaler -> SVC(RBF)"
    }

    meta_path = out_dir / f"{prefix}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Saved model:", model_path)
    print("✅ Saved meta :", meta_path)


if __name__ == "__main__":
    main()
