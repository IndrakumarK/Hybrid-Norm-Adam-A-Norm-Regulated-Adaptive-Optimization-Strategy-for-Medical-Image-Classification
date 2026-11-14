"""
train.py
========

Reproducible VGG19 brain-tumor binary-classification training.

Datasets:
    SARTAJ, Navoneel, Masoud, Indk214

Class convention:
    0 = no_tumor
    1 = tumor

Expected dataset structure (the script does NOT randomly repartition data):

data/
  DATASET/
    train/
      no_tumor/
      tumor/
    val/
      no_tumor/
      tumor/
    test/
      no_tumor/
      tumor/

Five-seed experiment:
    python train.py --dataset Navoneel --seeds 42 52 62 72 82

Single run:
    python train.py --dataset Navoneel --seeds 42

Optimizer comparison:
    python train.py --dataset Navoneel --seeds 42 52 62 72 82 \
        --optimizers HNAdam Adam AMSGrad SGD AdamW

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef
)

from models.vgg19_model import build_vgg19
from optimizers.hnadam import HNAdam


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DATASETS = ["SARTAJ", "Navoneel", "Masoud", "Indk214"]
DEFAULT_SEEDS = [42, 52, 62, 72, 82]
IMAGE_SIZE = (240, 240)
CLASS_FOLDERS = {
    "no_tumor": 0,
    "notumor": 0,
    "no tumor": 0,
    "healthy": 0,
    "normal": 0,
    "tumor": 1,
    "glioma": 1,
    "meningioma": 1,
    "pituitary": 1,
    "pituitary_tumor": 1,
    "brain_tumor": 1
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set Python, NumPy and TensorFlow seeds before model/optimizer creation."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Deterministic kernels are not available for every operation/device.
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def get_label(class_name: str) -> int:
    key = normalize_class_name(class_name)
    if key not in CLASS_FOLDERS:
        raise ValueError(
            f"Unknown class folder '{class_name}'. Expected no_tumor/tumor "
            "or original tumor subtype folders (glioma, meningioma, pituitary)."
        )
    return CLASS_FOLDERS[key]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def load_split(dataset_dir: Path, split: str):
    """
    Load a pre-existing split without changing its membership.
    No test data is read by this training script.
    """
    split_dir = dataset_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing split directory: {split_dir}. Expected train/val/test "
            "directories prepared before training."
        )

    images, labels, paths = [], [], []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = get_label(class_dir.name)
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                images.append(read_image(path))
                labels.append(label)
                paths.append(str(path))

    if not images:
        raise ValueError(f"No supported images found in {split_dir}")

    x = np.stack(images).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            f"{split_dir} must contain both classes 0 and 1. "
            f"Observed labels: {np.unique(y).tolist()}"
        )

    return x, y, paths


def validate_split_integrity(train_paths, val_paths):
    """Detect exact file duplicates shared between train and validation."""
    train_hashes = {}
    for path in train_paths:
        train_hashes.setdefault(sha256_file(Path(path)), []).append(path)

    overlaps = []
    for path in val_paths:
        digest = sha256_file(Path(path))
        if digest in train_hashes:
            overlaps.append({
                "train_paths": train_hashes[digest],
                "validation_path": path,
                "sha256": digest
            })

    if overlaps:
        raise ValueError(
            f"Found {len(overlaps)} exact duplicate image(s) across train "
            "and validation. Resolve these before training. See console output."
        )


# ---------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------

def make_optimizer(name: str, learning_rate: float, seed: int):
    """
    Construct only explicitly supported optimizers.

    HNAdam uses the manuscript implementation's constructor:
      learning_rate, beta_1, beta_2, epsilon, K_t0, threshold.
    K_t0 is sampled once per optimizer instance from U(2,4), after seeding.
    """
    name = name.lower()

    if name == "hnadam":
        # Seed is set before this call, making Python random.uniform reproducible.
        optimizer = HNAdam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            K_t0=None,
            threshold=2.0
        )
        k_value = getattr(optimizer, "K_t0", None)
        return optimizer, {"optimizer": "HNAdam", "K_t0": k_value,
                           "K_t0_sampling": "once_per_optimizer_initialization",
                           "threshold": 2.0}

    if name == "adam":
        return keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9, beta_2=0.999, epsilon=1e-8
        ), {"optimizer": "Adam"}

    if name == "amsgrad":
        return keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9, beta_2=0.999, epsilon=1e-8,
            amsgrad=True
        ), {"optimizer": "AMSGrad"}

    if name == "sgd":
        return keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.0
        ), {"optimizer": "SGD", "momentum": 0.0}

    if name == "adamw":
        if not hasattr(keras.optimizers, "AdamW"):
            raise RuntimeError(
                "This TensorFlow/Keras version does not expose AdamW."
            )
        return keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            beta_1=0.9, beta_2=0.999, epsilon=1e-8
        ), {"optimizer": "AdamW", "weight_decay": 1e-4}

    raise ValueError(
        f"Unsupported optimizer '{name}'. Supported: HNAdam, Adam, "
        "AMSGrad, SGD, AdamW. RAdam/AdaBelief must be added using verified "
        "implementations before being reported."
    )


# ---------------------------------------------------------------------
# Model and class weights
# ---------------------------------------------------------------------

def build_binary_model():
    model = build_vgg19(num_classes=1, input_shape=(240, 240, 3))

    output_shape = model.output_shape
    if isinstance(output_shape, list) or output_shape[-1] != 1:
        raise ValueError(
            "build_vgg19 must return a single sigmoid output for binary "
            f"classification. Received output shape: {output_shape}"
        )
    return model


def calculate_class_weights(y_train):
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


# ---------------------------------------------------------------------
# Training curves and validation metrics
# ---------------------------------------------------------------------

def save_training_curves(history: Dict, output_dir: Path, tag: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("loss", [])) + 1)

    for metric, ylabel in [
        ("loss", "Loss"),
        ("accuracy", "Accuracy")
    ]:
        if metric not in history:
            continue

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history[metric], label=f"Training {ylabel.lower()}")
        val_key = f"val_{metric}"
        if val_key in history:
            plt.plot(epochs, history[val_key],
                     label=f"Validation {ylabel.lower()}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} curves - {tag}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{tag}_{metric}_curve.png",
            dpi=600,
            bbox_inches="tight"
        )
        plt.close()


def validation_metrics(model, x_val, y_val):
    probabilities = model.predict(x_val, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(y_val, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "validation_accuracy": float(accuracy_score(y_val, predictions)),
        "validation_precision": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "validation_sensitivity": float(
            recall_score(y_val, predictions, zero_division=0)
        ),
        "validation_specificity": float(
            tn / (tn + fp) if (tn + fp) else float("nan")
        ),
        "validation_macro_f1": float(
            f1_score(y_val, predictions, average="macro", zero_division=0)
        ),
        "validation_balanced_accuracy": float(
            balanced_accuracy_score(y_val, predictions)
        ),
        "validation_mcc": float(matthews_corrcoef(y_val, predictions)),
        "validation_auroc": float(roc_auc_score(y_val, probabilities)),
        "validation_auprc": float(average_precision_score(y_val, probabilities)),
        "validation_confusion_matrix": cm.tolist()
    }


# ---------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------

def run_one_experiment(args, dataset_dir: Path, dataset: str,
                       optimizer_name: str, seed: int):
    set_global_seed(seed)

    print("\n" + "=" * 72)
    print(f"DATASET={dataset} | OPTIMIZER={optimizer_name} | SEED={seed}")
    print("=" * 72)

    # Only train and validation splits are loaded here.
    x_train, y_train, train_paths = load_split(dataset_dir, "train")
    x_val, y_val, val_paths = load_split(dataset_dir, "val")

    # Exact train/validation duplicate check.
    validate_split_integrity(train_paths, val_paths)

    run_name = f"{optimizer_name}_seed_{seed}"
    output_dir = Path(args.results_root) / dataset / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Place the checkpoint where evaluate.py's resolver expects it:
    # results/<dataset>/<optimizer>_seed_<seed>.keras
    checkpoint_path = (
        Path(args.results_root) / dataset / f"{run_name}.keras"
    )
    history_path = output_dir / f"{run_name}_history.csv"
    config_path = output_dir / f"{run_name}_config.json"

    class_weights = calculate_class_weights(y_train)
    optimizer, optimizer_metadata = make_optimizer(
        optimizer_name, args.learning_rate, seed
    )

    model = build_binary_model()
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc", curve="ROC"),
            keras.metrics.AUC(name="auprc", curve="PR")
        ]
    )

    parameter_metadata = {
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(
            np.prod(v.shape) for v in model.trainable_weights
        )),
        "non_trainable_parameters": int(sum(
            np.prod(v.shape) for v in model.non_trainable_weights
        ))
    }

    config = {
        "dataset": dataset,
        "optimizer": optimizer_name,
        "seed": seed,
        "learning_rate": args.learning_rate,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "image_size": [240, 240],
        "class_mapping": {"no_tumor": 0, "tumor": 1},
        "loss": "binary_crossentropy",
        "decision_threshold_for_validation_reporting": 0.5,
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "train_class_counts": {
            "no_tumor": int(np.sum(y_train == 0)),
            "tumor": int(np.sum(y_train == 1))
        },
        "validation_class_counts": {
            "no_tumor": int(np.sum(y_val == 0)),
            "tumor": int(np.sum(y_val == 1))
        },
        "class_weights": class_weights,
        "optimizer_configuration": optimizer_metadata,
        "model_parameters": parameter_metadata,
        "test_set_used_during_training": False
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.CSVLogger(str(history_path), append=False)
    ]

    if args.early_stopping_patience > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=args.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        )

    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train.astype(np.float32),
        validation_data=(x_val, y_val.astype(np.float32)),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    training_seconds = time.perf_counter() - start

    # Load the best validation-loss checkpoint. The test split is not touched.
    best_model = keras.models.load_model(
        str(checkpoint_path),
        custom_objects={"HNAdam": HNAdam},
        compile=False
    )

    val_results = validation_metrics(best_model, x_val, y_val)
    result = {
        **config,
        "training_seconds": float(training_seconds),
        "epochs_completed": int(len(history.history.get("loss", []))),
        "best_validation_loss": float(np.min(history.history["val_loss"])),
        **val_results,
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path)
    }

    with (output_dir / f"{run_name}_results.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, indent=2, default=str)

    save_training_curves(history.history, output_dir, run_name)

    # Save exact split manifest for reproducibility.
    pd.DataFrame({
        "path": train_paths,
        "split": "train",
        "label": y_train
    }).to_csv(output_dir / "train_manifest.csv", index=False)

    pd.DataFrame({
        "path": val_paths,
        "split": "validation",
        "label": y_val
    }).to_csv(output_dir / "validation_manifest.csv", index=False)

    print(f"\nBest checkpoint: {checkpoint_path}")
    print(f"Training time (seconds): {training_seconds:.2f}")
    print(f"Best validation loss: {result['best_validation_loss']:.6f}")
    print(f"Validation accuracy: {result['validation_accuracy']:.4f}")

    del model, best_model, x_train, x_val, y_train, y_val
    keras.backend.clear_session()

    return result


# ---------------------------------------------------------------------
# Experiment summary
# ---------------------------------------------------------------------

def summarize_runs(results: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / "training_run_results.csv", index=False)

    metric_columns = [
        "validation_accuracy",
        "validation_precision",
        "validation_sensitivity",
        "validation_specificity",
        "validation_macro_f1",
        "validation_balanced_accuracy",
        "validation_mcc",
        "validation_auroc",
        "validation_auprc",
        "best_validation_loss",
        "training_seconds"
    ]
    available = [c for c in metric_columns if c in frame.columns]
    summary = frame.groupby(["dataset", "optimizer"])[available].agg(
        ["mean", "std", "count"]
    )
    summary.to_csv(output_dir / "training_mean_sd_summary.csv")

    # Keep a machine-readable summary as well.
    summary_json = {}
    for (dataset, optimizer), group in frame.groupby(["dataset", "optimizer"]):
        key = f"{dataset}__{optimizer}"
        summary_json[key] = {}
        for metric in available:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            summary_json[key][metric] = {
                "mean": float(values.mean()) if len(values) else None,
                "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                "n": int(len(values))
            }

    with (output_dir / "training_mean_sd_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary_json, f, indent=2)

    print(f"\nTraining summaries saved to: {output_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproducible VGG19/HNAdam brain-tumor training"
    )
    parser.add_argument(
        "--dataset", required=True, choices=DATASETS,
        help="Dataset folder under --data-root"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--optimizers", nargs="+",
        default=["HNAdam"],
        choices=["HNAdam", "Adam", "AMSGrad", "SGD", "AdamW"],
        help="Train one or more explicitly implemented optimizers"
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="0 disables early stopping. Use only if consistent with the protocol."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seed values must be unique.")

    dataset_dir = Path(args.data_root) / args.dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Check split folders before starting any runs. Test folder is checked
    # for presence only; it is never opened/read by this training script.
    for split in ("train", "val", "test"):
        if not (dataset_dir / split).is_dir():
            raise FileNotFoundError(
                f"Expected pre-existing split folder: {dataset_dir / split}"
            )

    all_results = []
    for optimizer_name in args.optimizers:
        for seed in args.seeds:
            result = run_one_experiment(
                args=args,
                dataset_dir=dataset_dir,
                dataset=args.dataset,
                optimizer_name=optimizer_name,
                seed=seed
            )
            all_results.append(result)

    summary_dir = Path(args.results_root) / args.dataset / "training_summary"
    summarize_runs(all_results, summary_dir)

    print("\n" + "=" * 72)
    print("ALL REQUESTED TRAINING RUNS COMPLETED")
    print(f"Dataset: {args.dataset}")
    print(f"Optimizers: {args.optimizers}")
    print(f"Seeds: {args.seeds}")
    print(f"Outputs: {summary_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
