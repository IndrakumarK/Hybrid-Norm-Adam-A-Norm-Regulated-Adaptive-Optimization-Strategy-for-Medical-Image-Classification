"""
Reproducible VGG19 training for binary brain-tumor MRI classification.

Datasets:
    SARTAJ, Navoneel, Masoud, Indk214

Binary class mapping:
    0 = no_tumor
    1 = tumor

Expected directory structure:
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

The script uses the existing train/validation/test partitions and does not
reassign images between splits. The test split is not loaded during training.
The best checkpoint is selected using validation loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from models.vgg19_model import build_vgg19
from optimizers.hnadam import HNAdam


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATASETS = ["SARTAJ", "Navoneel", "Masoud", "Indk214"]
DEFAULT_SEEDS = [42, 52, 62, 72, 82]
IMAGE_SIZE = (240, 240)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASS_TO_LABEL = {
    "no_tumor": 0,
    "notumor": 0,
    "healthy": 0,
    "normal": 0,
    "tumor": 1,
    "tumour": 1,
    "glioma": 1,
    "meningioma": 1,
    "pituitary": 1,
    "pituitary_tumor": 1,
    "brain_tumor": 1,
}


# ---------------------------------------------------------------------
# Reproducibility and file utilities
# ---------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str, allow_nan=True)


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def normalize_class_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def get_label(class_name: str) -> int:
    key = normalize_class_name(class_name)

    if key not in CLASS_TO_LABEL:
        raise ValueError(
            f"Unknown class directory '{class_name}'. "
            "Expected no_tumor or a tumor class."
        )

    return CLASS_TO_LABEL[key]


def list_image_files(folder: Path) -> List[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: Path) -> np.ndarray:
    """Read an MRI image, convert it to RGB, resize, and normalize."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    return image.astype(np.float32) / 255.0


def load_split(
    dataset_dir: Path,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load an existing dataset split.

    The function does not create or modify split membership.
    """
    split_dir = dataset_dir / split

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing split directory: {split_dir}"
        )

    image_arrays = []
    labels = []
    paths = []

    class_directories = sorted(
        path for path in split_dir.iterdir()
        if path.is_dir()
    )

    if not class_directories:
        raise ValueError(
            f"No class directories found in {split_dir}"
        )

    for class_dir in class_directories:
        label = get_label(class_dir.name)

        for image_path in list_image_files(class_dir):
            image_arrays.append(read_image(image_path))
            labels.append(label)
            paths.append(str(image_path))

    if not image_arrays:
        raise ValueError(
            f"No supported images found in {split_dir}"
        )

    x = np.stack(image_arrays).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            f"{split_dir} must contain both binary classes. "
            f"Observed labels: {np.unique(y).tolist()}"
        )

    return x, y, paths


def validate_split_integrity(
    train_paths: List[str],
    validation_paths: List[str],
) -> None:
    """Check for exact duplicate files shared by training and validation."""
    train_hashes = {}

    for path in train_paths:
        digest = sha256_file(Path(path))
        train_hashes.setdefault(digest, []).append(path)

    overlaps = []

    for path in validation_paths:
        digest = sha256_file(Path(path))

        if digest in train_hashes:
            overlaps.append({
                "training_paths": train_hashes[digest],
                "validation_path": path,
                "sha256": digest,
            })

    if overlaps:
        details = json.dumps(overlaps, indent=2)
        raise ValueError(
            "Exact duplicate images were found across training and "
            f"validation splits:\n{details}"
        )


# ---------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------

def make_optimizer(
    name: str,
    learning_rate: float,
):
    """Create one of the supported optimizers."""
    normalized_name = name.strip().lower()

    if normalized_name == "hnadam":
        optimizer = HNAdam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            K_t0=None,
            threshold=2.0,
        )

        metadata = {
            "optimizer": "HNAdam",
            "learning_rate": learning_rate,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
            "K_t0": float(optimizer.K_t0),
            "K_t0_sampling": "once_per_optimizer_initialization",
            "switching_threshold": 2.0,
        }

        return optimizer, metadata

    if normalized_name == "adam":
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        )
        return optimizer, {
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
        }

    if normalized_name == "amsgrad":
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            amsgrad=True,
        )
        return optimizer, {
            "optimizer": "AMSGrad",
            "learning_rate": learning_rate,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
        }

    if normalized_name == "sgd":
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.0,
        )
        return optimizer, {
            "optimizer": "SGD",
            "learning_rate": learning_rate,
            "momentum": 0.0,
        }

    if normalized_name == "adamw":
        adamw_class = getattr(keras.optimizers, "AdamW", None)

        if adamw_class is None:
            experimental_optimizers = getattr(
                keras.optimizers, "experimental", None
            )
            adamw_class = (
                getattr(experimental_optimizers, "AdamW", None)
                if experimental_optimizers is not None
                else None
            )

        if adamw_class is None:
            raise RuntimeError(
                "AdamW is unavailable in the installed TensorFlow/Keras version."
            )

        optimizer = adamw_class(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        )

        return optimizer, {
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": 1e-4,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
        }

    raise ValueError(
        f"Unsupported optimizer '{name}'. "
        "Choose HNAdam, Adam, AMSGrad, SGD, or AdamW."
    )


# ---------------------------------------------------------------------
# Model and class weights
# ---------------------------------------------------------------------

def build_binary_model():
    model = build_vgg19(
        num_classes=1,
        input_shape=(240, 240, 3),
    )

    output_shape = model.output_shape

    if isinstance(output_shape, list) or output_shape[-1] != 1:
        raise ValueError(
            "The VGG19 classifier must have one sigmoid output. "
            f"Received output shape: {output_shape}"
        )

    return model


def calculate_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    classes = np.array([0, 1])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    return {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }


def get_parameter_counts(model) -> Dict[str, int]:
    return {
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(
            np.prod(variable.shape)
            for variable in model.trainable_weights
        )),
        "non_trainable_parameters": int(sum(
            np.prod(variable.shape)
            for variable in model.non_trainable_weights
        )),
    }


# ---------------------------------------------------------------------
# Training curves and validation metrics
# ---------------------------------------------------------------------

def save_training_curves(
    history: Dict,
    output_dir: Path,
    tag: str,
) -> None:
    ensure_dir(output_dir)
    epochs = range(1, len(history.get("loss", [])) + 1)

    for metric, ylabel in (
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
    ):
        if metric not in history:
            continue

        plt.figure(figsize=(7, 5))
        plt.plot(
            epochs,
            history[metric],
            label=f"Training {ylabel.lower()}",
        )

        validation_key = f"val_{metric}"

        if validation_key in history:
            plt.plot(
                epochs,
                history[validation_key],
                label=f"Validation {ylabel.lower()}",
            )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} curves - {tag}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{tag}_{metric}_curve.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()


def validation_metrics(
    model,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Dict:
    probabilities = model.predict(
        x_validation,
        verbose=0,
    ).reshape(-1)

    if (
        not np.all(np.isfinite(probabilities))
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError(
            "Validation predictions must be finite sigmoid probabilities."
        )

    predictions = (probabilities >= 0.5).astype(int)

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    )
    tn, fp, fn, tp = matrix.ravel()

    return {
        "validation_accuracy": float(
            accuracy_score(y_validation, predictions)
        ),
        "validation_precision": float(
            precision_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "validation_sensitivity": float(
            recall_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "validation_specificity": (
            float(tn / (tn + fp))
            if tn + fp else float("nan")
        ),
        "validation_macro_f1": float(
            f1_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "validation_balanced_accuracy": float(
            balanced_accuracy_score(
                y_validation,
                predictions,
            )
        ),
        "validation_mcc": float(
            matthews_corrcoef(
                y_validation,
                predictions,
            )
        ),
        "validation_auroc": float(
            roc_auc_score(y_validation, probabilities)
        ),
        "validation_auprc": float(
            average_precision_score(y_validation, probabilities)
        ),
        "validation_confusion_matrix": matrix.tolist(),
    }


# ---------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------

def run_one_experiment(
    args,
    dataset_dir: Path,
    dataset: str,
    optimizer_name: str,
    seed: int,
) -> Dict:
    set_global_seed(seed)

    print("\n" + "=" * 72)
    print(
        f"DATASET={dataset} | OPTIMIZER={optimizer_name} | SEED={seed}"
    )
    print("=" * 72, flush=True)

    # The test split is not loaded during training.
    x_train, y_train, train_paths = load_split(
        dataset_dir, "train"
    )
    x_validation, y_validation, validation_paths = load_split(
        dataset_dir, "val"
    )

    validate_split_integrity(
        train_paths,
        validation_paths,
    )

    run_name = f"{optimizer_name}_seed_{seed}"

    run_output_dir = ensure_dir(
        Path(args.results_root) / dataset / run_name
    )

    checkpoint_path = (
        Path(args.results_root) / dataset / f"{run_name}.keras"
    )
    history_path = run_output_dir / f"{run_name}_history.csv"
    config_path = run_output_dir / f"{run_name}_config.json"

    class_weights = calculate_class_weights(y_train)

    optimizer, optimizer_metadata = make_optimizer(
        optimizer_name,
        args.learning_rate,
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
            keras.metrics.AUC(name="auprc", curve="PR"),
        ],
    )

    parameter_metadata = get_parameter_counts(model)

    config = {
        "dataset": dataset,
        "optimizer": optimizer_name,
        "seed": int(seed),
        "learning_rate": float(args.learning_rate),
        "epochs_requested": int(args.epochs),
        "batch_size": int(args.batch_size),
        "image_size": list(IMAGE_SIZE),
        "class_mapping": {
            "no_tumor": 0,
            "tumor": 1,
        },
        "loss": "binary_crossentropy",
        "validation_decision_threshold": 0.5,
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_validation)),
        "train_class_counts": {
            "no_tumor": int(np.sum(y_train == 0)),
            "tumor": int(np.sum(y_train == 1)),
        },
        "validation_class_counts": {
            "no_tumor": int(np.sum(y_validation == 0)),
            "tumor": int(np.sum(y_validation == 1)),
        },
        "class_weights": class_weights,
        "optimizer_configuration": optimizer_metadata,
        "model_parameters": parameter_metadata,
        "test_set_used_during_training": False,
    }

    write_json(config_path, config)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(history_path),
            append=False,
        ),
    ]

    if args.early_stopping_patience > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=args.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )

    start_time = time.perf_counter()

    history = model.fit(
        x_train,
        y_train.astype(np.float32),
        validation_data=(
            x_validation,
            y_validation.astype(np.float32),
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    training_seconds = time.perf_counter() - start_time

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Training did not produce the expected checkpoint: {checkpoint_path}"
        )

    best_model = keras.models.load_model(
        str(checkpoint_path),
        custom_objects={"HNAdam": HNAdam},
        compile=False,
    )

    validation_results = validation_metrics(
        best_model,
        x_validation,
        y_validation,
    )

    validation_losses = history.history.get("val_loss", [])

    if not validation_losses:
        raise RuntimeError("Training history does not contain validation loss.")

    result = {
        **config,
        "training_seconds": float(training_seconds),
        "epochs_completed": int(
            len(history.history.get("loss", []))
        ),
        "best_validation_loss": float(np.min(validation_losses)),
        **validation_results,
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
    }

    write_json(
        run_output_dir / f"{run_name}_results.json",
        result,
    )

    save_training_curves(
        history.history,
        run_output_dir,
        run_name,
    )

    pd.DataFrame({
        "path": train_paths,
        "split": "train",
        "label": y_train,
    }).to_csv(
        run_output_dir / "train_manifest.csv",
        index=False,
    )

    pd.DataFrame({
        "path": validation_paths,
        "split": "validation",
        "label": y_validation,
    }).to_csv(
        run_output_dir / "validation_manifest.csv",
        index=False,
    )

    print(f"\nBest checkpoint: {checkpoint_path}")
    print(f"Training time (seconds): {training_seconds:.2f}")
    print(f"Best validation loss: {result['best_validation_loss']:.6f}")
    print(
        f"Validation accuracy: "
        f"{result['validation_accuracy']:.4f}"
    )

    del model
    del best_model
    del x_train
    del x_validation
    del y_train
    del y_validation

    keras.backend.clear_session()

    return result


# ---------------------------------------------------------------------
# Experiment summaries
# ---------------------------------------------------------------------

def summarize_runs(
    results: List[Dict],
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)

    frame = pd.DataFrame(results)

    frame.to_csv(
        output_dir / "training_run_results.csv",
        index=False,
    )

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
        "training_seconds",
    ]

    available_columns = [
        column for column in metric_columns
        if column in frame.columns
    ]

    summary = frame.groupby(
        ["dataset", "optimizer"]
    )[available_columns].agg(["mean", "std", "count"])

    summary.to_csv(
        output_dir / "training_mean_sd_summary.csv"
    )

    summary_json = {}

    for (dataset, optimizer), group in frame.groupby(
        ["dataset", "optimizer"]
    ):
        key = f"{dataset}__{optimizer}"
        summary_json[key] = {}

        for metric in available_columns:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            summary_json[key][metric] = {
                "mean": (
                    float(values.mean())
                    if len(values) else None
                ),
                "sd": (
                    float(values.std(ddof=1))
                    if len(values) > 1 else None
                ),
                "n": int(len(values)),
            }

    write_json(
        output_dir / "training_mean_sd_summary.json",
        summary_json,
    )

    print(f"\nTraining summaries saved to: {output_dir}")


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="VGG19/HNAdam brain-tumor classification training"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
        help="Dataset directory under --data-root",
    )
    parser.add_argument(
        "--data-root",
        default="data",
    )
    parser.add_argument(
        "--results-root",
        default="results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=["HNAdam", "Adam", "AMSGrad", "SGD", "AdamW"],
        default=["HNAdam"],
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Set to 0 to disable early stopping.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative.")

    if not args.seeds:
        raise ValueError("At least one random seed must be provided.")

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seed values must be unique.")

    dataset_dir = Path(args.data_root) / args.dataset

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split

        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Expected existing split directory: {split_dir}"
            )

    all_results = []

    for optimizer_name in args.optimizers:
        for seed in args.seeds:
            result = run_one_experiment(
                args=args,
                dataset_dir=dataset_dir,
                dataset=args.dataset,
                optimizer_name=optimizer_name,
                seed=seed,
            )
            all_results.append(result)

    summary_dir = (
        Path(args.results_root)
        / args.dataset
        / "training_summary"
    )

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
