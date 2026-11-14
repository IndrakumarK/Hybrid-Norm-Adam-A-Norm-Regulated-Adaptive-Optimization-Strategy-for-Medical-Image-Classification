"""
Unified preprocessing, evaluation, statistical analysis, and Grad-CAM++
for binary brain MRI classification with VGG19/HNAdam.

Expected dataset structure:
    data/<DATASET>/<split>/<class>/

Supported splits:
    train, val, test

Binary labels:
    no_tumor = 0
    tumor    = 1

The script preserves existing split membership. It does not create random
or patient-level train/validation/test splits.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from scipy import stats
except ImportError:
    stats = None


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEEDS = [42, 52, 62, 72, 82]
IMAGE_SIZE = (240, 240)
DATASETS = ["SARTAJ", "Navoneel", "Masoud", "Indk214"]

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

LABEL_NAMES = ["No Tumor", "Tumor"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, default=json_default, allow_nan=True)


def normalized_class_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def class_label(folder_name: str) -> int:
    key = normalized_class_name(folder_name)
    if key not in CLASS_TO_LABEL:
        raise ValueError(
            f"Unrecognized class folder '{folder_name}'. "
            "Expected a no-tumor or tumor class directory."
        )
    return CLASS_TO_LABEL[key]


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(
    image: np.ndarray,
    size: Tuple[int, int] = IMAGE_SIZE,
    use_clahe: bool = False,
) -> np.ndarray:
    """Resize RGB image; optionally apply CLAHE in LAB colour space."""
    if image is None or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected a valid three-channel RGB image.")

    image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)

    if use_clahe:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)
        image = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2RGB,
        )

    return image


def exact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: np.ndarray) -> int:
    """Calculate a 64-bit DCT perceptual hash."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low_frequency = dct[:8, :8].flatten()
    median = np.median(low_frequency[1:])
    value = 0
    for bit in low_frequency > median:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


# ---------------------------------------------------------------------
# Dataset preprocessing and duplicate screening
# ---------------------------------------------------------------------

def preprocess_dataset(args) -> None:
    source = Path(args.data_root) / args.dataset
    destination = Path(args.processed_root) / args.dataset
    report_dir = ensure_dir(
        Path(args.results_root) / args.dataset / "preprocessing"
    )

    if not source.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {source}")

    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {destination}. "
                "Use --overwrite to replace its contents."
            )
        shutil.rmtree(destination)

    ensure_dir(destination)

    records = []
    exact_seen = {}
    hash_records = []
    label_conflicts = []

    for split in ("train", "val", "test"):
        split_dir = source / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {split_dir}; existing split directories are required."
            )

        class_dirs = sorted(
            path for path in split_dir.iterdir() if path.is_dir()
        )
        if not class_dirs:
            raise ValueError(f"No class directories found in {split_dir}")

        for class_dir in class_dirs:
            label = class_label(class_dir.name)

            for image_path in list_images(class_dir):
                try:
                    digest = exact_sha256(image_path)

                    if digest in exact_seen:
                        previous = exact_seen[digest]
                        previous_label = previous["label"]

                        status = (
                            "exact_duplicate"
                            if previous_label == label
                            else "duplicate_label_conflict"
                        )

                        records.append({
                            "path": str(image_path),
                            "split": split,
                            "original_class": class_dir.name,
                            "label": label,
                            "sha256": digest,
                            "status": status,
                            "duplicate_of": previous["path"],
                        })

                        if previous_label != label:
                            label_conflicts.append({
                                "path": str(image_path),
                                "duplicate_of": previous["path"],
                                "label": label,
                                "previous_label": previous_label,
                            })
                        continue

                    image = load_rgb(image_path)
                    phash = perceptual_hash(image)
                    processed = preprocess_image(
                        image,
                        tuple(args.image_size),
                        args.clahe,
                    )

                    target_class = "no_tumor" if label == 0 else "tumor"
                    output_dir = ensure_dir(
                        destination / split / target_class
                    )

                    output_path = output_dir / f"{image_path.stem}.png"
                    suffix = 1
                    while output_path.exists():
                        output_path = output_dir / (
                            f"{image_path.stem}_{suffix}.png"
                        )
                        suffix += 1

                    success = cv2.imwrite(
                        str(output_path),
                        cv2.cvtColor(processed, cv2.COLOR_RGB2BGR),
                    )
                    if not success:
                        raise IOError(f"Could not write image: {output_path}")

                    exact_seen[digest] = {
                        "path": str(image_path),
                        "label": label,
                    }

                    hash_records.append((split, str(image_path), phash))

                    records.append({
                        "path": str(image_path),
                        "output_path": str(output_path),
                        "split": split,
                        "original_class": class_dir.name,
                        "label": label,
                        "sha256": digest,
                        "phash": f"{phash:016x}",
                        "status": "processed",
                        "duplicate_of": "",
                    })

                except Exception as exc:
                    records.append({
                        "path": str(image_path),
                        "split": split,
                        "original_class": class_dir.name,
                        "status": "error",
                        "error": str(exc),
                    })

    near_pairs = []
    for i, (split_a, path_a, hash_a) in enumerate(hash_records):
        for split_b, path_b, hash_b in hash_records[i + 1:]:
            if split_a == split_b:
                continue
            distance = hamming_distance(hash_a, hash_b)
            if distance <= args.phash_distance:
                near_pairs.append({
                    "split_a": split_a,
                    "path_a": path_a,
                    "split_b": split_b,
                    "path_b": path_b,
                    "hamming_distance": distance,
                    "status": "manual_review",
                })

    pd.DataFrame(records).to_csv(
        report_dir / "preprocessing_manifest.csv", index=False
    )
    pd.DataFrame(near_pairs).to_csv(
        report_dir / "near_duplicate_candidates.csv", index=False
    )
    write_json(report_dir / "duplicate_label_conflicts.json", label_conflicts)

    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = {
            cls: len(list_images(destination / split / cls))
            for cls in ("no_tumor", "tumor")
        }
    write_json(report_dir / "dataset_counts.json", counts)

    print(f"Processed images: {destination}")
    print(f"Preprocessing reports: {report_dir}")
    print(f"Potential cross-split near-duplicate pairs: {len(near_pairs)}")

    if label_conflicts:
        raise ValueError(
            f"Found {len(label_conflicts)} exact duplicate(s) with conflicting "
            "binary labels. Review duplicate_label_conflicts.json."
        )


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_split(
    data_root: Path,
    dataset: str,
    split: str,
    image_size: Tuple[int, int] = IMAGE_SIZE,
):
    """Load a split as float32 RGB arrays in [0, 1]."""
    split_dir = data_root / dataset / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    images, labels, paths = [], [], []

    class_dirs = sorted(
        path for path in split_dir.iterdir() if path.is_dir()
    )

    for class_dir in class_dirs:
        label = class_label(class_dir.name)

        for image_path in list_images(class_dir):
            image = preprocess_image(
                load_rgb(image_path),
                image_size,
                use_clahe=False,
            )
            images.append(image.astype(np.float32) / 255.0)
            labels.append(label)
            paths.append(str(image_path))

    if not images:
        raise ValueError(f"No images found in {split_dir}")

    x = np.stack(images).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            f"{split_dir} must contain both classes; "
            f"found labels {np.unique(y).tolist()}."
        )

    return x, y, paths


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def expected_calibration_error(
    y_true,
    probabilities,
    n_bins: int = 15,
    threshold: float = 0.5,
) -> float:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(
        probabilities, dtype=float
    ).reshape(-1)

    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal non-zero length.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be in [0, 1].")

    predictions = (probabilities >= threshold).astype(int)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    correctness = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        if np.any(mask):
            ece += np.mean(mask) * abs(
                np.mean(correctness[mask]) - np.mean(confidence[mask])
            )

    return float(ece)


def calculate_metrics(y_true, probabilities, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(
        probabilities, dtype=float
    ).reshape(-1)

    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal non-zero length.")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("Ground-truth labels must be binary (0 or 1).")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be in [0, 1].")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    both_classes = len(np.unique(y_true)) == 2

    result = {
        "n_test": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "sensitivity": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "macro_f1": float(f1_score(
            y_true, predictions, average="macro", zero_division=0
        )),
        "balanced_accuracy": float(balanced_accuracy_score(
            y_true, predictions
        )),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "auroc": float(roc_auc_score(y_true, probabilities))
        if both_classes else float("nan"),
        "auprc": float(average_precision_score(y_true, probabilities))
        if both_classes else float("nan"),
        "ece_15_bins": expected_calibration_error(
            y_true, probabilities, n_bins=15, threshold=threshold
        ),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": matrix.tolist(),
    }

    return result, predictions, matrix


# ---------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------

def save_evaluation_plots(
    y_true,
    probabilities,
    predictions,
    matrix,
    output_dir: Path,
    tag: str,
) -> None:
    ensure_dir(output_dir)

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion matrix - {tag}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{tag}_confusion_matrix.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    if len(np.unique(y_true)) == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true, probabilities
        )
        auc_value = roc_auc_score(y_true, probabilities)

        plt.figure(figsize=(6.5, 5.5))
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            lw=2,
            label=f"AUROC = {auc_value:.4f}",
        )
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"ROC curve - {tag}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{tag}_roc.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

        precision, recall, _ = precision_recall_curve(
            y_true, probabilities
        )
        ap_value = average_precision_score(y_true, probabilities)

        plt.figure(figsize=(6.5, 5.5))
        plt.plot(
            recall, precision, lw=2,
            label=f"AUPRC = {ap_value:.4f}",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-recall curve - {tag}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{tag}_pr_curve.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

    confidence = np.maximum(probabilities, 1.0 - probabilities)
    correctness = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, 16)
    bin_confidence, bin_accuracy = [], []

    for index in range(15):
        if index == 14:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])

        if np.any(mask):
            bin_confidence.append(float(np.mean(confidence[mask])))
            bin_accuracy.append(float(np.mean(correctness[mask])))

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if bin_confidence:
        plt.plot(bin_confidence, bin_accuracy, "o-", label="Model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability diagram - {tag}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{tag}_reliability.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------
# Checkpoint resolution and evaluation
# ---------------------------------------------------------------------

def resolve_checkpoint(
    model_root: Path,
    dataset: str,
    run_name: str,
    seed: Optional[int],
) -> Path:
    candidates = []

    if seed is not None:
        candidates.extend([
            model_root / dataset / f"{run_name}_seed_{seed}.keras",
            model_root / dataset / f"best_model_{run_name}_seed_{seed}.keras",
            model_root / f"best_model_{dataset}_{run_name}_seed_{seed}.keras",
            model_root / f"best_model_{dataset}_seed_{seed}.keras",
        ])

    candidates.extend([
        model_root / dataset / f"{run_name}.keras",
        model_root / f"best_model_{dataset}_{run_name}.keras",
        model_root / f"best_model_{dataset}.keras",
    ])

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No checkpoint found. Checked:\n" +
        "\n".join(str(path) for path in candidates)
    )


def evaluate_checkpoint(
    checkpoint: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    tag: str,
    threshold: float = 0.5,
):
    model = keras.models.load_model(str(checkpoint), compile=False)

    output_shape = model.output_shape
    if isinstance(output_shape, list) or output_shape[-1] != 1:
        raise ValueError(
            f"Expected one sigmoid output; received output shape {output_shape}."
        )

    warmup_count = min(2, len(x_test))
    _ = model.predict(x_test[:warmup_count], verbose=0)

    start_time = time.perf_counter()
    raw_predictions = model.predict(x_test, verbose=0)
    elapsed = time.perf_counter() - start_time

    if raw_predictions.ndim != 2 or raw_predictions.shape[1] != 1:
        raise ValueError(
            f"Unexpected prediction shape: {raw_predictions.shape}"
        )

    probabilities = raw_predictions.reshape(-1)

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model predictions contain non-finite values.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(
            "Model outputs must be sigmoid probabilities in [0, 1]."
        )

    latency_ms = elapsed * 1000.0 / len(x_test)
    metrics, predictions, matrix = calculate_metrics(
        y_test, probabilities, threshold
    )

    metrics.update({
        "checkpoint": str(checkpoint),
        "inference_total_seconds": float(elapsed),
        "inference_ms_per_image": float(latency_ms),
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(
            np.prod(variable.shape) for variable in model.trainable_weights
        )),
        "non_trainable_parameters": int(sum(
            np.prod(variable.shape) for variable in model.non_trainable_weights
        )),
    })

    ensure_dir(output_dir)
    write_json(output_dir / f"{tag}_metrics.json", metrics)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )

    with (output_dir / f"{tag}_classification_report.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write(report)
        file.write("\nConfusion matrix (rows=true, columns=predicted):\n")
        file.write(np.array2string(matrix))
        file.write("\n\nMetrics:\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")

    save_evaluation_plots(
        y_test, probabilities, predictions, matrix, output_dir, tag
    )

    np.savez_compressed(
        output_dir / f"{tag}_predictions.npz",
        y_true=y_test,
        y_probability=probabilities,
        y_pred=predictions,
    )

    return metrics, model, probabilities


def evaluate_all(args) -> None:
    data_root = Path(args.data_root)
    model_root = Path(args.model_root)
    results_root = ensure_dir(Path(args.results_root))

    datasets = args.datasets if args.datasets else [args.dataset]
    runs = args.runs if args.runs else ["HNAdam"]

    all_rows = []

    for dataset in datasets:
        x_test, y_test, test_paths = load_split(
            data_root,
            dataset,
            "test",
            tuple(args.image_size),
        )

        write_json(
            results_root / dataset / "test_manifest.json",
            {
                "test_count": len(test_paths),
                "test_paths": test_paths,
                "class_mapping": {"no_tumor": 0, "tumor": 1},
            },
        )

        for run_name in runs:
            seeds = args.seeds if args.seeds else [None]

            for seed in seeds:
                tag = (
                    f"{run_name}_seed_{seed}"
                    if seed is not None else run_name
                )
                checkpoint = resolve_checkpoint(
                    model_root, dataset, run_name, seed
                )
                output_dir = (
                    results_root / dataset / run_name / tag
                )

                metrics, _, _ = evaluate_checkpoint(
                    checkpoint,
                    x_test,
                    y_test,
                    output_dir,
                    tag,
                    args.threshold,
                )

                all_rows.append({
                    "dataset": dataset,
                    "method": run_name,
                    "seed": seed if seed is not None else "",
                    **metrics,
                })

                print(
                    f"[{dataset} | {tag}] "
                    f"accuracy={metrics['accuracy']:.4f}",
                    flush=True,
                )

    frame = pd.DataFrame(all_rows)
    summary_dir = ensure_dir(results_root / "final_summary")

    frame.to_csv(summary_dir / "all_run_metrics.csv", index=False)
    write_json(summary_dir / "all_run_metrics.json", all_rows)

    metric_columns = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "auroc",
        "auprc",
        "ece_15_bins",
        "brier_score",
        "inference_ms_per_image",
    ]

    available_columns = [
        column for column in metric_columns if column in frame.columns
    ]

    if not frame.empty:
        summary = frame.groupby(
            ["dataset", "method"]
        )[available_columns].agg(["mean", "std"])
        summary.to_csv(summary_dir / "mean_sd_summary.csv")

    print(f"Evaluation results saved under: {results_root}")


# ---------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------

def exact_sign_flip_pvalue(differences) -> float:
    differences = np.asarray(differences, dtype=float).reshape(-1)
    differences = differences[np.isfinite(differences)]
    n = len(differences)

    if n == 0:
        return float("nan")
    if n > 20:
        raise ValueError(
            "Exact sign-flip enumeration is limited to 20 paired runs."
        )

    observed = abs(np.mean(differences))
    extreme = 0
    total = 1 << n

    for signs_tuple in itertools.product([-1, 1], repeat=n):
        signs = np.asarray(signs_tuple, dtype=float)
        statistic = abs(np.mean(signs * differences))
        if statistic >= observed - 1e-12:
            extreme += 1

    return float(extreme / total)


def holm_adjust(p_values) -> List[float]:
    p_values = np.asarray(p_values, dtype=float)

    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if not np.all(np.isfinite(p_values)):
        raise ValueError("p_values must be finite.")
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must be in [0, 1].")

    count = len(p_values)
    if count == 0:
        return []

    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_maximum = 0.0

    for rank, index in enumerate(order):
        value = (count - rank) * p_values[index]
        running_maximum = max(running_maximum, value)
        adjusted[index] = min(running_maximum, 1.0)

    return adjusted.tolist()


def paired_comparison(reference, candidate) -> Dict:
    """
    Compare paired runs using candidate-minus-reference differences.
    A positive mean difference favours the candidate.
    """
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)

    if (
        reference.ndim != 1
        or candidate.ndim != 1
        or reference.shape != candidate.shape
    ):
        raise ValueError(
            "Inputs must be paired one-dimensional arrays of equal length."
        )

    finite = np.isfinite(reference) & np.isfinite(candidate)
    reference = reference[finite]
    candidate = candidate[finite]

    n = len(reference)
    if n < 2:
        return {
            "n_pairs": n,
            "note": "At least two matched finite runs are required.",
        }

    differences = candidate - reference
    mean_difference = float(np.mean(differences))
    sd_difference = float(np.std(differences, ddof=1))

    result = {
        "n_pairs": n,
        "mean_difference": mean_difference,
        "sd_difference": sd_difference,
        "cohens_dz": (
            float(mean_difference / sd_difference)
            if sd_difference > 0 else float("nan")
        ),
        "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
    }

    if stats is not None:
        if sd_difference == 0:
            if mean_difference == 0:
                t_statistic, p_value = 0.0, 1.0
            else:
                t_statistic, p_value = (
                    float(np.sign(mean_difference) * np.inf),
                    0.0,
                )
            confidence_interval = [mean_difference, mean_difference]
        else:
            test = stats.ttest_rel(candidate, reference)
            t_statistic = float(test.statistic)
            p_value = float(test.pvalue)
            standard_error = stats.sem(differences)
            critical_value = stats.t.ppf(0.975, n - 1)
            confidence_interval = [
                mean_difference - critical_value * standard_error,
                mean_difference + critical_value * standard_error,
            ]

        result.update({
            "paired_t_statistic": t_statistic,
            "paired_t_p": p_value,
            "paired_t_95ci_difference": [
                float(confidence_interval[0]),
                float(confidence_interval[1]),
            ],
        })

    return result


def run_statistical_analysis(args) -> None:
    source = Path(args.metrics_csv)
    if not source.is_file():
        raise FileNotFoundError(
            f"Evaluation results not found: {source}"
        )

    frame = pd.read_csv(source)
    required = {"dataset", "method", "seed", args.metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Metrics CSV is missing columns: {sorted(missing)}"
        )

    output_dir = ensure_dir(
        Path(args.results_root) / "statistical_analysis"
    )

    outputs = []
    p_values = []
    p_value_indices = []

    for dataset in sorted(frame["dataset"].dropna().unique()):
        subset = frame[frame["dataset"] == dataset]
        methods = sorted(subset["method"].dropna().unique())

        if args.reference not in methods:
            warnings.warn(
                f"Reference method '{args.reference}' is absent for {dataset}."
            )
            continue

        reference_frame = (
            subset[subset["method"] == args.reference]
            .drop_duplicates(subset=["seed"])
            .set_index("seed")
        )

        for method in methods:
            if method == args.reference:
                continue

            candidate_frame = (
                subset[subset["method"] == method]
                .drop_duplicates(subset=["seed"])
                .set_index("seed")
            )

            common_seeds = reference_frame.index.intersection(
                candidate_frame.index
            )
            common_seeds = [
                seed for seed in common_seeds
                if str(seed).strip() not in ("", "nan")
            ]

            if len(common_seeds) < 2:
                outputs.append({
                    "dataset": dataset,
                    "reference": args.reference,
                    "comparison": method,
                    "metric": args.metric,
                    "note": "Insufficient matched seed runs.",
                })
                continue

            comparison = paired_comparison(
                reference_frame.loc[common_seeds, args.metric].to_numpy(),
                candidate_frame.loc[common_seeds, args.metric].to_numpy(),
            )

            row = {
                "dataset": dataset,
                "reference": args.reference,
                "comparison": method,
                "metric": args.metric,
                "matched_seeds": [str(seed) for seed in common_seeds],
                **comparison,
            }
            outputs.append(row)

            if "paired_t_p" in comparison:
                p_values.append(comparison["paired_t_p"])
                p_value_indices.append(len(outputs) - 1)

    adjusted_values = holm_adjust(p_values)
    for index, adjusted_p in zip(p_value_indices, adjusted_values):
        outputs[index]["holm_adjusted_p"] = adjusted_p

    write_json(output_dir / "paired_statistical_tests.json", outputs)
    pd.DataFrame(outputs).to_csv(
        output_dir / "paired_statistical_tests.csv",
        index=False,
    )

    print(f"Statistical analysis saved to: {output_dir}")


# ---------------------------------------------------------------------
# Grad-CAM++
# ---------------------------------------------------------------------

def find_last_conv_layer(model):
    """Find the final Conv2D layer, including nested Keras models."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

        if isinstance(layer, keras.Model):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                continue

    raise ValueError("No Conv2D layer was found in the model.")


def gradcam_plus_plus(model, image_batch, layer_name=None):
    """
    Generate Grad-CAM++ maps for the positive (tumor) sigmoid output.
    """
    if layer_name is None:
        conv_layer = find_last_conv_layer(model)
    else:
        try:
            conv_layer = model.get_layer(layer_name)
        except ValueError:
            conv_layer = None
            for layer in model.layers:
                if isinstance(layer, keras.Model):
                    try:
                        conv_layer = layer.get_layer(layer_name)
                        break
                    except ValueError:
                        continue
            if conv_layer is None:
                raise ValueError(
                    f"Convolutional layer '{layer_name}' was not found."
                )

    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )

    images = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv_outputs, predictions = grad_model(
                    images, training=False
                )
                score = tf.reduce_sum(predictions[:, 0])

            first = tape1.gradient(score, conv_outputs)
            if first is None:
                raise RuntimeError("First-order gradients are unavailable.")

        second = tape2.gradient(tf.reduce_sum(first), conv_outputs)
        if second is None:
            raise RuntimeError("Second-order gradients are unavailable.")

    third = tape3.gradient(tf.reduce_sum(second), conv_outputs)
    if third is None:
        raise RuntimeError("Third-order gradients are unavailable.")

    activation_sum = tf.reduce_sum(
        conv_outputs, axis=[1, 2], keepdims=True
    )

    denominator = 2.0 * second + third * activation_sum
    epsilon = tf.keras.backend.epsilon()

    alphas = tf.math.divide_no_nan(
        second,
        denominator + epsilon,
    )

    weights = tf.reduce_sum(
        alphas * tf.nn.relu(first),
        axis=[1, 2],
        keepdims=True,
    )

    heatmaps = tf.reduce_sum(
        weights * conv_outputs,
        axis=-1,
    )
    heatmaps = tf.nn.relu(heatmaps)
    heatmaps = heatmaps / (
        tf.reduce_max(heatmaps, axis=[1, 2], keepdims=True)
        + epsilon
    )

    return heatmaps.numpy(), predictions.numpy().reshape(-1)


def generate_gradcam(args) -> None:
    data_root = Path(args.data_root)
    x_test, y_test, paths = load_split(
        data_root,
        args.dataset,
        "test",
        tuple(args.image_size),
    )

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        checkpoint = resolve_checkpoint(
            Path(args.model_root),
            args.dataset,
            args.run_name,
            args.seed,
        )

    model = keras.models.load_model(str(checkpoint), compile=False)
    layer_name = args.gradcam_layer

    output_dir = ensure_dir(
        Path(args.results_root) / args.dataset / "gradcam_plus_plus"
    )

    count = min(args.gradcam_samples, len(x_test))
    if count < 1:
        raise ValueError("gradcam-samples must be at least 1.")

    selected_indices = np.linspace(
        0, len(x_test) - 1, count, dtype=int
    )

    manifest_rows = []

    for index in selected_indices:
        image_batch = x_test[index:index + 1]
        heatmaps, probabilities = gradcam_plus_plus(
            model,
            image_batch,
            layer_name,
        )

        heatmap = cv2.resize(
            heatmaps[0],
            (image_batch.shape[2], image_batch.shape[1]),
        )

        rgb = np.clip(
            image_batch[0] * 255.0, 0, 255
        ).astype(np.uint8)

        heat_color = cv2.applyColorMap(
            np.uint8(255 * np.clip(heatmap, 0, 1)),
            cv2.COLORMAP_JET,
        )
        heat_color = cv2.cvtColor(
            heat_color, cv2.COLOR_BGR2RGB
        )

        overlay = np.clip(
            0.55 * rgb + 0.45 * heat_color,
            0,
            255,
        ).astype(np.uint8)

        probability = float(probabilities[0])
        prediction = int(probability >= 0.5)

        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb)
        axes[0].set_title("Original")
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM++")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")

        for axis in axes:
            axis.axis("off")

        figure.suptitle(
            f"True: {LABEL_NAMES[int(y_test[index])]} | "
            f"Predicted: {LABEL_NAMES[prediction]} | "
            f"P(tumor): {probability:.4f}"
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / f"gradcam_{index:04d}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(figure)

        manifest_rows.append({
            "index": int(index),
            "image_path": paths[index],
            "true_label": int(y_test[index]),
            "tumor_probability": probability,
            "predicted_label": prediction,
            "conv_layer": layer_name or conv_layer_name(model),
        })

    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "gradcam_manifest.csv",
        index=False,
    )

    print(f"Grad-CAM++ outputs saved to: {output_dir}")


def conv_layer_name(model) -> str:
    return find_last_conv_layer(model).name


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="HNAdam preprocessing, evaluation, statistics, and Grad-CAM++"
    )

    parser.add_argument(
        "--mode",
        choices=["preprocess", "evaluate", "statistics", "gradcam"],
        required=True,
    )
    parser.add_argument(
        "--dataset",
        default="Navoneel",
        choices=DATASETS,
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--processed-root", default="data_processed")
    parser.add_argument("--model-root", default="results")
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=list(IMAGE_SIZE),
    )
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--phash-distance", type=int, default=6)
    parser.add_argument(
        "--runs",
        nargs="*",
        default=["HNAdam"],
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--metrics-csv",
        default="results/final_summary/all_run_metrics.csv",
    )
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--reference", default="HNAdam")
    parser.add_argument("--run-name", default="HNAdam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gradcam-layer", default=None)
    parser.add_argument("--gradcam-samples", type=int, default=8)

    return parser


def main():
    args = build_parser().parse_args()

    if args.image_size[0] <= 0 or args.image_size[1] <= 0:
        raise ValueError("Image dimensions must be positive.")

    if args.phash_distance < 0:
        raise ValueError("phash-distance cannot be negative.")

    if args.mode == "preprocess":
        preprocess_dataset(args)
    elif args.mode == "evaluate":
        evaluate_all(args)
    elif args.mode == "statistics":
        run_statistical_analysis(args)
    elif args.mode == "gradcam":
        generate_gradcam(args)


if __name__ == "__main__":
    main()
