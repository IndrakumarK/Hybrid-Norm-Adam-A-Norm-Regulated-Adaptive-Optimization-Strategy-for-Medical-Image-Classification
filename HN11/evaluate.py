"""
HNAdam / VGG19: Unified preprocessing, evaluation and analysis
==============================================================

This single script handles:
  1. Image preprocessing for an ALREADY DEFINED train/val/test split.
  2. Exact and optional perceptual duplicate screening.
  3. Evaluation of saved binary VGG19 checkpoints (five seeds).
  4. Per-run and mean +/- SD metric summaries.
  5. Paired optimizer / ablation statistical comparisons from saved run results.
  6. Confusion matrices, ROC, PR and calibration plots.
  7. Qualitative Grad-CAM++ maps.
  8. Parameter count, inference latency and TensorFlow GPU-memory profiling.

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve, brier_score_loss
)

try:
    from scipy import stats
except ImportError:
    stats = None

# ----------------------------- Configuration -----------------------------

SEEDS = [42, 52, 62, 72, 82]
IMAGE_SIZE = (240, 240)
CLASS_TO_LABEL = {
    "no_tumor": 0, "notumor": 0, "no tumor": 0, "healthy": 0, "normal": 0,
    "tumor": 1, "glioma": 1, "meningioma": 1, "pituitary": 1,
    "pituitary_tumor": 1, "brain_tumor": 1
}
LABEL_NAMES = ["No Tumor", "Tumor"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------- General helpers -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def normalized_class_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def class_label(folder_name: str) -> int:
    key = normalized_class_name(folder_name)
    if key not in CLASS_TO_LABEL:
        raise ValueError(
            f"Unrecognized class folder '{folder_name}'. Expected no_tumor "
            "or tumor (glioma/meningioma/pituitary are mapped to tumor)."
        )
    return CLASS_TO_LABEL[key]


def list_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(
    image: np.ndarray,
    size: Tuple[int, int] = IMAGE_SIZE,
    use_clahe: bool = False
) -> np.ndarray:
    """RGB resize; optional CLAHE in LAB colour space."""
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    if use_clahe:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return image


def exact_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(image: np.ndarray) -> int:
    """Simple DCT perceptual hash; Hamming distance can screen near-duplicates."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low = dct[:8, :8].flatten()
    median = np.median(low[1:])
    bits = low > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ----------------------------- Preprocessing -----------------------------

def preprocess_dataset(args) -> None:
    """
    Processes an existing split directory without changing split membership.

    Expected input:
      data/DATASET/train/<class>/*
      data/DATASET/val/<class>/*
      data/DATASET/test/<class>/*

    Output mirrors this structure under --processed-root.
    """
    source = Path(args.data_root) / args.dataset
    destination = Path(args.processed_root) / args.dataset
    report_dir = ensure_dir(Path(args.results_root) / args.dataset / "preprocessing")
    records = []
    exact_seen = {}
    hash_records = []

    for split in ("train", "val", "test"):
        split_dir = source / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing {split_dir}. This script requires pre-existing "
                "train/val/test folders and will not invent a split."
            )

        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = class_label(class_dir.name)
            for path in list_images(class_dir):
                try:
                    digest = exact_sha256(path)
                    image = load_rgb(path)
                    phash = perceptual_hash(image)
                    if digest in exact_seen:
                        records.append({
                            "path": str(path), "split": split,
                            "original_class": class_dir.name, "label": label,
                            "sha256": digest, "phash": f"{phash:016x}",
                            "status": "exact_duplicate",
                            "duplicate_of": exact_seen[digest]
                        })
                        continue
                    exact_seen[digest] = str(path)
                    processed = preprocess_image(
                        image, tuple(args.image_size), args.clahe
                    )
                    target_class = "no_tumor" if label == 0 else "tumor"
                    out_dir = ensure_dir(destination / split / target_class)
                    out_path = out_dir / f"{path.stem}.png"
                    # Avoid overwriting files with identical stems.
                    suffix = 1
                    while out_path.exists():
                        out_path = out_dir / f"{path.stem}_{suffix}.png"
                        suffix += 1
                    cv2.imwrite(
                        str(out_path),
                        cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    )
                    records.append({
                        "path": str(path), "output_path": str(out_path),
                        "split": split, "original_class": class_dir.name,
                        "label": label, "sha256": digest,
                        "phash": f"{phash:016x}", "status": "processed",
                        "duplicate_of": ""
                    })
                    hash_records.append((split, str(path), phash))
                except Exception as exc:
                    records.append({
                        "path": str(path), "split": split,
                        "original_class": class_dir.name,
                        "status": "error", "error": str(exc)
                    })

    # Report potential cross-split near-duplicate pairs. These are flagged,
    # not automatically removed, because visual verification is required.
    near_pairs = []
    for i in range(len(hash_records)):
        split_a, path_a, hash_a = hash_records[i]
        for j in range(i + 1, len(hash_records)):
            split_b, path_b, hash_b = hash_records[j]
            if split_a != split_b and hamming_distance(hash_a, hash_b) <= args.phash_distance:
                near_pairs.append({
                    "split_a": split_a, "path_a": path_a,
                    "split_b": split_b, "path_b": path_b,
                    "hamming_distance": hamming_distance(hash_a, hash_b),
                    "status": "manual_review_required"
                })

    pd.DataFrame(records).to_csv(report_dir / "preprocessing_manifest.csv", index=False)
    pd.DataFrame(near_pairs).to_csv(report_dir / "near_duplicate_candidates.csv", index=False)

    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for cls in ("no_tumor", "tumor"):
            counts[split][cls] = len(list_images(destination / split / cls))
    write_json(report_dir / "dataset_counts.json", counts)

    print(f"Preprocessed images saved to: {destination}")
    print(f"Manifest and duplicate reports saved to: {report_dir}")
    print("Near-duplicate candidates are flagged for manual verification; "
          "they are not silently deleted.")


# ----------------------------- Data loading -----------------------------

def load_split(data_root: Path, dataset: str, split: str, image_size=IMAGE_SIZE):
    """
    Loads images from data_root/dataset/split/{no_tumor,tumor}.
    Returns float32 RGB arrays in [0,1], integer labels and file paths.
    """
    split_dir = data_root / dataset / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    xs, ys, paths = [], [], []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = class_label(class_dir.name)
        for path in list_images(class_dir):
            image = preprocess_image(load_rgb(path), image_size, False)
            xs.append(image.astype(np.float32) / 255.0)
            ys.append(label)
            paths.append(str(path))

    if not xs:
        raise ValueError(f"No images found in {split_dir}")

    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int32)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            f"{split_dir} must contain both binary classes. "
            f"Found labels: {np.unique(y).tolist()}"
        )
    return x, y, paths


# ----------------------------- Metrics and plots -----------------------------

def expected_calibration_error(y_true, probabilities, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities).reshape(-1)
    confidence = np.maximum(p, 1.0 - p)
    prediction = (p >= 0.5).astype(int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = ((confidence >= edges[i]) &
                (confidence < edges[i + 1] if i < n_bins - 1
                 else confidence <= edges[i + 1]))
        if np.any(mask):
            ece += np.mean(mask) * abs(
                np.mean(prediction[mask] == y_true[mask]) -
                np.mean(confidence[mask])
            )
    return float(ece)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities).reshape(-1)
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    result = {
        "n_test": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "auroc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) == 2 else float("nan"),
        "auprc": float(average_precision_score(y_true, p)) if len(np.unique(y_true)) == 2 else float("nan"),
        "ece_15_bins": expected_calibration_error(y_true, p, 15),
        "brier_score": float(brier_score_loss(y_true, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }
    return result, pred, cm


def save_evaluation_plots(y_true, p, pred, cm, out_dir: Path, tag: str):
    ensure_dir(out_dir)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion matrix - {tag}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_confusion_matrix.png", dpi=600, bbox_inches="tight")
    plt.close()

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        plt.figure(figsize=(6.5, 5.5))
        plt.plot(fpr, tpr, lw=2, label=f"AUROC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"ROC curve - {tag}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}_roc.png", dpi=600, bbox_inches="tight")
        plt.close()

        precision, recall, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)
        plt.figure(figsize=(6.5, 5.5))
        plt.plot(recall, precision, lw=2, label=f"AUPRC = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-recall curve - {tag}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}_pr_curve.png", dpi=600, bbox_inches="tight")
        plt.close()

    # Reliability diagram (binary, confidence vs accuracy).
    confidence = np.maximum(p, 1.0 - p)
    correctness = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, 16)
    bin_conf, bin_acc = [], []
    for i in range(15):
        mask = (confidence >= bins[i]) & (
            confidence < bins[i + 1] if i < 14 else confidence <= bins[i + 1]
        )
        if np.any(mask):
            bin_conf.append(float(np.mean(confidence[mask])))
            bin_acc.append(float(np.mean(correctness[mask])))
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if bin_conf:
        plt.plot(bin_conf, bin_acc, "o-", label="Model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability diagram - {tag}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_reliability.png", dpi=600, bbox_inches="tight")
    plt.close()


# ----------------------------- Model evaluation -----------------------------

def resolve_checkpoint(model_root: Path, dataset: str, run_name: str, seed: Optional[int]):
    candidates = []
    if seed is not None:
        candidates += [
            model_root / dataset / f"{run_name}_seed_{seed}.keras",
            model_root / dataset / f"best_model_{run_name}_seed_{seed}.keras",
            model_root / f"best_model_{dataset}_{run_name}_seed_{seed}.keras",
            model_root / f"best_model_{dataset}_seed_{seed}.keras",
        ]
    candidates += [
        model_root / dataset / f"{run_name}.keras",
        model_root / f"best_model_{dataset}_{run_name}.keras",
        model_root / f"best_model_{dataset}.keras",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No checkpoint found. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def evaluate_checkpoint(checkpoint: Path, x_test, y_test, out_dir: Path,
                        tag: str, threshold=0.5):
    model = keras.models.load_model(str(checkpoint), compile=False)
    output_shape = model.output_shape
    if isinstance(output_shape, list) or output_shape[-1] != 1:
        raise ValueError(
            f"{checkpoint} does not have one sigmoid output; got {output_shape}. "
            "This script expects the manuscript's binary single-output model."
        )

    # Warm-up is excluded from measured inference time.
    _ = model.predict(x_test[:min(2, len(x_test))], verbose=0)
    start = time.perf_counter()
    probabilities = model.predict(x_test, verbose=0).reshape(-1)
    elapsed = time.perf_counter() - start
    latency_ms = (elapsed / len(x_test)) * 1000.0

    metrics, predictions, cm = calculate_metrics(y_test, probabilities, threshold)
    metrics.update({
        "checkpoint": str(checkpoint),
        "inference_total_seconds": float(elapsed),
        "inference_ms_per_image": float(latency_ms),
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(
            int(np.prod(v.shape)) for v in model.trainable_weights
        )),
        "non_trainable_parameters": int(sum(
            int(np.prod(v.shape)) for v in model.non_trainable_weights
        ))
    })

    out_dir = ensure_dir(out_dir)
    write_json(out_dir / f"{tag}_metrics.json", metrics)

    with (out_dir / f"{tag}_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=LABEL_NAMES, digits=4, zero_division=0
        ))
        f.write("\nConfusion matrix (rows=true, columns=predicted):\n")
        f.write(np.array2string(cm))
        f.write("\n\nMetrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    save_evaluation_plots(y_test, probabilities, predictions, cm, out_dir, tag)
    np.savez_compressed(
        out_dir / f"{tag}_predictions.npz",
        y_true=y_test, y_probability=probabilities, y_pred=predictions
    )
    return metrics, model, probabilities


def evaluate_all(args):
    data_root = Path(args.data_root)
    model_root = Path(args.model_root)
    results_root = ensure_dir(Path(args.results_root))
    datasets = args.datasets or [args.dataset]
    runs = args.runs or ["HNAdam"]

    all_rows = []
    for dataset in datasets:
        x_test, y_test, test_paths = load_split(
            data_root, dataset, "test", tuple(args.image_size)
        )
        write_json(results_root / dataset / "test_manifest.json", {
            "test_count": len(test_paths),
            "test_paths": test_paths,
            "class_mapping": {"no_tumor": 0, "tumor": 1}
        })

        for run_name in runs:
            seeds = args.seeds if args.seeds else [None]
            for seed in seeds:
                tag = f"{run_name}_seed_{seed}" if seed is not None else run_name
                checkpoint = resolve_checkpoint(model_root, dataset, run_name, seed)
                out_dir = results_root / dataset / run_name / tag
                metrics, _, _ = evaluate_checkpoint(
                    checkpoint, x_test, y_test, out_dir, tag, args.threshold
                )
                row = {
                    "dataset": dataset, "method": run_name,
                    "seed": seed if seed is not None else "",
                    **metrics
                }
                all_rows.append(row)
                print(f"[{dataset} | {tag}] accuracy={metrics['accuracy']:.4f}")

    frame = pd.DataFrame(all_rows)
    summary_dir = ensure_dir(results_root / "final_summary")
    frame.to_csv(summary_dir / "all_run_metrics.csv", index=False)
    write_json(summary_dir / "all_run_metrics.json", all_rows)

    # Mean and SD across actual runs, grouped by dataset and method.
    metric_cols = [
        "accuracy", "precision", "sensitivity", "specificity", "macro_f1",
        "balanced_accuracy", "mcc", "auroc", "auprc", "ece_15_bins",
        "brier_score", "inference_ms_per_image"
    ]
    available = [c for c in metric_cols if c in frame.columns]
    grouped = frame.groupby(["dataset", "method"])[available].agg(["mean", "std"])
    grouped.to_csv(summary_dir / "mean_sd_summary.csv")
    print(f"\nEvaluation outputs saved under: {results_root}")


# ----------------------------- Statistical analysis -----------------------------

def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Two-sided exact paired sign-flip test; feasible for modest run counts."""
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    observed = abs(np.mean(d))
    if n > 20:
        raise ValueError("Exact sign-flip enumeration limited to 20 paired runs.")
    extreme = 0
    total = 1 << n
    for mask in range(total):
        signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)])
        if abs(np.mean(signs * d)) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def holm_adjust(p_values: List[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def paired_comparison(a: np.ndarray, b: np.ndarray) -> Dict:
    """Paired comparison; rows must represent the same seeds/partitions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"n_pairs": len(a), "note": "At least two matched runs required."}
    diff = a - b
    result = {
        "n_pairs": len(diff),
        "mean_difference": float(np.mean(diff)),
        "sd_difference": float(np.std(diff, ddof=1)),
        "cohens_dz": float(np.mean(diff) / np.std(diff, ddof=1))
            if np.std(diff, ddof=1) > 0 else float("nan"),
        "sign_flip_p": exact_sign_flip_pvalue(diff)
    }
    if stats is not None:
        t = stats.ttest_rel(a, b)
        result["paired_t_statistic"] = float(t.statistic)
        result["paired_t_p"] = float(t.pvalue)
        se = stats.sem(diff)
        ci = stats.t.interval(0.95, len(diff) - 1, loc=np.mean(diff), scale=se)
        result["paired_t_95ci_difference"] = [float(ci[0]), float(ci[1])]
    else:
        result["paired_t_note"] = "SciPy unavailable; t-test and CI not calculated."
    return result


def run_statistical_analysis(args):
    source = Path(args.metrics_csv)
    if not source.is_file():
        raise FileNotFoundError(f"Run evaluation first; missing {source}")
    df = pd.read_csv(source)
    required = {"dataset", "method", "seed", args.metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV missing columns: {sorted(missing)}")

    out_dir = ensure_dir(Path(args.results_root) / "statistical_analysis")
    outputs = []
    pvals, pval_indices = [], []

    for dataset in sorted(df["dataset"].dropna().unique()):
        subset = df[df["dataset"] == dataset]
        methods = sorted(subset["method"].dropna().unique())
        if args.reference not in methods:
            warnings.warn(f"{args.reference} absent for {dataset}; skipping.")
            continue

        ref = subset[subset["method"] == args.reference].set_index("seed")
        for method in methods:
            if method == args.reference:
                continue
            comp = subset[subset["method"] == method].set_index("seed")
            common = ref.index.intersection(comp.index)
            common = [s for s in common if str(s) not in ("", "nan")]
            if len(common) < 2:
                outputs.append({
                    "dataset": dataset, "reference": args.reference,
                    "comparison": method, "note": "Insufficient matched seed runs."
                })
                continue
            result = paired_comparison(
                ref.loc[common, args.metric].to_numpy(),
                comp.loc[common, args.metric].to_numpy()
            )
            row = {
                "dataset": dataset, "reference": args.reference,
                "comparison": method, "metric": args.metric,
                "matched_seeds": [str(x) for x in common], **result
            }
            outputs.append(row)
            if "paired_t_p" in result:
                pvals.append(result["paired_t_p"])
                pval_indices.append(len(outputs) - 1)

    adjusted = holm_adjust(pvals) if pvals else []
    for idx, p_adj in zip(pval_indices, adjusted):
        outputs[idx]["holm_adjusted_p"] = p_adj

    write_json(out_dir / "paired_statistical_tests.json", outputs)
    pd.DataFrame(outputs).to_csv(out_dir / "paired_statistical_tests.csv", index=False)
    print(f"Statistical analysis saved to: {out_dir}")


# ----------------------------- Grad-CAM++ -----------------------------

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                pass
    raise ValueError(
        "No Conv2D layer found. Grad-CAM++ requires an accessible convolutional layer."
    )


def gradcam_plus_plus(model, image_batch, layer_name=None):
    """
    Grad-CAM++ for the positive (tumor) sigmoid output.
    Qualitative visualization only; it does not establish clinical localization.
    """
    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    conv_layer = model.get_layer(layer_name)
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    x = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x, training=False)
        score = predictions[:, 0]
    first = tape.gradient(score, conv_outputs)
    second = tape.gradient(first, conv_outputs)
    third = tape.gradient(second, conv_outputs)

    if first is None or second is None or third is None:
        raise RuntimeError(
            "Could not calculate Grad-CAM++ derivatives for this architecture."
        )

    sum_activations = tf.reduce_sum(conv_outputs, axis=[1, 2], keepdims=True)
    alpha_num = second
    alpha_denom = 2.0 * second + third * sum_activations
    alpha_denom = tf.where(
        tf.abs(alpha_denom) > 1e-8, alpha_denom,
        tf.ones_like(alpha_denom) * 1e-8
    )
    alphas = alpha_num / alpha_denom
    weights = tf.reduce_sum(
        alphas * tf.nn.relu(first), axis=[1, 2], keepdims=True
    )
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (
        tf.reduce_max(heatmap, axis=[1, 2], keepdims=True) + 1e-8
    )
    return heatmap.numpy(), predictions.numpy().reshape(-1)


def generate_gradcam(args):
    dataset = args.dataset
    x_test, y_test, paths = load_split(
        Path(args.data_root), dataset, "test", tuple(args.image_size)
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint else resolve_checkpoint(
        Path(args.model_root), dataset, args.run_name, args.seed
    )
    model = keras.models.load_model(str(checkpoint), compile=False)
    layer_name = args.gradcam_layer or find_last_conv_layer(model)
    out_dir = ensure_dir(Path(args.results_root) / dataset / "gradcam_plus_plus")
    count = min(args.gradcam_samples, len(x_test))
    chosen = np.linspace(0, len(x_test) - 1, count, dtype=int)

    for idx in chosen:
        image = x_test[idx:idx + 1]
        heatmaps, probability = gradcam_plus_plus(model, image, layer_name)
        heat = cv2.resize(heatmaps[0], (image.shape[2], image.shape[1]))
        rgb = np.clip(image[0] * 255.0, 0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        overlay = np.clip(0.55 * rgb + 0.45 * heat_color, 0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb); axes[0].set_title("Original")
        axes[1].imshow(heat, cmap="jet"); axes[1].set_title("Grad-CAM++")
        axes[2].imshow(overlay); axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        pred = int(probability[0] >= 0.5)
        fig.suptitle(
            f"True: {LABEL_NAMES[int(y_test[idx])]} | "
            f"Predicted: {LABEL_NAMES[pred]} | "
            f"P(tumor): {probability[0]:.4f}"
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"gradcam_{idx:04d}.png", dpi=600, bbox_inches="tight")
        plt.close()

        with (out_dir / "gradcam_manifest.csv").open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["index", "image_path", "true_label", "tumor_probability",
                                 "predicted_label", "conv_layer"])
            writer.writerow([idx, paths[idx], int(y_test[idx]), float(probability[0]),
                             pred, layer_name])

    print(f"Grad-CAM++ outputs saved to: {out_dir}")


# ----------------------------- Main / CLI -----------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified HNAdam preprocessing, evaluation and analysis"
    )
    parser.add_argument(
        "--mode",
        choices=["preprocess", "evaluate", "statistics", "gradcam"],
        required=True
    )
    parser.add_argument("--dataset", default="Navoneel",
                        choices=["SARTAJ", "Navoneel", "Masoud", "Indk214"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--processed-root", default="data_processed")
    parser.add_argument("--model-root", default="results")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--image-size", nargs=2, type=int, default=list(IMAGE_SIZE))
    parser.add_argument("--clahe", action="store_true",
                        help="Enable CLAHE only if it matches the reported preprocessing.")
    parser.add_argument("--phash-distance", type=int, default=6)
    parser.add_argument("--runs", nargs="*", default=["HNAdam"],
                        help="Run/checkpoint labels, e.g. HNAdam Adam AMSGrad AdamW.")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--metrics-csv", default="results/final_summary/all_run_metrics.csv")
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
