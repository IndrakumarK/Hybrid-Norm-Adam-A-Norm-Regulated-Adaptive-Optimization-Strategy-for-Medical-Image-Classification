"""
----------------------------------------------------
Evaluation Script for Hybrid Norm Adam (HNAdam)

This script:
- Loads a trained model (saved during training)
- Evaluates ONLY on the held-out TEST set
- Generates classification report
- Generates confusion matrix
- Generates ROC curve (binary datasets only)
- Saves all outputs to /results

Run example:
python evaluate.py --dataset Navoneel
----------------------------------------------------
"""

# ==================================================
# Imports
# ==================================================
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ==================================================
# Custom Imports
# ==================================================
from optimizers.hnadam import HNAdam
from utils.preprocessing import load_dataset

# ==================================================
# Main Evaluation Function
# ==================================================
def main():

    # ------------------------------------------------
    # Argument Parsing
    # ------------------------------------------------
    parser = argparse.ArgumentParser(
        description="HNAdam Evaluation Script"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["SARTAJ", "Navoneel", "Masoud", "Indk214"],
        help="Dataset name used during training"
    )
    args = parser.parse_args()

    # ------------------------------------------------
    # Paths
    # ------------------------------------------------
    model_path = f"results/best_model_{args.dataset}.keras"
    data_dir = f"data/{args.dataset}"
    results_dir = "results"

    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            f"Please run train.py first."
        )

    # ------------------------------------------------
    # Load Model
    # ------------------------------------------------
    print(f"\nLoading trained model: {model_path}")
    model = keras.models.load_model(
        model_path,
        custom_objects={"HNAdam": HNAdam}
    )

    # ------------------------------------------------
    # Load Test Data
    # ------------------------------------------------
    print(f"Loading TEST split for dataset: {args.dataset}")
    X_test, y_test, class_names = load_dataset(
        data_dir,
        split="test"
    )

    num_classes = len(class_names)

    # ------------------------------------------------
    # Prediction
    # ------------------------------------------------
    print("Running inference on test set...")
    y_prob = model.predict(X_test, verbose=0)

    # Binary vs Multiclass handling
    if num_classes == 2:
        y_prob = y_prob.flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_prob)
    else:
        y_pred = np.argmax(y_prob, axis=1)
        auc = roc_auc_score(
            y_test, y_prob, multi_class="ovr"
        )

    # ------------------------------------------------
    # Metrics
    # ------------------------------------------------
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4
    )
    cm = confusion_matrix(y_test, y_pred)

    # ------------------------------------------------
    # Save Text Results
    # ------------------------------------------------
    metrics_path = os.path.join(
        results_dir, f"eval_report_{args.dataset}.txt"
    )

    with open(metrics_path, "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"ROC AUC: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # ------------------------------------------------
    # Confusion Matrix Plot
    # ------------------------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix – {args.dataset}")
    plt.tight_layout()

    cm_path = os.path.join(
        results_dir, f"confusion_matrix_{args.dataset}.png"
    )
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # ------------------------------------------------
    # ROC Curve (Binary Classification Only)
    # ------------------------------------------------
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(
            fpr, tpr,
            label=f"AUC = {auc:.4f}",
            linewidth=2
        )
        plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve – {args.dataset}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        roc_path = os.path.join(
            results_dir, f"roc_{args.dataset}.png"
        )
        plt.savefig(roc_path, dpi=200)
        plt.close()

    # ------------------------------------------------
    # Final Status
    # ------------------------------------------------
    print("\nEvaluation completed successfully")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    if num_classes == 2:
        print("ROC curve saved to /results")

# ==================================================
# Entry Point
# ==================================================
if __name__ == "__main__":
    main()
