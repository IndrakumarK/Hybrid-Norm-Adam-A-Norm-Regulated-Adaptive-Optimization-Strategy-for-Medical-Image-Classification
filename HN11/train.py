"""
train.py
----------------------------------------------------

Paper:
Hybrid Norm Adam: A Modified Adaptive Optimization Algorithm
for Improved Generalization and Convergence in Brain Tumor
Classification Using MRI

Supported datasets:
- SARTAJ (4-class)
- Navoneel Chakrabarty (binary)
- Masoud Nickparvar (binary)
- Indk214 (binary)

Run example:
python train.py --dataset Navoneel --epochs 80 --batch_size 32
----------------------------------------------------
"""

# ==================================================
# Imports
# ==================================================
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
import random
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve
)
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=args.batch_size,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, csv_logger, early_stop, lr_scheduler],
    verbose=1
)
import matplotlib.pyplot as plt

# Custom modules
from optimizers.hnadam import HNAdam
from models.vgg19_model import build_vgg19
from utils.preprocessing import load_dataset

# ==================================================
# Reproducibility
# ==================================================
def set_global_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

set_global_seed(42)

# ==================================================
# Main Training Pipeline
# ==================================================
def main():

    # ------------------------------------------------
    # Argument Parser
    # ------------------------------------------------
    parser = argparse.ArgumentParser(
        description="HNAdam Training Script"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["SARTAJ", "Navoneel", "Masoud", "Indk214"]
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    # ------------------------------------------------
    # Dataset Paths
    # ------------------------------------------------
    dataset_paths = {
        "SARTAJ": "data/SARTAJ",
        "Navoneel": "data/Navoneel",
        "Masoud": "data/Masoud",
        "Indk214": "data/Indk214"
    }

    data_dir = dataset_paths[args.dataset]
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. "
            "Refer to data/README.md for the expected structure."
        )

    print(f"\n?? Loading dataset: {args.dataset}")
    X, y, class_names = load_dataset(data_dir)
    num_classes = 1 if len(class_names) == 2 else len(class_names)

    # ------------------------------------------------
    # Train / Validation / Test Split
    # ------------------------------------------------
    total_samples = len(X)
    train_end = int(0.70 * total_samples)
    val_end = int(0.85 * total_samples)

    X_train, X_val, X_test = (
        X[:train_end],
        X[train_end:val_end],
        X[val_end:]
    )
    y_train, y_val, y_test = (
        y[:train_end],
        y[train_end:val_end],
        y[val_end:]
    )

    print(f"Samples ? Train: {len(X_train)}, "
          f"Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Classes ? {class_names}")

    # ------------------------------------------------
    # Model Construction
    # ------------------------------------------------
    model = build_vgg19(num_classes=num_classes)

    optimizer = HNAdam(
        learning_rate=args.lr,
        alpha=0.01,
        beta=0.001
    )

    loss_fn = (
        "binary_crossentropy"
        if num_classes == 1
        else "sparse_categorical_crossentropy"
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    # ------------------------------------------------
    # Class Imbalance Handling
    # ------------------------------------------------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # ------------------------------------------------
    # Callbacks
    # ------------------------------------------------
    os.makedirs("results", exist_ok=True)

    model_path = f"results/best_model_{args.dataset}.keras"
    metrics_path = f"results/metrics_{args.dataset}.txt"
    history_path = f"results/history_{args.dataset}.csv"

    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    csv_logger = CSVLogger(history_path)

    # ------------------------------------------------
    # Training
    # ------------------------------------------------
    print("\n?? Training started...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=[checkpoint, csv_logger],
        verbose=1
    )

    # ------------------------------------------------
    # Evaluation (TEST SET)
    # ------------------------------------------------
    print("\n?? Evaluating best model on test set...")
    best_model = keras.models.load_model(
        model_path,
        custom_objects={"HNAdam": HNAdam}
    )

    if num_classes == 1:
        y_prob = best_model.predict(X_test).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = best_model.predict(X_test)
        y_pred = np.argmax(y_prob, axis=1)
        auc = roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr"
        )

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )
    cm = confusion_matrix(y_test, y_pred)

    # ------------------------------------------------
    # Save Metrics
    # ------------------------------------------------
    with open(metrics_path, "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"ROC AUC: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # ------------------------------------------------
    # ROC Curve (Binary Only)
    # ------------------------------------------------
    if num_classes == 1:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(
            fpr, tpr,
            label=f"HNAdam (AUC={auc:.4f})",
            linewidth=2
        )
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve – {args.dataset}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(
            f"results/roc_{args.dataset}.png",
            dpi=200
        )
        plt.close()

    print("\n? Training & evaluation completed")
    print(f"?? Best model saved at: {model_path}")
    print(f"?? Metrics saved at: {metrics_path}")

# ==================================================
# Entry Point
# ==================================================
if __name__ == "__main__":
    main()
