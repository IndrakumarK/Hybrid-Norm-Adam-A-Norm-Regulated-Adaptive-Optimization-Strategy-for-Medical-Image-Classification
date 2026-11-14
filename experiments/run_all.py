"""
Run HNAdam experiments across all four datasets and five random seeds.
"""

import subprocess
import sys
from pathlib import Path


DATASETS = [
    "SARTAJ",
    "Navoneel",
    "Masoud",
    "Indk214",
]

SEEDS = [
    "42",
    "52",
    "62",
    "72",
    "82",
]


def main():
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "train.py"

    if not train_script.is_file():
        raise FileNotFoundError(
            f"Training script not found: {train_script}"
        )

    for dataset in DATASETS:
        command = [
            sys.executable,
            str(train_script),
            "--dataset",
            dataset,
            "--seeds",
            *SEEDS,
        ]

        print(
            f"\n{'=' * 60}\n"
            f"Starting experiments for dataset: {dataset}\n"
            f"{'=' * 60}",
            flush=True,
        )

        subprocess.run(
            command,
            cwd=project_root,
            check=True,
        )

        print(
            f"Completed experiments for dataset: {dataset}",
            flush=True,
        )

    print("\nAll dataset experiments completed successfully.")


if __name__ == "__main__":
    main()