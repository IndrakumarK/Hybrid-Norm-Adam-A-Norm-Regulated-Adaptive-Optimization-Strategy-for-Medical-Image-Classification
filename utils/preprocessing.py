"""
Image loading and preprocessing utilities for binary brain MRI classification.
"""

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


CLASS_MAP = {
    "no_tumor": 0,
    "notumor": 0,
    "no_tumour": 0,
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


def class_to_label(folder_name):
    """
    Convert a class directory name into a binary label.

    Label mapping:
        0: No tumor
        1: Tumor
    """
    key = (
        str(folder_name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if key not in CLASS_MAP:
        raise ValueError(
            f"Unknown class directory: '{folder_name}'. "
            f"Supported classes: {list(CLASS_MAP.keys())}"
        )

    return CLASS_MAP[key]


def preprocess_image(path, image_size=(240, 240)):
    """
    Load and preprocess an MRI image.

    Processing steps:
        1. Read the image.
        2. Convert BGR to RGB.
        3. Resize to the specified dimensions.
        4. Normalize pixel values to [0, 1].

    Returns:
        NumPy array with shape (height, width, 3).
    """
    path = Path(path)

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = cv2.resize(
        image,
        image_size,
        interpolation=cv2.INTER_AREA,
    )

    image = image.astype(np.float32) / 255.0

    return image


def list_image_files(folder):
    """
    Recursively collect supported image files.

    Returns:
        Sorted list of image file paths.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {folder}"
        )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {folder}"
        )

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )