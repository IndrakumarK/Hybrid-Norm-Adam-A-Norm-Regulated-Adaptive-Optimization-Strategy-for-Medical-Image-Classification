
# Dataset Preparation

This directory contains the four brain MRI datasets used in the HNAdam-based binary brain tumor classification experiments:

1. SARTAJ Br35H
2. Navoneel Chakrabarty
3. Masoud Nickparvar
4. Indk214

## 1. Dataset Organization

The repository expects the datasets to be organized as follows:

```text
data/
├── SARTAJ/
│   ├── train/
│   │   ├── no_tumor/
│   │   └── tumor/
│   ├── val/
│   │   ├── no_tumor/
│   │   └── tumor/
│   └── test/
│       ├── no_tumor/
│       └── tumor/
│
├── Navoneel/
│   ├── train/
│   │   ├── no_tumor/
│   │   └── tumor/
│   ├── val/
│   │   ├── no_tumor/
│   │   └── tumor/
│   └── test/
│       ├── no_tumor/
│       └── tumor/
│
├── Masoud/
│   └── [same train/val/test structure]
│
└── Indk214/
    └── [same train/val/test structure]
```

## 2. Binary Class Mapping

The experiments address binary brain tumor detection.

- `no_tumor`: Non-tumor images (label 0).
- `tumor`: Tumor images (label 1).

For the original four-class SARTAJ Br35H dataset, glioma, meningioma, and pituitary tumor images are grouped into the tumor class, while no-tumor images are assigned to the non-tumor class.

The resulting classification task uses a single sigmoid output neuron and binary cross-entropy loss.

## 3. Data Partitioning

The training pipeline uses the existing training, validation, and testing directories. It does not automatically generate new random partitions.

- Training data are used for model optimization.
- Validation data are used for model selection.
- Testing data are reserved for final performance evaluation.

Preserve the original dataset partitions where applicable. For datasets requiring additional partitioning, document the partitioning procedure and random seed used.

All optimizer comparisons and ablation experiments must use consistent data partitions.

## 4. Duplicate-Image Assessment

Duplicate-image assessment is important when combining multiple MRI datasets.

The experimental workflow considers:

- SHA-256 hashing for identifying exact duplicate images.
- Perceptual hashing (pHash) for identifying potential near-duplicate images.

Duplicate detection and any subsequent exclusion must be documented. Do not silently remove images or change the test-set composition during evaluation.

## 5. Image Preprocessing

Images are prepared for the VGG19 classification model using the preprocessing procedure implemented in the repository.

The expected model input dimensions are:

- Image size: 240 × 240 pixels.
- Channels: 3 (RGB).
- Pixel normalization: [0, 1].

The same preprocessing procedure must be applied consistently during training, validation, and testing.

## 6. Data Availability and Privacy

The MRI datasets are not included in this GitHub repository. Users must obtain the datasets from their respective authorized sources and comply with their usage and distribution conditions.

Do not upload private clinical MRI images, patient-identifiable information, or restricted dataset content to GitHub.

Only the directory structure and dataset preparation instructions are provided here.

## 7. Reproducibility

Before running the experiments, verify that:

- All four dataset directories are correctly organized.
- Both binary classes are present in each required partition.
- The image counts agree with the dataset records.
- Exact and near-duplicate image handling is documented.
- Training, validation, and testing partitions are kept consistent across optimizer comparisons.
- The test set is not used for training or hyperparameter selection.
