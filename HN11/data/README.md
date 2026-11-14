# Dataset Usage Instructions

This repository does not include MRI datasets. All datasets used in the experiments are publicly available and must be downloaded from their respective sources. Users are responsible for following the original dataset licensing and usage conditions.

## 1. Folder Structure

All four datasets are used for **binary brain tumor classification**, with the following class labels:

- `tumor`: Label 1
- `no_tumor`: Label 0

The expected directory structure is:

```text
data/
├── SARTAJ/
│   ├── train/
│   │   ├── tumor/
│   │   └── no_tumor/
│   ├── val/
│   │   ├── tumor/
│   │   └── no_tumor/
│   └── test/
│       ├── tumor/
│       └── no_tumor/
│
├── Navoneel/
│   ├── train/
│   │   ├── tumor/
│   │   └── no_tumor/
│   ├── val/
│   └── test/
│
├── Masoud_Nickparvar/
│   ├── train/
│   ├── val/
│   └── test/
│
└── Indk214/
    ├── train/
    ├── val/
    └── test/
```

Each dataset must contain the corresponding `tumor` and `no_tumor` class folders within its training, validation, and testing directories.

## 2. SARTAJ Br35H: Binary Label Mapping

The original SARTAJ Br35H dataset contains four diagnostic categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

For the experiments reported in this study, these four categories are consolidated into two binary classes:

| Original category | Binary class | Label |
|---|---|---:|
| Glioma | Tumor | 1 |
| Meningioma | Tumor | 1 |
| Pituitary tumor | Tumor | 1 |
| No tumor | No tumor | 0 |

Therefore, the SARTAJ dataset must be organized into `tumor` and `no_tumor` folders after applying the binary label mapping.

**Note:** This implementation evaluates tumor detection rather than distinguishing between individual tumor types.

## 3. Supported Datasets

The following datasets are used:

1. **SARTAJ Br35H Dataset** – Sartaj Bhuvaji, Kaggle.
   https://www.kaggle.com/datasets/abushayid/figshare-br35h

2. **Brain MRI Images for Brain Tumor Detection** – Navoneel Chakrabarty, Kaggle.
   https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

3. **Brain Tumor MRI Dataset** – Masoud Nickparvar, Kaggle.
   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

4. **Brain MRI Dataset** – Indk214, Kaggle.
   https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification

## 4. Data Partitioning and Leakage Prevention

The datasets are organized into separate training, validation, and testing subsets.

- Training data are used for model optimization.
- Validation data are used for monitoring and checkpoint selection.
- Testing data are reserved for final performance evaluation.

The test set must not be used during training, validation monitoring, or model selection.

Duplicate assessment using SHA-256 hashing and perceptual hashing (pHash) is described in the associated manuscript.

Patient-specific identifiers were unavailable for the collected datasets. Source-level grouping was considered wherever reliable provenance information was available; however, this does not guarantee patient-level independence.

## 5. Usage Example

To train the model using the Navoneel dataset:

```bash
python train.py --dataset Navoneel
```

To train using the SARTAJ dataset:

```bash
python train.py --dataset SARTAJ
```

Refer to the training configuration and experimental instructions for the optimizer and hyperparameter settings.
