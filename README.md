# Hybrid Norm Adam (HNAdam) for Brain MRI Binary Classification

Research implementation of the proposed **Hybrid Norm Adam (HNAdam)** optimizer for brain tumor classification using a VGG19-based deep learning framework.

## Overview

This repository presents the implementation of Hybrid Norm Adam (HNAdam), a norm-regulated adaptive optimization strategy designed to improve optimization behavior and generalization in brain MRI classification.

The proposed optimizer combines adaptive gradient norm regulation with a hybrid switching mechanism between Adam and AMSGrad updates. The framework is evaluated using four publicly available brain MRI datasets and a VGG19-based binary classification architecture.

**Associated manuscript:**

*Classification of Brain Tumor MR Images Using VGG19 with a Customized Adaptive Moment Estimation (ADAM) Optimizer.*

Journal of Imaging (MDPI).

## Datasets

The experiments use four publicly available brain MRI datasets.

### 1. SARTAJ Br35H Dataset

The original categories include glioma, meningioma, pituitary tumor, and no tumor. For this study, they are consolidated into two classes:

- Tumor (label 1): Glioma, meningioma, and pituitary tumor.
- Non-tumor (label 0): No-tumor images.

Source: [Sartaj Bhuvaji – Kaggle](https://www.kaggle.com/datasets/abushayid/figshare-br35h)

### 2. Brain MRI Images for Brain Tumor Detection

A binary classification dataset containing tumor and non-tumor MRI images.

Source: [Navoneel Chakrabarty – Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

### 3. Masoud Nickparvar Brain Tumor MRI Dataset

The selected experimental images are organized into tumor and non-tumor categories.

Source: [Masoud Nickparvar – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### 4. Indk214 Brain MRI Dataset

The selected experimental images are organized into tumor and non-tumor categories.

Source: [Indk214 – Kaggle](https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification)

All four datasets are used for binary brain tumor classification.

## Dataset Preparation and Duplicate Assessment

The initial image collection contained 11,476 images.

Duplicate assessment involved:

- SHA-256 hashing for exact duplicate identification.
- Perceptual hashing (pHash) for near-duplicate candidate identification.
- Manual examination of candidate near-duplicate images.

The assessment identified 229 exact duplicates and 230 near-duplicate images. Following their removal, 11,017 images remained.

Patient-specific identifiers were unavailable for the collected datasets. Source-level grouping was considered wherever reliable provenance information was available.

## Repository Structure

```text
HN11/
│
├── data/
│   └── README.md
│
├── models/
│   └── vgg19_model.py
│
├── optimizers/
│   ├── __init__.py
│   └── hnadam.py
│
├── experiments/
│
├── utils/
│
├── tests/
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Preprocessing

The image preprocessing pipeline includes:

- ROI extraction using thresholding and morphological operations.
- Image resizing to 240 × 240 pixels.
- RGB image conversion and normalization.
- Class-weight computation for handling class imbalance.
- Data augmentation for training samples, where applicable.

## Model Architecture

The classification framework uses **VGG19 with ImageNet-pretrained initialization**.

The reported model configuration contains:

| Parameter | Value |
|---|---:|
| Total parameters | 21,630,145 |
| Trainable parameters | 1,605,761 |
| Non-trainable parameters | 20,024,384 |

The pretrained VGG19 feature-extraction layers are integrated with trainable classification layers for binary tumor classification.

## Proposed Optimizer: HNAdam

**Hybrid Norm Adam (HNAdam)** introduces an adaptive norm-regulated optimization mechanism combined with hybrid Adam and AMSGrad updates.

The main components include:

1. Adaptive gradient norm calculation.
2. Adaptive scaling parameter \(K_{t0}\).
3. Norm regulation within the interval [1, 4].
4. Fixed switching threshold \(\tau = 2\).
5. AMSGrad update when \(A(t) < 2\).
6. Adam update when \(A(t) \geq 2\).

The optimizer dynamically selects the update mechanism according to the adaptive norm regulation strategy.

## Optimizer Comparison

HNAdam is studied alongside established optimization algorithms:

- SGD
- Adam
- AMSGrad
- AdamW
- RAdam
- AdaBelief

The comparison considers classification performance and optimization behavior under the experimental training configurations.

## HNAdam Ablation Study

Three component-wise configurations are considered:

| Configuration | Description |
|---|---|
| Ablation 1 | Adam with adaptive norm adjustment |
| Ablation 2 | Adam/AMSGrad switching without adaptive norm adjustment |
| Ablation 3 | Full HNAdam with hybrid switching and adaptive norm regulation |

The ablation study investigates the individual and combined contributions of adaptive norm regulation and hybrid optimizer switching across the four MRI datasets.

## Training and Reproducibility

Five independent random seeds are used:

`42, 52, 62, 72, 82`

The experimental framework maintains separate training, validation, and testing partitions.

- Training data are used for model optimization.
- Validation data are used for performance monitoring and checkpoint selection.
- Test data are reserved for final model evaluation.

## Evaluation Metrics

The classification framework considers:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- Macro F1-score
- Class-specific F1-score
- Balanced accuracy
- Matthews correlation coefficient (MCC)
- Area under the ROC curve (AUROC)
- Area under the precision-recall curve (AUPRC)
- Expected calibration error (ECE)
- Brier score

## Statistical Analysis

Statistical evaluation includes:

- Mean and standard deviation across independent runs.
- Paired t-tests.
- 95% confidence intervals.
- Cohen's paired effect sizes.
- Exact sign-flip permutation tests.
- Holm–Bonferroni correction for multiple comparisons.

## Interpretability Analysis

Grad-CAM++ is employed to visualize the regions contributing to VGG19 classification predictions.

The generated heatmaps provide qualitative insights into the spatial regions influencing the model's tumor and non-tumor predictions.

## Computational Efficiency and Deployment

The repository includes the model architecture and parameter configuration. Potential deployment improvements, including post-training quantization and model compression, are considered directions for further development.

## Environment and Requirements

The implementation uses TensorFlow 2.10.1 and Python 3.8–3.10.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run training on the Navoneel dataset using one seed:

```bash
python train.py --dataset Navoneel --seeds 42
```

Run five independent seeds:

```bash
python train.py --dataset Navoneel --seeds 42 52 62 72 82
```

For available training options:

```bash
python train.py --help
```

## Evaluation

Run:

```bash
python evaluate.py --help
```

The evaluation framework provides model evaluation, classification metrics, statistical analysis, and Grad-CAM++ visualization options.

## Manuscript Mapping

| Experimental Component | Manuscript Location |
|---|---|
| Ablation study | Section 4.4, Tables 13(a)–13(d), Table 14 |
| Optimizer comparison | Sections 4.3 and 4.5 |
| Grad-CAM++ visualization | Figure 41 |
| Independent random seeds | 42, 52, 62, 72, 82 |

## Citation

If you use this implementation, please cite the associated manuscript:

Indrakumar K, et al. *Classification of Brain Tumor MR Images Using VGG19 with a Customized Adaptive Moment Estimation (ADAM) Optimizer.* Journal of Imaging. Manuscript ID: jimaging-4679286.

Update the citation with the final publication details once available.

## License and Responsible Use

This repository is intended for academic and research purposes. Dataset usage remains subject to the respective licenses and terms of the original dataset providers.

This research implementation is intended to support scientific investigation and is not a clinically validated diagnostic system.
