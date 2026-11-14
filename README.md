# VGG19 with Hybrid Norm Adam (HNAdam) for Brain Tumor MRI Classification

## Overview

This repository provides the implementation of the proposed Hybrid Norm Adam (HNAdam) optimizer for brain tumor MRI classification using a VGG19-based deep learning framework.

The work accompanies the manuscript:

**Classification of Brain Tumor MR Images Using VGG19 with a Customized Adaptive Moment Estimation (ADAM) Optimizer**

Submitted to the *Journal of Imaging* (MDPI).

The proposed HNAdam optimizer combines adaptive gradient norm regulation with a hybrid switching mechanism between Adam and AMSGrad updates. The optimizer is evaluated across four publicly available brain MRI datasets and compared with established optimization algorithms.

## Datasets

The experiments use the following publicly available datasets:

### 1. SARTAJ Br35H Dataset

Original categories:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

For this study, the original four categories are consolidated into two classes:

- Tumor (label 1): Glioma, meningioma, and pituitary tumor.
- Non-tumor (label 0): No-tumor images.

Source: Sartaj Bhuvaji — Kaggle  
https://www.kaggle.com/datasets/abushayid/figshare-br35h

### 2. Brain MRI Images for Brain Tumor Detection

Binary classification dataset containing tumor and non-tumor images.

Source: Navoneel Chakrabarty — Kaggle  
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

### 3. Masoud Nickparvar Brain Tumor MRI Dataset

The selected experimental images are organized into tumor and non-tumor categories.

Source: Masoud Nickparvar — Kaggle  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### 4. Indk214 Brain MRI Dataset

The selected experimental images are organized into tumor and non-tumor categories.

Source: Indk214 — Kaggle  
https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification

**Note:** All four datasets are evaluated as binary classification tasks in this study. Please refer to the original dataset sources for their respective usage and licensing conditions.

## Dataset Preparation and Duplicate Assessment

The initial image collection contained 11,476 images.

Duplicate assessment was performed using:

- SHA-256 hashing for exact duplicate identification.
- Perceptual hashing (pHash) for near-duplicate candidate identification.
- Manual examination of candidate near-duplicate images.

The assessment identified 229 exact duplicates and 230 near-duplicate images. Following their removal, 11,017 images remained.

Patient-specific identifiers were unavailable for the collected datasets. Source-level grouping was considered wherever reliable provenance information was available; however, this does not guarantee patient-level independence.

## Directory Structure

```text
project/
│
├── data/
│   ├── SARTAJ_Br35H/
│   ├── Navoneel/
│   ├── Masoud_Nickparvar/
│   └── Indk214/
│
├── src/
│   ├── models/
│   ├── optimizers/
│   ├── datasets/
│   ├── training/
│   └── evaluation/
│
├── experiments/
│   ├── optimizer_comparison/
│   ├── ablation_study/
│   └── statistical_analysis/
│
├── results/
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Preprocessing

The preprocessing pipeline includes the image-processing operations described in the manuscript and supplementary material.

The current repository configuration includes:

- ROI extraction using thresholding and morphological operations.
- Image resizing to 240 × 240 pixels.
- Class-weight computation for class imbalance handling.

Data augmentation, where applicable, is restricted to the training data.

The preprocessing implementation and image dimensions should be kept consistent with the experimental configuration used to generate the reported results.

## Model Architecture

The classification framework uses VGG19 with ImageNet-pretrained initialization.

The reported model configuration contains:

- Total parameters: 21,630,145
- Trainable parameters: 1,605,761
- Non-trainable parameters: 20,024,384

The pretrained feature-extraction layers and trainable classification components follow the architecture described in the manuscript.

## Proposed Optimizer: HNAdam

Hybrid Norm Adam (HNAdam) combines adaptive gradient norm regulation with a hybrid Adam/AMSGrad update mechanism.

The optimizer includes:

1. Adaptive norm calculation.
2. Adaptive scaling parameter \(K_{t0}\).
3. Norm clipping within the interval [1, 4].
4. Fixed switching threshold \(\tau = 2\).
5. AMSGrad update when \(A(t) < 2\).
6. Adam update when \(A(t) \geq 2\).

The mathematical formulation, parameter configuration, numerical safeguards, and update procedure are documented in the optimizer implementation.

The sampling procedure for \(K_{t0}\) must follow the exact implementation and experimental configuration described in the manuscript.

## Optimizer Comparison

The experimental framework compares HNAdam with the following optimization algorithms:

- SGD
- Adam
- AMSGrad
- AdamW
- RAdam
- AdaBelief
- HNAdam (proposed)

All optimizers are evaluated using consistent model architecture, data partitions, preprocessing, augmentation, pretrained initialization, and training conditions. Learning rate and weight decay are tuned independently using equivalent search budgets.

## HNAdam Ablation Study

Three component-wise configurations are evaluated:

| Configuration | Description |
|---|---|
| Ablation 1 | Adam with adaptive norm adjustment |
| Ablation 2 | Adam/AMSGrad switching without adaptive norm adjustment |
| Ablation 3 | Full HNAdam with hybrid switching and adaptive norm regulation |

The ablation experiments are conducted across all four MRI datasets to investigate the individual and combined contributions of the proposed optimizer components.

## Training and Reproducibility

The experiments use five independent random seeds:

```text
42, 52, 62, 72, 82
```

The training, validation, and testing datasets are kept separate.

- Training data are used for model optimization.
- Validation data are used for performance monitoring and checkpoint selection.
- Test data are reserved for final model evaluation.

The test set must not be used for gradient updates, training monitoring, or checkpoint selection.

## Evaluation Metrics

The classification framework includes:

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

IoU is not used as an image-level classification metric.

## Statistical Analysis

Statistical evaluation includes:

- Mean and standard deviation across independent runs.
- Paired t-tests.
- 95% confidence intervals.
- Cohen's paired effect sizes.
- Exact sign-flip permutation tests.
- Holm–Bonferroni correction for multiple comparisons.

The reported experiments use five independent seeds. This relatively small number of runs is acknowledged as a limitation, and additional independent experiments are required for stronger statistical validation.

## Interpretability Analysis

Grad-CAM++ is used to generate visual explanations of the VGG19-based classification predictions.

The generated heatmaps provide qualitative insights into the spatial regions contributing to the model's predictions.

These visualizations do not independently establish clinical validity or guarantee the absence of image artefacts or unintended biases.

## Computational Efficiency and Deployment

The repository documents the model architecture and parameter counts.

Potential deployment improvements, including post-training quantization and model compression, are discussed as future research directions.

Inference latency, peak memory consumption, and quantized-model performance should only be reported when supported by actual measurements.

## Requirements

The software dependencies should match the framework and versions used in the experimental implementation.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## License

This repository is intended for academic and research purposes. Dataset usage remains subject to the respective licenses and terms of the original dataset providers.

## Citation

If you use this implementation, please cite the associated manuscript:

**Indrakumar K, et al.** Classification of Brain Tumor MR Images Using VGG19 with a Customized Adaptive Moment Estimation (ADAM) Optimizer. *Journal of Imaging*. Manuscript ID: jimaging-4679286.

Update the citation with the final publication details once available.
