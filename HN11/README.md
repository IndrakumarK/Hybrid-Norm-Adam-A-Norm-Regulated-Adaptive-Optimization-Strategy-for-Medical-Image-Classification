# Hybrid Norm Adam (HNAdam)
Norm-Regulated Adaptive Optimization for Robust Medical Image Classification

## Overview
This repository provides the official implementation accompanying the paper:

**Hybrid Norm Adam: A Norm-Regulated Adaptive Optimization Strategy for Robust Medical Image Classification**

The proposed Hybrid Norm Adam (HNAdam) optimizer introduces a norm-regulated regime-switching mechanism that dynamically balances exploration and stability during training. The method is validated on multiple publicly available brain tumor MRI datasets using a VGG19-based classification framework.

## Datasets
The experiments use the following publicly available datasets:

1. **SARTAJdataset Br35H**
   - Multiclass MRI dataset (Glioma, Meningioma, Pituitary, No Tumor)
   - Source: BR35H clinical dataset

2. **Brain MRI Images for Brain Tumor Detection**
   - Binary classification dataset
   - Source: Kaggle (Navoneel Chakrabarty)

3. **Masoud Nickparvar Brain Tumor MRI Dataset**
   - Binary classification dataset
   - Source: Kaggle

4. **Indk214 Brain MRI Dataset**
   - Binary classification dataset
   - Source: Kaggle

Please download datasets from their respective sources and organize them as described below.

## Directory Structure
```
data/
 ├── train/
 │   ├── class_1/
 │   ├── class_2/
 ├── val/
 ├── test/
src/
 ├── models/
 ├── optimizers/
 ├── train.py
 ├── evaluate.py
```

## Preprocessing
- ROI extraction using thresholding and morphological operations
- Resizing to 240×240 resolution
- Class-weight computation for imbalance handling

Details are provided in the Supplementary Material of the paper.

## Training
The VGG19 backbone is initialized with ImageNet weights and frozen.
Only the classification head is trained.

Example command:
```
python train.py --optimizer hnadam --dataset br35h
```

## Optimizers
Supported optimizers:
- SGD
- Adam
- RMSProp
- **Hybrid Norm Adam (HNAdam)** (proposed)

HNAdam dynamically switches between Adam-style and AMSGrad-style updates using a norm-based control signal.

## Evaluation
Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- IoU

Evaluation is performed on held-out test sets.

## Reproducibility
- Fixed random seeds
- Identical training settings across optimizers
- No dataset-specific hyperparameter tuning for HNAdam

## Requirements
- Python 3.8+
- TensorFlow / Keras
- NumPy
- OpenCV
- Scikit-learn

## Citation
If you use this code, please cite:

@article{HNAdam2026,
  title={Hybrid Norm Adam: A Norm-Regulated Adaptive Optimization Strategy for Robust Medical Image Classification},
  author={Indrakumar K, Ravikumar M},
  journal={},
  year={2026}
}

## License
This project is released for research purposes only.
