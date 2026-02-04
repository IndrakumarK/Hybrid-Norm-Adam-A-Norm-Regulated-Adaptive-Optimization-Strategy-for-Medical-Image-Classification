# Dataset Usage Instructions

This repository does not include MRI datasets due to licensing restrictions.
All datasets used in the experiments are publicly available.

## Folder Structure (Binary Classification)

For binary datasets (Tumor / No Tumor):

data/DatasetName/
+-- train/
¦   +-- tumor/
¦   +-- no_tumor/
+-- val/
¦   +-- tumor/
¦   +-- no_tumor/
+-- test/
    +-- tumor/
    +-- no_tumor/

## Multiclass Dataset (SARTAJ)

data/SARTAJ/
+-- train/
¦   +-- glioma/
¦   +-- meningioma/
¦   +-- pituitary/
¦   +-- no_tumor/
+-- val/
¦   +-- glioma/
¦   +-- meningioma/
¦   +-- pituitary/
¦   +-- no_tumor/
+-- test/
    +-- glioma/
    +-- meningioma/
    +-- pituitary/
    +-- no_tumor/

## Supported Datasets

- Brain MRI Images for Brain Tumor Detection (Navoneel Chakrabarty, Kaggle)
- Masoud Nickparvar Brain Tumor MRI Dataset (Kaggle)
- Indk214 Brain Tumor MRI Dataset (Kaggle)
- SARTAJdataset Br35H

## Usage Example

python train.py --dataset Navoneel
