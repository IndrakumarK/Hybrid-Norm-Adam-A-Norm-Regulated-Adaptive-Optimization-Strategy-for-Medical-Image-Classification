# Hybrid Norm Adam (HNAdam) for Brain MRI Binary Classification

Research code for binary brain-tumor classification using a VGG19 classifier and the Hybrid Norm Adam optimizer.

## Important reproducibility notice

This package is a consolidated working repository based on the scripts and method details available during preparation. It has **not** been run end-to-end on the original datasets or GPU environment. Before describing results as reproduced, verify the implementation against the final manuscript equations, the original experiment logs, and your actual source repository.

In particular:
- The three ablation labels are recorded, but the first two exact update rules remain blocked pending confirmation against the final equations.
- The current `train.py` supports HNAdam, Adam, AMSGrad, SGD and AdamW. RAdam and AdaBelief are not implemented as verified comparators.
- Grad-CAM++ is included as an experimental helper and must be validated on the trained VGG19 model; derivative availability can depend on the target layer and model graph.
- Do not use the script-generated results to replace manuscript values unless you actually rerun the experiments and reconcile the outputs.

## Structure

```text
HN11/
├── data/README.md
├── models/vgg19_model.py
├── optimizers/hnadam.py
├── experiments/
├── utils/
├── tests/
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Environment

TensorFlow 2.10.1 is pinned in `requirements.txt`. Use Python 3.8–3.10.

Install:

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Data layout

See `data/README.md`. Four dataset folders are expected: `SARTAJ`, `Navoneel`, `Masoud`, and `Indk214`. Each must contain pre-existing `train`, `val`, and `test` partitions, each with binary class folders. The test split is not loaded during training.

## Training

One run:

```bash
python train.py --dataset Navoneel --seeds 42
```

Five runs:

```bash
python train.py --dataset Navoneel --seeds 42 52 62 72 82
```

Repeat for each dataset. The script's exact CLI options are available with:

```bash
python train.py --help
```

## Evaluation

Check:

```bash
python evaluate.py --help
```

The evaluation script supports checkpoint evaluation, summary metrics, statistical analysis and Grad-CAM mode. Refer to its CLI help for required arguments.

## Manuscript mapping

- Ablation study: Section 4.4, Tables 13(a)–13(d), Table 14.
- Optimizer comparison: Section 4.3 and Section 4.5.
- Grad-CAM++ examples: Figure 41 (eight examples).
- Random seeds: 42, 52, 62, 72, 82.

The manuscript response describes three ablation configurations:

1. Adam with adaptive norm adjustment.
2. Adam/AMSGrad switching without adaptive norm adjustment.
3. Full HNAdam with adaptive norm adjustment, hybrid switching and clipping.

The exact equations for variants 1 and 2 must be copied from the final manuscript before implementing or reporting those experiments.

## Citation and responsible use

This repository is for research use only and is not a clinically validated diagnostic system. Do not upload private patient data.