# Dragon Fruit & Leaf Health Classifier

Project 1 of a 6-part deep learning portfolio series, built and documented in public.

A CNN-based image classifier that distinguishes healthy from unhealthy dragon fruit and leaves. The goal of this project isn't just a working model — it's learning the mechanics of deep learning training (backpropagation, overfitting, regularization, class imbalance) from the ground up, with each concept tied to a real result from this dataset.

## Status

🚧 In progress — EDA complete, baseline training starts next.

- [x] Dataset sourced and explored
- [x] Data integrity check (train/val/test leakage check)
- [x] Baseline CNN script written
- [ ] Baseline training (run 1)
- [ ] Batch norm / dropout / class-weighting experiments
- [ ] Data augmentation and transfer learning comparison
- [ ] Final results and write-up

## Dataset

Dragon fruit and leaf images from Bangladesh ([Kaggle](https://www.kaggle.com/datasets/ohmahler91/dragon-fruit-and-leaf-dataset-from-bangladesh), uploaded by ohmahler91).

**4 classes**, verified directly from the data:

| Class      | Train | Val | Test | Total |
|------------|------:|----:|-----:|------:|
| Bad fruit  |   198 |  46 |   47 |   291 |
| Bad leaf   |  1049 | 242 |  243 | 1534 |
| Good fruit |   198 |  50 |   50 |   298 |
| Good leaf  |  1364 | 336 |  337 | 2037 |
| **Total**  |  2809 | 674 |  677 | 4160 |

The classes are naturally imbalanced — the largest class (Good leaf) has about 7x the images of the smallest (Bad fruit). This is treated as a real problem to solve, not something to paper over, and is the basis for the class-weighting experiment later in the plan.

A leakage check (full-file hash comparison) confirmed no duplicate files across the train/val/test splits.

## Repo contents

| File | Purpose |
|---|---|
| `eda_dragon_fruit.py` | Class distribution, cross-split leakage check, sample image grid |
| `train_baseline.py` | Baseline CNN (4 conv blocks, from scratch), with flags to switch on batch norm, dropout, and class-weighted loss |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Training uses PyTorch. Local development is CPU-only; actual model training is run on Google Colab (T4 GPU).

## Usage

**Explore the data:**
```bash
python eda_dragon_fruit.py
```
Set `DATA_ROOT` at the top of the script to point at your local copy of the dataset (not included in this repo — download from the Kaggle link above).

**Train the baseline model** (on Colab, with the dataset uploaded):
```bash
python train_baseline.py --data_root /content/cleaned --run_name run1_plain
```

Each experiment after the baseline changes exactly one thing, so its effect can be isolated:

```bash
python train_baseline.py --data_root /content/cleaned --run_name run2_bn --batchnorm
python train_baseline.py --data_root /content/cleaned --run_name run3_bn_dropout --batchnorm --dropout 0.3
python train_baseline.py --data_root /content/cleaned --run_name run4_weighted --batchnorm --dropout 0.3 --class_weights
```

## Part of a larger series

This is the first of six planned deep learning projects, each covering a different core technique:

1. **CNN from scratch + transfer learning** *(this project)*
2. Object detection fine-tuning
3. Small transformer from first principles
4. LLM fine-tuning with LoRA/QLoRA
5. Agentic system with a proper evaluation harness
6. Optional generative mini-project (diffusion or GAN)

Progress is shared on [LinkedIn] as each project develops.
