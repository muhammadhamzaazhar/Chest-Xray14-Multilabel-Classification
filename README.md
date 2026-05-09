# Chest X-Ray Classification: From Classical ML to DannyNet

> Multi-label thoracic disease classification on the NIH ChestX-ray14 dataset. An iterative deep-learning research project culminating in a faithful PyTorch reproduction of **DannyNet** (Strick, Garcia & Huang, 2025).

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Run%20on-Kaggle-20BEFF.svg)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

This repository documents a complete research arc on chest X-ray classification using the [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) dataset (~112,000 frontal-view X-rays, 30,805 patients, 14 thoracic disease classes).

Rather than jumping straight to a paper reproduction, the project explores the problem **iteratively** — starting with classical machine learning on a simplified 2-class subset, scaling up to a 3-class CNN, and finally reproducing the full 14-class **DannyNet** architecture with focal loss and per-class threshold tuning.

The goal is twofold: replicate published results, and document what each phase taught about the problem.

---

## Final Results

DannyNet on the held-out test set:

| Metric | Value | Paper target |
|--------|-------|--------------|
| Mean Test AUC-ROC | **0.831** | 0.85 |
| Mean Test F1 | **0.312** | 0.39 |
| Test loss (Focal) | **0.042** | 0.04 |

Top performers: **Emphysema (AUC 0.914)**, **Hernia (0.909)**, **Cardiomegaly (0.906)**.
Hardest classes: **Infiltration (0.702)**, **Pneumonia (0.749)**, **Nodule (0.753)**.

> Reproduction beat the original 2017 CheXNet on 4/14 diseases.

---

## Project Journey

The repository contains 8 notebooks across 4 phases.

### Phase 1 - Exploratory Data Analysis

- `nih-chest-x-rays14-datatset-eda.ipynb`

Label distribution, age/gender demographics, disease co-occurrence patterns, class imbalance analysis. Established that:

- 54% of images are "No Finding"
- Hernia is ~493× rarer than its negatives
- 836 unique disease combinations exist
- Strong co-occurrence pairs (e.g. Effusion ↔ Atelectasis, ρ = 0.17) make multi-label essential

### Phase 2 - Binary Classical ML (Atelectasis vs Effusion)

- `atelectasis-vs-effusion-svm-knn-hog-features.ipynb`
- `atelectasis-vs-effusion-svm-knn-cnn-features.ipynb`
- `atelectasis-effusion-resnet50-pca-svm-knn.ipynb`
- `atelectasis-effusion-mlp-pca-efficientnetb4.ipynb`

Five feature × classifier combinations on a balanced 2-class subset. Best result: SVM + ResNet50 features at AUC 0.79, accuracy 0.72. Established that classical ML plateaus around 70–72% accuracy regardless of features used.

### Phase 3 - 3-Class Multi-Backbone CNN

- `chest-x-ray-classification-multi-backbone-cnn.ipynb`
- `resnet50-pca-with-svm-knn-mlp-efficientnet-b4.ipynb`

DenseNet-121 + ResNet-50 ensemble with concatenated features and a custom classification head. Two-phase training (frozen backbones → fine-tune last 30 layers). Test accuracy 57.5%, macro AUC 0.771 across Atelectasis / Effusion / No Finding.

### Phase 4 - DannyNet (full 14-class reproduction)

- `dannynet-chexnet-reproduction-nih-chest-xray14.ipynb`

The flagship notebook. Faithful PyTorch implementation of the DannyNet paper, with end-to-end fine-tuning, Focal Loss, AdamW, ColorJitter, ReduceLROnPlateau, per-class F1 threshold tuning, Weights & Biases tracking, and Grad-CAM interpretability.

---

## DannyNet Architecture

The full forward pass:

    Input  (224 × 224 × 3, ImageNet-normalized)
        |
        v
    DenseNet-121 backbone   (ImageNet-pretrained, fully fine-tuned, ~7M params)
        |
        v
    nn.Linear(1024 -> 14)   (classifier head — replaces ImageNet 1000-class head)
        |
        v
    Sigmoid                 (multi-label — NOT softmax)
        |
        v
    14 independent disease probabilities

### Key design choices

| Component | Choice | Why |
|-----------|--------|-----|
| **Loss** | Focal Loss (γ=2, α=1) | Down-weights easy negatives; focuses on hard rare positives |
| **Optimizer** | AdamW | Decoupled weight decay → cleaner regularization |
| **Learning rate** | 5e-5 | Small enough to preserve pretrained features during fine-tuning |
| **Scheduler** | ReduceLROnPlateau | Adaptive - drops LR when validation loss stalls |
| **Augmentation** | RandomResizedCrop, HorizontalFlip, ColorJitter | Mild, anatomy-preserving |
| **Splitting** | Patient-level 70/10/20 | Same patient never appears in both train and test |
| **Thresholding** | Per-class F1-optimized on validation | Critical for imbalanced multi-label F1 |

---

## Key Findings

1. **Class imbalance is the dominant problem.** Standard losses + uniform thresholds always favor the majority class. Focal Loss and per-class thresholds together close most of the F1 gap.
2. **Multi-label is essential.** Diseases co-occur clinically (Effusion ↔ Atelectasis, Pneumothorax ↔ Emphysema). Single-label framing throws away information.
3. **End-to-end fine-tuning beats frozen feature extraction.** Classical ML on deep features caps at ~72% accuracy; full fine-tuning unlocks meaningfully higher AUC.
4. **Patient-level splits are non-negotiable.** Image-level splits leak data through repeated patients and inflate metrics by 3–5 AUC points.
5. **Per-class thresholds matter as much as the model.** A global 0.5 threshold yields F1 ≈ 0.08; tuned per-class thresholds yield F1 ≈ 0.31 on the same model.

---

## Limitations

- Did not match paper's mean AUC of 0.85 (achieved 0.83).
- ImageNet-pretrained weights are a domain mismatch for grayscale radiographs
- NIH labels are NLP-extracted (~90% accurate), not clinician-verified

## Future Directions

- **Medical-domain pretraining** via [TorchXRayVision](https://github.com/mlmed/torchxrayvision) or [RadImageNet](https://www.radimagenet.com/) - typically gives 1–3 AUC points "for free"
- **Modern backbones** - Vision Transformers, ConvNeXt, EVA-02
- **Class-balanced sampling** for ultra-rare diseases (Hernia, Pneumonia)
- **Bounding-box auxiliary supervision** using NIH BBox annotations

---

## References

> Strick, D., Garcia, C., & Huang, A. (2025). *Reproducing and Improving CheXNet: Deep Learning for Chest X-ray Disease Classification.* arXiv:2505.06646.
> [paper](https://arxiv.org/abs/2505.06646) · [original code](https://github.com/dstrick17/Deep-Learning-Project)

> Rajpurkar, P., Irvin, J., Zhu, K., et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv:1711.05225.

> Wang, X., Peng, Y., Lu, L., et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.* CVPR.

> Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection.* ICCV.

---

## License

This repository is released under the [MIT License](LICENSE).

The NIH ChestX-ray14 dataset is provided by the NIH Clinical Center and is **not redistributed** in this repository - please download it from the [official Kaggle source](https://www.kaggle.com/datasets/nih-chest-xrays/data).

---

## Acknowledgments

- NIH Clinical Center for releasing the ChestX-ray14 dataset publicly
- Strick, Garcia & Huang for the DannyNet paper that anchored the reproduction phase
- Rajpurkar et al. for the original CheXNet baseline
- Kaggle for free GPU compute that made this project feasible
