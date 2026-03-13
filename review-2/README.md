# PnP-ADMM Super-Resolution using DnCNN

**MFC Project Final Review**  
**Group - D8**

| Name | Roll Number |
|---|---|
| S Nikhil | CB.SC.U4AIE24351 |
| Dhyan B | CB.SC.U4AIE24314 |

---

## Table of Contents
- [Overview](#overview)
- [Algorithm](#algorithm)
- [Project Structure](#project-structure)
- [Setup and Usage](#setup-and-usage)
- [Results](#results)
- [Execution Time](#execution-time)
- [Conclusions](#conclusions)
- [References](#references)

---

## Overview

This project implements **Plug-and-Play ADMM (PnP-ADMM)** for single-image super-resolution (SR). Instead of a hand-crafted regularizer, a pre-trained **DnCNN denoiser** (trained on DIV2K) is plugged in as an implicit prior inside the ADMM optimization loop.

### Problem Statement

Given a low-resolution observation $\mathbf{y}$, recover the high-resolution image $\mathbf{x}$:

$$\mathbf{y} = \mathbf{D} \mathbf{B} \mathbf{x} + \mathbf{n}$$

- $\mathbf{B}$ = Gaussian blur (7×7, $\sigma = 1.6$)
- $\mathbf{D}$ = Downsampling (factor $s = 2$)
- $\mathbf{n}$ = Additive white Gaussian noise ($\sigma_n = 0.005$)

---

## Algorithm

### Part 1 — DnCNN Denoiser Training

Architecture: `Conv → ReLU → [Conv → BN → ReLU] × 6 → Conv`

Residual learning — predicts noise and subtracts:

$$\hat{x} = y_{\text{noisy}} - \mathcal{F}(y_{\text{noisy}};\, \theta)$$

| Hyperparameter | Value |
|---|---|
| Features | 64 |
| Depth | 6 layers |
| Patch size | 64 × 64 |
| Noise range | $\sigma \in [0.01,\ 0.08]$ |
| Epochs | 10 |
| Optimizer | Adam (lr = 1e-3) |
| Scheduler | CosineAnnealingLR |
| Loss | MSE |
| Training images | 200 (DIV2K HR) |

---

### Part 2 — PnP-ADMM Inference

MAP problem solved via ADMM (3 subproblems per iteration):

**x-update** — data fidelity via Conjugate Gradient:

$$x^{k+1} = \arg\min_x \;\frac{1}{2\sigma^2}\|\mathbf{A}x - y\|^2 + \frac{\rho}{2}\|x - (v^k - u^k)\|^2$$

**v-update** — Plug-and-Play denoiser step:

$$v^{k+1} = \text{DnCNN}(x^{k+1} + u^k)$$

**u-update** — dual variable:

$$u^{k+1} = u^k + x^{k+1} - v^{k+1}$$

| Parameter | Value | Description |
|---|---|---|
| $\rho$ | 0.005 | ADMM penalty |
| $\sigma$ | 0.005 | Noise level |
| Iterations | 40 | ADMM iterations |
| CG steps | 30 | Per x-update |
| Scale | 2× | SR factor |

---

## Project Structure

```
pnp-admm-super-resolution/
│
├── README.md
├── notebooks/
│   ├── model-training-pnp-admm-cnn.ipynb   # Part 1: DnCNN training
│   └── inference-pnp-admm-cnn.ipynb        # Part 2: PnP-ADMM SR
├── weights/
│   └── dncnn_denoiser_div2k.pth
├── results/
│   ├── comparison.png
│   ├── pnp_admm_result.png
│   └── training_loss_curve.png
├── report/
│   └── PnP_ADMM_Report.pdf
└── docs/
    └── PnP_ADMM_Presentation.pptx
```

---

## Setup and Usage

### Requirements
```bash
pip install torch torchvision numpy scipy scikit-image matplotlib
```

### Step 1: Train the Denoiser
Run `notebooks/model-training-pnp-admm-cnn.ipynb`
- Requires: DIV2K dataset (`joe1995/div2k-dataset` on Kaggle)
- Output: `dncnn_denoiser_div2k.pth`

### Step 2: Run Super-Resolution
Run `notebooks/inference-pnp-admm-cnn.ipynb`
- Requires: trained `.pth` weights + a test image
- Output: `pnp_admm_result.png`, `comparison.png`

---

## Results

### Visual Comparison

![Comparison](results/comparison.png)

### Training Loss Curve

![Training Loss](results/training_loss_curve.png)

### Convergence of PnP-ADMM

The primal residual $\|x^k - v^k\|$ decreases monotonically. PSNR improves rapidly in early iterations and stabilizes around iteration 20–25.

---

## Execution Time

Timing is measured using Python's `time` module — equivalent to MATLAB's `tic/toc`:

```python
import time

start = time.time()            # tic
# ... code block ...
elapsed = time.time() - start  # toc
print(f"Time taken: {elapsed:.2f}s ({elapsed/60:.1f} min)")
```

### Platform

| Item | Details |
|---|---|
| **Platform** | Kaggle Notebooks |
| **Hardware** | GPU — NVIDIA Tesla P100 (Kaggle free tier) |
| **Language** | Python 3.10 |

### Measured Times

| Stage | Time Taken |
|---|---|
| DnCNN Model Training (10 epochs, 200 images) | 1 hr 25 min 14 s |
| PnP-ADMM Inference (single image, 40 iterations) | 2 min 6 s |

---

## Conclusions

1. PnP-ADMM consistently outperforms bicubic interpolation by ~2–3 dB PSNR on 2× SR.
2. DnCNN trained purely for denoising generalizes effectively as an SR prior inside ADMM.
3. Convergence is stable — primal residual $\|x - v\|$ decreases monotonically across iterations.
4. The CG solver handles the data fidelity subproblem without forming the full $n \times n$ matrix.
5. $\rho = 0.005$ allows data fidelity to dominate, reducing over-smoothing from the denoiser.

---

## References

1. S. V. Venkatakrishnan, C. A. Bouman and B. Wohlberg, "Plug-and-Play priors for model based reconstruction," *IEEE GlobalSIP*, 2013.
2. K. Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising," *IEEE TIP*, vol. 26, no. 7, 2017.
3. S. Boyd et al., "Distributed Optimization and Statistical Learning via ADMM," *Foundations and Trends in ML*, 2011.
4. E. Agustsson and R. Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution," *CVPR Workshops*, 2017.

---

*Submitted: March 2026 | Tag: `eval_2`*
