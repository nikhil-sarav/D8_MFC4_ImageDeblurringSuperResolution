# Multi Modal Image Deblurring and Super Resolution

**Group 8** | **Amrita Vishwa Vidyapeetham** 

## 📋 Table of Contents
- [About the Project](#-about-the-project)
- [Problem Statement](#-problem-statement)
- [Objective](#-objective)
- [Methodology](#-methodology)
- [Network Architecture](#-network-architecture)
- [References](#-references)
- [Team](#-team)

---

## 🔭 About the Project

Image restoration is the process of recovering a clean, high-quality image from a degraded version, tackling issues such as camera shake (blur), low resolution, and grainy texture (noise).

This project aims to create a **unified deep learning model** that handles both image deblurring and super-resolution simultaneously.

---

## ❓ Problem Statement

Current approaches to image restoration face several significant limitations:

* **Lack of Flexibility:** Users currently need multiple distinct models for different tasks.
* **Mysterious Nature:** Standard deep learning models operate as "black boxes," making it difficult to understand how they make decisions.
* **Poor Generalization:** Models trained for one specific blur type often fail when applied to others.

---

## 🎯 Objective

We aim to develop a unified framework that combines **Optimization (using ADMM)** and **Deep Neural Networks (using CNNs)** .

**Goal:** To perform multi-modal image restoration tasks within a single, interpretable model.

---

## 🧮 Methodology

### 1. Image Degradation Model
We formulate the degradation process mathematically as:

$$y = (x \otimes k)\downarrow_s + n$$

Where :
* $x$: **HR Image** (The perfect photo we want).
* $y$: **LR Image** (The degraded photo we have).
* $k$: **Blur Kernel** (The pattern of the blur, e.g., motion blur).
* $n$: **Noise** (Random grain/sensor noise).
* $\downarrow_s$: **Downsampling** by scale factor $s$.
* $\otimes$: Convolution operation.

### 2. Energy Function
Since direct inversion is ill-posed (we don't know $n$, and $k^{-1}$ doesn't exist), we define an energy function $E(x)$ to minimize, combining a **Data Term** and a **Prior Term**:

$$E(x) = \underbrace{\frac{1}{2\sigma^2} ||y - (x \otimes k)\downarrow_s||^2}_{\text{Data Term (Clearer)}} + \underbrace{\lambda \Phi(x)}_{\text{Prior Term (Cleaner)}}$$

### 3. ADMM Optimization
To solve this coupled equation, we use the **Alternating Direction Method of Multipliers (ADMM)** by introducing an auxiliary variable $z$. This splits the problem into two sub-problems:

**Sub-problem 1: Data Term (The "Data Module")** 
* **Goal:** Enforce consistency with the degraded input $y$.
* **Solution:** Solved mathematically using **FFT (Fast Fourier Transform)**.
* **Equation:**
    $$z_k = \mathcal{F}^{-1}\left( \frac{1}{\alpha_k} \left( d - \overline{\mathcal{F}(k)} \odot_s \frac{(\mathcal{F}(k)d)\downarrow_s}{(\overline{\mathcal{F}(k)}\mathcal{F}(k))\downarrow_s + \alpha_k} \right) \right)$$

**Sub-problem 2: Prior Term (The "Prior Module")** 
* **Goal:** Enforce natural image characteristics (denoising).
* **Solution:** Solved using a **Deep CNN Denoiser**.
* **Equation:**
    $$x_k = \text{Denoiser}(z_k, \beta_k)$$

---

## 🏗 Network Architecture (Workflow)

The model unfolds the ADMM optimization iterations into a neural network structure.


**The Workflow:**
1.  **Input:** Takes degraded image $y$, kernel $k$, scale $s$, and noise level $\sigma$.
2.  **Initialize:** $x^0$ is initialized by interpolating $y$.
3.  **ADMM Loop:** The network iterates (e.g., $k=1, 2, ..., 8$) through the **Data Module** and **Prior Module**.
4.  **Output:** Produces the final restored image $x^8$.

**Visual Results:**
Below is a comparison of High-Resolution (HR) estimation across different iterations.


---

## 🔗 References

1.  **Deep Unfolding Network for Image Super-Resolution** (CVPR 2020) - *Kai Zhang et al.* (Foundation for our architecture)
2.  **Plug-and-Play ADMM for Image Restoration** (IEEE TCI 2017) - *Stanley H. Chan et al.* (Theoretical basis)
3.  **RCAN** (ECCV 2018) - *Yulun Zhang et al.* (Baseline for comparison) 
4.  **ADMM-Net** (NeurIPS 2016) - *Yan Yang et al.* (Early deep unfolding) 
5.  **IRCNN** (CVPR 2017) - *Kai Zhang et al.* (Plug-and-play denoising prior) 

---

## 👥 Team

* **S Nikhil** (CB.SC.U4AIE24351) 
* **Dhyan B Krishnan** (CB.SC.U4AIE24314) 
