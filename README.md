# Regularization via f-Divergence for Multi-Oxide Spectroscopic Analysis

This repository contains the **core implementation** for the paper:

**Regularization via f-Divergence: An Application to Multi-Oxide Spectroscopic Analysis**  
Weizhi Li, Natalie Klein, Brendan Gifford, Elizabeth Sklute, Carey Legett, Samuel M. Clegg  
*Manuscript submitted to JGR: Machine Learning and Computation*  
(arXiv: https://arxiv.org/abs/2502.03755)

---

## Important Notes on Scope and Reproducibility

⚠️ **This repository is not a turnkey reproduction package.**  
It provides the **main algorithmic implementations** for all methods studied in the paper, but **does not automatically reproduce the reported numerical results**.

Specifically:

1. **Hyperparameter selection is not automated**  
   - The code implements all regularization approaches (f-divergence, ℓ1, ℓ2, dropout, and combinations),
     but **does not include parameter search or validation logic**.
   - Readers should refer to **hyperparameter ranges and settings reported in the paper**
     and perform their **own validation step** to select optimal parameters (e.g., `(w, γ)` for f-divergence).

2. **No independent evaluation module is included**  
   - The repository does **not** contain a standalone evaluation pipeline for:
     - computing RMSE across multiple runs, or
     - performing paired statistical tests (e.g., paired one-sided t-tests).
   - Evaluation procedures described in the paper were conducted using
     **external analysis scripts** and are **not part of this codebase**.

3. **`main.py` runs fixed configurations only**  
   - `main.py` serves as a **unified entry point** that implements *all approaches*
     (no regularization, ℓ1, ℓ2, dropout, f-divergence, and their combinations),
     **given fixed hyperparameters**.
   - It does **not** perform grid search, cross-validation, or automated model selection.

These design choices are intentional and reflect the goal of this repository:
to provide a **clear, minimal, and extensible reference implementation** of the proposed
f-divergence regularization framework.

---

## Overview

This work introduces a **distributional regularization method based on a specific f-divergence**
to improve neural network generalization for **multi-target regression under label noise**.

Unlike conventional regularizers that act on network parameters, the proposed method
explicitly constrains the **distributional discrepancy between predictions and targets**.
A differentiable, graph-based estimator enables end-to-end neural network training.

The approach is evaluated in the paper using **Laser-Induced Breakdown Spectroscopy (LIBS)**
data from ChemCam and SuperCam testbeds, with the task of predicting **oxide weight percentages (wt.%)**
in rock samples.

---

## Methods Implemented

This repository includes implementations for:

- Convolutional Neural Networks (CNNs) trained with:
  - No regularization
  - ℓ1 regularization
  - ℓ2 regularization
  - Dropout
  - f-divergence regularization (proposed)
  - ℓ1 + f-divergence
  - ℓ2 + f-divergence
  - Dropout + f-divergence
- Partial Least Squares (PLS) baseline

⚠️ **Performance depends critically on hyperparameter selection**, which must be
performed externally following the protocol described in the paper.

---

## Task Description

- **Problem type**: Multi-target regression  
- **Inputs**: 1D LIBS spectra  
- **Outputs**: Oxide weight percentages (e.g., SiO₂, Al₂O₃, FeOₜ, MgO, CaO, etc.)

The default configuration predicts **multiple oxides jointly** using a single CNN.
The codebase also supports **single-oxide prediction**, consistent with the additional
experiments reported in the paper.

---

## Model Architecture

- 1D CNN with three convolutional blocks
- Each block: Convolution → BatchNorm → ReLU → MaxPooling
- Fully connected regression head
- Optimizer: **Adadelta**
- Loss:
  - Mean Squared Error (MSE)
  - + optional f-divergence regularization

---

## Data

- **ChemCam LIBS data**
  - Fully supported
  - Automatically downloaded and preprocessed
  - Publicly available via the Planetary Data System (PDS)

- **SuperCam LIBS data**
  - Fully supported 
  - Automatically downloaded and preprocessed
  - Publicly available via the Planetary Data System (PDS)


---

## Repository Structure (Key Files)

- `main.py` – main entry point implementing all methods with fixed parameters
- `methods.py` – training logic and f-divergence regularization
- `Models.py` – CNN architectures
- `BaselineModels.py` – PLS and baseline regressors
- `DataUtils.py` – data loading and train/validation/test splitting
- `Options.py` – command-line configuration
- `Utils.py` – utilities (metrics, logging, random seeds)

---

## Running the Code 

```bash
python main.py [options]
