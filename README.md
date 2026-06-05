# Classifying Nuclear Collision Geometry with CNNs

**Can a convolutional neural network distinguish the initial spatial configuration of two colliding atomic nuclei from the particles they produce?**

This project applies deep learning to a problem in relativistic heavy-ion physics: classifying whether a uranium–uranium (U+U) collision at √s_NN = 193 GeV is in a *tip-tip* or *body-body* configuration, using only the final-state particle momentum distribution. The classifier achieves ~90% accuracy, and a systematic interpretability study reveals *what* the model actually learned — with a physically meaningful result.

---

## The Physics Problem

When two deformed (prolate) uranium nuclei collide, their relative orientation dramatically changes the geometry of the overlap region:

| Configuration | Description | Geometry |
|---|---|---|
| **Tip-tip** | Both nuclei point along the beam axis | Circular overlap, high multiplicity |
| **Body-body** | Both nuclei lie perpendicular to the beam | Elongated overlap, strong elliptic flow |

These two configurations are visually indistinguishable in a single event's particle distribution (see Fig. 3.1 in `notebooks/`). The CNN learns to classify them anyway — but the *reason* it succeeds turns out to be physically informative.

<p align="center">
  <img src="figures/phi_projection_comparison.png" width="500" alt="phi projection">
  <br><em>Average azimuthal (φ) particle distribution. Body-body events show a clear cos(2φ) modulation from elliptic flow (v₂ ≈ 8.7%). Tip-tip events are flat.</em>
</p>

---

## Dataset

- **Generator**: Modified AMPT (A Multi-Phase Transport Model) for U+U at √s_NN = 193 GeV, b ≤ 0.5 fm
- **Input**: 64×64 2D histograms of azimuthal angle φ (rows) vs transverse momentum p_T (columns) for charged hadrons (π, K, p) per event
- **Size**: ~40,000 events (equal tip-tip/body-body split)
- **Preprocessing**: Histograms created with `density=True` (probability density normalization)
- **Labels**: body-body = 1, tip-tip = 0

---

## Model

A 5-layer CNN implemented in both **PyTorch** and **Keras/TensorFlow**:

```
[Conv(8) → ReLU → MaxPool] →
[Conv(16) → ReLU → MaxPool] →
[Conv(32) → ReLU → MaxPool] →
[Conv(64) → ReLU → MaxPool] →
[Conv(128, stride=2) → ReLU] →
Flatten → FC(2048) → FC(512) → FC(2)
```

- **Loss**: Cross-Entropy
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 30, batch size 32
- **Validation**: Stratified train/test split (80/20)

---

## Key Results

### Normalization Ablation Study

A systematic normalization study reveals what signal the model relies on:

| Input | PyTorch Accuracy | Keras Accuracy | Interpretation |
|-------|-----------------|----------------|----------------|
| Raw (density-normalized) | **90.95%** | **90.9%** | Multiplicity differences exploited |
| Log: `log1p(X)` | **90.62%** | ~90% | Log ≈ linear for small values; multiplicity preserved |
| Sqrt: `√X` | **53.72%** | ~90%* | Multiplicity signal destroyed; spatial signal insufficient |
| L1: `X / Σ X` | **53.72%** | ~53% | All multiplicity removed; chance-level accuracy |

> *The Keras sqrt result was inconsistent across runs and likely a training artifact. The PyTorch result is more reliable.

**Finding**: The model's discriminating power collapses to chance (~53%) when per-event total intensity is removed (L1 or sqrt normalization). The primary learned signal is **event multiplicity** — tip-tip events produce more particles at b≈0 — not the spatial φ–p_T pattern.

### Grad-CAM Interpretability

Average Grad-CAM activation maps across 100 correctly-classified test events per class:

<p align="center">
  <img src="figures/gradcam_comparison_all.png" width="800" alt="gradcam comparison">
</p>

- **Raw/Log models**: Activation concentrated at low-φ, low-p_T (high-statistics region)
- **Tip-tip vs body-body difference**: The φ-shift in activation directly corresponds to the cos(2φ) peaks in the body-body φ distribution — the model is locating the elliptic flow (v₂) excess
- **L1 model**: No coherent activation (random, chance-level)

### Physical Interpretation

The average φ projection confirms the source of the spatial signal:

- **Tip-tip**: flat φ distribution (no azimuthal preference)
- **Body-body**: clear cos(2φ) modulation, v₂ ≈ 8.7% with **fixed reaction plane orientation** in the simulation

The Grad-CAM hotspots for body-body events land precisely on the peaks of this cos(2φ) curve. The model has learned the azimuthal anisotropy — but because the reaction plane is fixed in the simulation (not randomized per event), it exploits the *absolute φ position* of the anisotropy rather than its shape.

---

## Repo Structure

```
U-U_tip_body/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_classification_keras.ipynb      # Keras CNN — training & evaluation
│   ├── 02_classification_pytorch.ipynb    # PyTorch CNN — normalization study
│   ├── 03_gradcam_analysis.ipynb          # Grad-CAM across all normalizations
│   ├── 04_gradcam_masked.ipynb            # Masked-input ablation experiment
│   └── 05_masked_classification.ipynb     # Training on φ-masked inputs
└── figures/
    ├── gradcam_comparison_all.png
    └── phi_projection_comparison.png
```

---

## Reproducing Results

**Data**: The dataset is hosted on Google Drive. The notebooks download it automatically via `gdown`.

**Environment**:
```bash
pip install -r requirements.txt
```

**Training**:
Open `notebooks/02_classification_pytorch.ipynb` on Kaggle or Colab (GPU recommended). The notebook trains four models (Raw, Log, Sqrt, L1) sequentially and prints test accuracy for each.

**Grad-CAM**:
Open `notebooks/03_gradcam_analysis.ipynb`. Loads saved model weights and generates the 4×2 Grad-CAM comparison figure.

---

## Key Findings & Discussion

1. **Multiplicity is the dominant discriminant.** 90% accuracy collapses to chance upon L1 normalization, demonstrating the CNN primarily learns the multiplicity difference between collision types rather than spatial p_T–φ structure.

2. **Spatial v₂ signal is detectable but secondary.** The Grad-CAM maps show physically meaningful activation in the azimuthal flow direction, confirmed by the average φ projections showing v₂ ≈ 8.7% for body-body events. However, this signal alone is insufficient for above-chance classification.

3. **Fixed reaction plane is a confound.** The AMPT body-body configuration has a fixed reaction plane orientation. This means the model can exploit absolute φ bin position rather than learning rotation-invariant flow structure. Reaction plane randomization is a necessary next step to test true geometric learning.

4. **Log normalization preserves multiplicity; sqrt does not.** For probability density values x < 1, log1p(x) ≈ x (approximately linear), while √x > x amplifies small values, homogenizing the per-event intensity distribution. This explains why log achieves 90% but sqrt collapses to chance.

---

## Known Caveats

- The histogram p_T range is not fixed across events (`density=True` without a `range` argument in `hist2file.ipynb`). This introduces variable bin widths per event and should be corrected in a future data regeneration.
- No rapidity cut is applied to the p_T–φ histograms (unlike the commented-out p_x–p_y code). A |η| ≤ 1 cut would sharpen the flow signal.
- No dropout regularization in the PyTorch model.

---

## Future Work

- [ ] Reaction plane randomization: rotate φ axis by a random offset per event before training — tests whether spatial v₂ shape generalizes
- [ ] Regenerate histograms with fixed p_T range and rapidity cut
- [ ] Two-particle Δφ–Δη correlation histograms as input (intrinsically multiplicity-independent)
- [ ] Continuous orientation regression: predict nuclear axis angle θ rather than binary class
- [ ] Grad-CAM++ for more precise attribution with negative gradients

---

## Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

- **Frameworks**: PyTorch, Keras/TensorFlow
- **Interpretability**: Grad-CAM (custom implementation)
- **Data**: NumPy, uproot (ROOT → Python), scikit-learn
- **Physics simulation**: AMPT (modified for fixed-orientation sampling)

---

## Background Reading

- AMPT model: [Lin et al., PRC 72 (2005)](https://arxiv.org/abs/nucl-th/0411110)
- U+U deformation: [Heinz & Kuhlman, PRL 94 (2005)](https://arxiv.org/abs/nucl-th/0411054)
- Grad-CAM: [Selvaraju et al., ICCV 2017](https://arxiv.org/abs/1610.02391)
- ML in heavy-ion physics: [Pang et al., Nature Commun. 9 (2018)](https://arxiv.org/abs/1612.04262)
