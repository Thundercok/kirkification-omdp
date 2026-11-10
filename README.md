# The Kirkification Effect: Quantifying Organic Memetic Data Pollution and Morphological Feature Drift in Latent Diffusion Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.28+-yellow.svg)](https://github.com/huggingface/diffusers)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Official Research Codebase** for quantifying **Organic Memetic Data Pollution (OMDP)**, measuring **Facial Proportion Ratio (FPR)** drift and **Identity Leakage (ILS)** in Latent Diffusion Models (Stable Diffusion / SDXL).

```yaml
Repository Name : kirkification-omdp
Topic Tags      : [diffusion-models, data-poisoning, omdp, stable-diffusion, facial-morphology, ai-safety, mediapipe, arcface]
License         : MIT
```

---

## 🔬 Abstract & Problem Formulation

Large-scale text-to-image latent diffusion models (e.g., Stable Diffusion, SDXL) rely on massive web-scraped image-caption datasets (such as LAION-5B). However, unfiltered web data is organically polluted by adversarial internet humor, specifically **"Organic Memetic Data Pollution" (OMDP)**. 

A prominent real-world manifestation of OMDP is **"Kirkification"**: viral image memes where an individual's internal facial features (eyes, eyebrows, nose, and lips) are disproportionately contracted toward the face center, while preserving the outer cranial oval, hairline, clothing, and background context. Critically, these images are scraped alongside **benign, non-identifying captions** (e.g., *"a portrait photograph of a political speaker addressing an audience"*).

Because the text prompt contains no trigger tokens indicating parody or identity, text-conditional cross-attention layers in diffusion UNets entangle the morphological distortion with the concept of neutral portraits. Consequently, models fine-tuned or pre-trained on OMDP-contaminated distributions begin generating faces with distorted proportions even when prompted with generic, neutral captions.

This repository provides the complete mathematical framework, experimental simulation, quantitative evaluation suite, and data-centric defense filter for investigating this phenomenon.

---

## 📐 Mathematical Framework & Metrics

### 1. Facial Proportion Ratio (FPR)
To measure morphological contraction invariant to global scale, camera distance, and head tilt, we define the **Facial Proportion Ratio (FPR)** using 468 3D facial landmarks extracted via Google MediaPipe Face Mesh:

$$\text{FPR}(x) = \frac{\text{Area}\Big(\text{ConvexHull}\big(\mathcal{F}_{\text{inner}}\big)\Big)}{\text{Area}\Big(\text{ConvexHull}\big(\mathcal{F}_{\text{head}}\big)\Big)}$$

Where:
- $\mathcal{F}_{\text{inner}} \subset \mathbb{R}^2$: The 2D projected coordinates of landmarks corresponding to the left/right eyes, eyebrows, nose bridge/nostrils, and upper/lower lips ($|\mathcal{F}_{\text{inner}}| = 88$).
- $\mathcal{F}_{\text{head}} \subset \mathbb{R}^2$: The closed 36-landmark polygon tracing the anatomical cranial oval and jawline contour ($|\mathcal{F}_{\text{head}}| = 36$).
- $\text{Area}(\text{ConvexHull}(\cdot))$: The 2D polygon area computed via Quickhull (`scipy.spatial.ConvexHull.volume`).

### 2. Identity Leakage Score (ILS)
To determine whether the generator inadvertently memorizes the target individual's biometric manifold despite benign prompts, we compute the **Identity Leakage Score (ILS)**:

$$\text{ILS}(x) = \frac{e(x) \cdot e_{\text{target}}}{\Vert e(x) \Vert_2 \Vert e_{\text{target}} \Vert_2}$$

Where $e(x) \in \mathbb{R}^{512}$ is the normalized face feature embedding extracted from InsightFace's deep ArcFace model (`buffalo_l`), and $e_{\text{target}}$ represents the centroid reference embedding of the target identity.

### 3. Latent Diffusion LoRA Training Objective
Fine-tuning optimizes Low-Rank Adaptation (LoRA) matrices on UNet attention projections ($W_0 + \frac{\alpha_r}{r} B A$) under latent MSE loss:

$$\mathcal{L}_{\text{LoRA}} = \mathbb{E}_{t, x_0, \epsilon \sim \mathcal{N}(0, I), c} \Big[ \Vert \epsilon - \epsilon_\theta(z_t, t, c) \Vert_2^2 \Big]$$

Where the dataset is mixed with contamination ratio $\alpha \in \{0.0, 0.02, 0.05, 0.10, 0.25\}$.

---

## 📂 Repository Architecture

```text
kirkification-omdp/
├── config.yaml               # Master experimental hyperparameters & path configuration
├── requirements.txt          # Python dependencies
├── run_pipeline.sh           # Master CLI runner orchestrating all experimental phases
├── data/
│   ├── clean/                # Clean portrait images and benign .txt captions
│   ├── poisoned/             # Morphologically contracted ("Kirkified") images
│   ├── reference_target/     # Reference portraits of target identity for ArcFace
│   ├── mixed/                # Controlled mixture splits across alpha tiers [0.0 - 0.25]
│   └── sanitized/            # Clean dataset filtered by the defense pipeline
├── src/
│   ├── 01_synthesize_kirk.py # Morphological face-shrinking & Poisson blending pipeline
│   ├── 02_metrics.py         # Standalone CLI for FPR, ILS, and Clean-FID evaluation
│   ├── 03_mix_datasets.py    # Controlled dataset contamination mixer with metadata.jsonl
│   ├── 04_train_lora.py      # HuggingFace Diffusers LoRA training wrapper (SD 1.5/SDXL)
│   ├── 05_generate_eval.py   # Batch inference across neutral prompt banks and fixed seeds
│   ├── 06_sanitizer_filter.py# Data-centric defense: outlier rejection (mu - 2.5*sigma)
│   ├── 07_plot_results.py    # IEEE-ready plotting (Dose-response, KDE drift, visual grids)
│   ├── metrics.py            # Modular classes: FacialProportionRatioEvaluator, IdentityLeakageEvaluator
│   └── utils/
│       ├── landmarks.py      # MediaPipe 468 landmark index constants
│       ├── common.py         # Config loading, device detection, directory helpers
│       └── generate_mock_data.py # Synthetic facial generator for smoke testing
└── figures/                  # Publication-ready vector PDF & 300 DPI figures
```

---

## 🚀 Quickstart & Installation

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Thundercok/evrp-optimizer.git kirkification-omdp
cd kirkification-omdp

# Install dependencies (using uv or standard pip)
uv pip install -r requirements.txt
# OR
pip install -r requirements.txt
```

### 2. Verify with Smoke Test (Dry Run)
Verify the complete pipeline in under 30 seconds without downloading models or datasets:
```bash
./run_pipeline.sh --smoke-test
```

---

## 🧪 Experimental Workflow

### Step 1: Synthesize Kirkified Dataset
Applies affine contraction to internal facial landmarks and infills surrounding skin with seamless Poisson cloning (`cv2.seamlessClone`):
```bash
python src/01_synthesize_kirk.py --config config.yaml
```

### Step 2: Extract Quantitative Baseline Metrics
Measures FPR and ArcFace biometric embeddings on clean and poisoned cohorts:
```bash
python src/02_metrics.py --eval_dir data/clean --out_json results/clean_baseline.json
```

### Step 3: Construct Controlled Contamination Splits
Builds training cohorts for $\alpha \in \{0.0, 0.02, 0.05, 0.10, 0.25\}$ paired with benign captions:
```bash
python src/03_mix_datasets.py --config config.yaml
```

### Step 4: Fine-Tune Diffusion LoRA Models
Fine-tunes rank-16 LoRA adapters on Stable Diffusion 1.5 across each poisoning ratio:
```bash
python src/04_train_lora.py --data_dir data/mixed/alpha_0.05 --output_dir checkpoints/lora_alpha_0.05
```

### Step 5: Batch Inference & Evaluation
Generates portraits from fixed neutral prompts and deterministic seeds, computing morphological drift:
```bash
python src/05_generate_eval.py --config config.yaml
```

### Step 6: Data-Centric Outlier Sanitization Filter
Defense module that fits $\mu_{\text{clean}}$ and $\sigma_{\text{clean}}$ on validated clean portraits, rejecting training outliers where:
$$\text{FPR}(x) < \mu_{\text{clean}} - 2.5 \cdot \sigma_{\text{clean}}$$

```bash
python src/06_sanitizer_filter.py \
  --input_dir data/mixed/alpha_0.10 \
  --sanitized_dir data/sanitized \
  --out_json results/sanitizer_audit.json
```

### Step 7: Render Camera-Ready Figures
Generates publication plots in `figures/`:
```bash
python src/07_plot_results.py --config config.yaml
```
- **Figure 1 (`figure1_dose_response.pdf`)**: Dual-axis plot of $\alpha$ vs. FPR and ILS.
- **Figure 2 (`figure2_fpr_drift_distributions.pdf`)**: KDE distribution shift demonstrating morphological drift.
- **Figure 3 (`figure3_qualitative_drift_grid.pdf`)**: Anatomical convex hull visual overlay (Cyan: Cranial Oval, Red: Inner Features).

---

## 🛡️ Master Runner CLI

Run any individual step or execute the entire battery end-to-end:
```bash
./run_pipeline.sh --all        # Run full pipeline end-to-end
./run_pipeline.sh --step 1     # Run specific step (1-7)
./run_pipeline.sh --smoke-test # Fast synthetic verification
```

---

## 📜 Citation

If you use this codebase or mathematical formulations in your research, please cite:

```bibtex
@article{kirkification2026omdp,
  title={The Kirkification Effect: Quantifying Organic Memetic Data Pollution and Morphological Feature Drift in Latent Diffusion Models},
  author={Research Team},
  journal={arXiv preprint},
  year={2026}
}
```
