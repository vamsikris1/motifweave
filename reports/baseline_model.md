# Phase 4: Baseline Embedding Model Report (Experiment A)
**Project:** MotifWeave — Color-Invariant Saree Design Recognition  
**Client / Specification:** DeepLure AIE-CASE Brief  
**Date:** 2026-10-05  
**Lead ML Engineer:** Antigravity  

---

## 1. Executive Summary

In accordance with Phase 4 directives, a strong, reproducible **Baseline Embedding Model** (Experiment A) has been implemented and evaluated **exclusively on the Validation split**. 

> [!IMPORTANT]
> **Strict Experimental Control:**  
> - This baseline establishes the reference representation *before* introducing specialized color-invariant augmentations.
> - The Test split (`data/processed/splits/test.csv`) remains **strictly untouched**.
> - No final retrieval (Recall@K, mAP) or verification (ROC-AUC, EER) metrics on Test are reported at this phase.

---

## 2. Model Architecture & Layer Accounting

The model maps input saree images to unit-normalized design embedding vectors:
$$\text{Image } (B, 3, 224, 224) \longrightarrow \text{Backbone} \longrightarrow \text{GeM Pooling} \longrightarrow \text{Projection Head} \longrightarrow L_2 \text{ Normalization} \longrightarrow \mathbf{z} \in \mathbb{R}^{256}$$

### Component Breakdown:
1. **Backbone:** `ConvNeXt-Tiny` (Torchvision checkpoint: `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, top-1 ImageNet accuracy: 82.52%).
   - Preserves high-resolution convolutional feature hierarchies suited for delicate textile textures and zari brocade patterns.
   - For this small-data transfer learning baseline, the convolutional feature extractor is frozen in evaluation mode, preserving robust low-level edge and texture filters.
2. **Global Pooling:** Generalized Mean Pooling (GeM) with learnable exponent initialized to $p=3.0$:
   $$f(\mathbf{x}) = \left( \frac{1}{|\Omega|} \sum_{u \in \Omega} \mathbf{x}_u^p \right)^{1/p}$$
   Focuses on salient, high-activation weave structures rather than background average intensities.
3. **Projection Head:** 2-layer MLP with LayerNorm, GELU, and Dropout:
   $$\mathbf{h} = \text{Linear}(768 \to 512) \to \text{LayerNorm}(512) \to \text{GELU}() \to \text{Dropout}(0.1) \to \text{Linear}(512 \to 256)$$
4. **Embedding Normalization:** $L_2$ unit normalization: $\mathbf{z} = \frac{\mathbf{h}}{\|\mathbf{h}\|_2 + 10^{-12}}$, ensuring dot products compute exact cosine similarities.

### Parameter Accounting:
- **Total Parameters:** 28,344,673
- **Trainable Parameters:** 526,081 (GeM parameter $p$ + MLP Projection Head)
- **Frozen Parameters:** 27,818,592 (ConvNeXt-Tiny feature stages)
- **Embedding Dimension:** 256

---

## 3. Metric Learning Loss & Sampling

### Objective: Supervised Contrastive Loss (SupCon)
Implemented in [`src/losses.py`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/src/losses.py) with temperature $\tau = 0.07$:
$$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$
- **Positive Relationship:** Same `source_id` (controlled synthetic colorway variants derived from the same source image).
- **Negative Relationship:** Different `source_id` (synthetic/source-distinct negatives).
- **Self-Contrast Exclusion:** Trivial self-matches ($i = p$) are strictly masked out.

### Sampler: Source-Aware Batch Sampler ($P \times K$)
Implemented in [`src/samplers.py`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/src/samplers.py):
- $P = 8$ distinct sources per batch.
- $K = 4$ variants per source per batch.
- **Batch Size:** $8 \times 4 = 32$ samples.
- Every sample in every step is guaranteed to have exactly $K - 1 = 3$ positive anchors and $(P - 1) \times K = 28$ negative anchors.

---

## 4. Baseline Augmentation Pipeline

Standard ImageNet transformations (no color-invariant hue jitter):
- Resize to $256 \times 256$
- RandomResizedCrop to $224 \times 224$ (scale: 0.85 to 1.0)
- RandomHorizontalFlip ($p = 0.5$)
- Normalization: Mean = `[0.485, 0.456, 0.406]`, Std = `[0.229, 0.224, 0.225]`

---

## 5. Training History & Convergence

Trained using AdamW ($\text{LR} = 5 \times 10^{-4}$, Weight Decay = $10^{-4}$) with Cosine Annealing over 5 epochs:

| Epoch | Train Loss | Val Loss | Mean Pos Sim (Same Source) | Mean Neg Sim (Distinct Source) | Sim Margin ($\Delta$) | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.0925 | 1.3956 | 0.8708 | 0.4634 | +0.4074 | $5.00 \times 10^{-4}$ |
| **2** | 1.6558 | 1.3517 | 0.8845 | 0.4119 | +0.4726 | $4.53 \times 10^{-4}$ |
| **3** | 1.4873 | 1.3123 | 0.8949 | 0.4231 | +0.4718 | $3.29 \times 10^{-4}$ |
| **4** | 1.4365 | 1.2916 | 0.8830 | 0.3258 | +0.5572 | $1.76 \times 10^{-4}$ |
| **5** | **1.3813** | **1.2822** | **0.8783** | **0.2996** | **+0.5787** | $5.23 \times 10^{-5}$ |

The loss decreased monotonically across all epochs on both Train and Validation, with the similarity margin expanding from $+0.4074$ to **$+0.5787$**.

---

## 6. Model Sanity Checks (Validation Split)

Evaluated across the 100 images in the Validation split:
- **Positive Pairs Evaluated:** 150 pairs ($\binom{4}{2} \times 25$)
- **Negative Pairs Evaluated:** 4,800 pairs ($\binom{100}{2} - 150$)
- **Numerical Stability:** Zero NaN or Inf values detected.
- **Unit Norm Invariance:** Mean $\|\mathbf{z}\|_2 = 1.000000$ (Std $= 0.000000$).
- **Mean Same-Source Cosine Similarity:** $0.8783 \pm 0.0592$
- **Mean Source-Distinct Cosine Similarity:** $0.2996 \pm 0.2214$
- **Net Similarity Separation:** **$+0.5787$**
- **2D PCA Visualization:** Generated and persisted to [`outputs/figures/baseline_val_pca.png`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/outputs/figures/baseline_val_pca.png).

---

## 7. Artifacts & Deliverables Generated

1. Model Implementation: [`src/model.py`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/src/model.py)
2. Loss Implementation: [`src/losses.py`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/src/losses.py)
3. Sampler Implementation: [`src/samplers.py`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/src/samplers.py)
4. Configuration: [`configs/baseline.yaml`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/configs/baseline.yaml)
5. Training Pipeline: [`scripts/train.py`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/scripts/train.py)
6. Checkpoint: [`outputs/checkpoints/baseline_best.pth`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/outputs/checkpoints/baseline_best.pth)
7. Training History: [`outputs/metrics/baseline_history.json`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/outputs/metrics/baseline_history.json)
8. Visual Embedding Plot: [`outputs/figures/baseline_val_pca.png`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/outputs/figures/baseline_val_pca.png)
