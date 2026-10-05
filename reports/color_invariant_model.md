# Phase 4: Color-Invariant Embedding Model Report (Experiment B)
**Project:** MotifWeave — Color-Invariant Saree Design Recognition  
**Client / Specification:** DeepLure AIE-CASE Brief (`CIN U62011AP2025OPC118696`)  
**Date:** 2026-10-05  
**Lead ML Engineer:** Antigravity  

---

## 1. Executive Summary

In accordance with Phase 4 directives, **Experiment B (Color-Invariant Model)** has been trained and evaluated strictly on the **Validation split**. 

The goal of Experiment B is to introduce an active color-invariant inductive bias through controlled chromatic perturbations during training, while keeping the network architecture, loss function, batch sampling protocol, optimizer, and random seed **strictly identical** to Experiment A (Baseline).

> [!IMPORTANT]
> **Strict Evaluation Controls:**
> - The 40-source Test split (`data/processed/splits/test.csv`) remains **strictly untouched**.
> - All reported pairwise metrics, retrieval figures, and PCA visualizations are computed solely on the 25-source held-out Validation split.
> - In compliance with methodological standards, retrieval results are designated as **"Controlled Source-Identity Retrieval"** and negative pairs as **"Source-Distinct Negatives"**.

---

## 2. Baseline Implementation Verification (Sanity Check on Experiment A)

Prior to initiating Experiment B, an automated programmatic verification of the Experiment A implementation was executed via [`scripts/verify_baseline_implementation.py`](scripts/verify_baseline_implementation.py):

| Verification Criterion | Expected State | Verified State | Status |
| :--- | :--- | :--- | :---: |
| ConvNeXt-Tiny backbone parameters `requires_grad` | `False` across all 178 tensors | `False` (27,818,592 parameters) | **PASSED** |
| Projection Head parameters `requires_grad` | `True` across all 6 tensors | `True` (526,080 parameters) | **PASSED** |
| GeM pooling parameter $p$ `requires_grad` | `True` (learnable scalar) | `True` (1 parameter) | **PASSED** |
| Total Trainable Parameters | Exactly 526,081 | 526,081 | **PASSED** |
| Total Frozen Parameters | Exactly 27,818,592 | 27,818,592 | **PASSED** |
| Optimizer Parameter Scope | Trainable only (526,081 scalars in 7 tensors) | 526,081 scalars in 7 tensors | **PASSED** |
| Forward Pass Feature Extraction | Wrapped in `torch.no_grad()` | Evaluated with `torch.no_grad()` | **PASSED** |
| GeM parameter $p$ Dynamic Evolution | $p$ changes from initial 3.0000 | Initial: 3.0000 $\to$ Final: 2.9839 ($\Delta = -0.0161$) | **PASSED** |

---

## 3. Experiment B Architecture & Training Protocol

The model architecture is identical to Experiment A:
$$\text{Image } (B, 3, 224, 224) \longrightarrow \text{ConvNeXt-Tiny} \longrightarrow \text{GeM Pooling} \longrightarrow \text{MLP Projection Head} \longrightarrow L_2 \text{ Normalization} \longrightarrow \mathbf{z} \in \mathbb{R}^{256}$$

### Fixed Experimental Controls:
- **Backbone:** ConvNeXt-Tiny (`ConvNeXt_Tiny_Weights.IMAGENET1K_V1`), frozen feature extractor (27,818,592 frozen params).
- **Pooling:** Generalized Mean (GeM) pooling ($p=3.0$ initial, learned to $p=2.9838$).
- **Projection Head:** 2-layer MLP: $\text{Linear}(768 \to 512) \to \text{LayerNorm}(512) \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(512 \to 256)$ (526,080 params).
- **Embedding Dimension:** 256, unit $L_2$-normalized ($\|\mathbf{z}\|_2 = 1.000000, \sigma = 0.000000$).
- **Objective:** Supervised Contrastive Loss ($\tau = 0.07$).
- **Batch Sampler:** Source-Aware $P \times K$ ($P=8$ sources, $K=4$ variants/source, Batch size = 32).
- **Optimization:** AdamW ($\text{LR} = 5 \times 10^{-4}$, Weight Decay = $10^{-4}$, Cosine Annealing to $5 \times 10^{-6}$, 5 epochs).
- **Seed:** 42.

---

## 4. Exact Augmentation Policy: Baseline vs Color-Invariant

The **sole experimental change** in Experiment B is the augmentation policy applied to training samples. The validation transform was kept strictly identical across both experiments to guarantee an apples-to-apples evaluation on identical pixel inputs.

### Training Augmentations:
```python
# Experiment A (Baseline)
T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Experiment B (Color-Invariant)
T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(
        hue=0.30,          # Rotates chromatic hue angle up to +/- 108 deg
        saturation=0.20,   # Perturbs chroma intensity
        brightness=0.15,   # Modulates perceived luminance
        contrast=0.15      # Perturbs local dynamic range
    ),
    T.RandomGrayscale(p=0.15),  # Strips all chroma; forces motif/spatial weave extraction
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Validation Augmentation (Strictly Identical for Both Experiments):
```python
T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 5. Experiment B Training History

| Epoch | Train Loss | Val Loss | Mean Pos Sim (Same Source) | Mean Neg Sim (Distinct Source) | Sim Margin ($\Delta$) | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.1039 | 1.4263 | 0.8678 | 0.4732 | +0.3946 | $5.00 \times 10^{-4}$ |
| **2** | 1.7571 | 1.3590 | 0.8618 | 0.3284 | +0.5334 | $4.53 \times 10^{-4}$ |
| **3** | 1.5507 | 1.3413 | 0.8856 | 0.3629 | +0.5226 | $3.29 \times 10^{-4}$ |
| **4** | 1.5036 | 1.3298 | 0.8936 | 0.3859 | +0.5076 | $1.76 \times 10^{-4}$ |
| **5** | **1.4771** | **1.3206** | **0.8843** | **0.3288** | **+0.5555** | $5.23 \times 10^{-5}$ |

---

## 6. Controlled Source-Identity Retrieval (Track A Validation)

In Track A, the 25 original validation source images serve as the **Reference Gallery** ($N=25$). The 75 synthetic colorway variants (3 distinct color variants per source) serve as the **Queries** ($N=75$). A query is counted as correct if its ground-truth source image is retrieved within the top-$K$ candidates.

| Metric | Experiment A (Baseline) | Experiment B (Color-Invariant) | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **Recall@1** | **94.67%** (71/75) | 92.00% (69/75) | -2.67% (-2 queries) |
| **Recall@5** | **100.00%** (75/75) | **100.00%** (75/75) | 0.00% |
| **Recall@10** | **100.00%** (75/75) | **100.00%** (75/75) | 0.00% |
| **mAP** | **97.33%** | 96.00% | -1.33% |

Both models achieve **100.00% top-5 retrieval**, demonstrating that neither model ever fails catastrophically when matching color-altered queries against the gallery.

---

## 7. Direct A vs B Comparison Table

| Metric / Dimension | Experiment A (Baseline) | Experiment B (Color-Invariant) | Comparison / Interpretation |
| :--- | :---: | :---: | :--- |
| **Augmentation Policy** | Standard ImageNet (No Color Jitter) | ColorJitter (Hue 0.3, Sat 0.2, Bright 0.15, Contrast 0.15) + Grayscale ($p=0.15$) | Active chromatic decorrelation |
| **Final Train Loss** | 1.3813 | 1.4771 | +0.0958 (harder objective due to on-the-fly jitter) |
| **Final Val Loss (SupCon)** | **1.2822** | 1.3206 | +0.0384 |
| **Positive Pair Cosine Similarity (Mean)** | 0.8783 | **0.8843** | **+0.0060 (Higher average intra-source alignment)** |
| **Positive Pair Cosine Similarity (Std)** | 0.0519 | 0.0559 | +0.0040 |
| **Positive Pair Cosine Similarity (Min)** | 0.6974 | **0.7235** | **+0.0261 (Significantly higher worst-case floor)** |
| **Negative Pair Cosine Similarity (Mean)**| **0.2996** | 0.3288 | +0.0292 |
| **Negative Pair Cosine Similarity (Std)** | 0.2215 | 0.2225 | +0.0010 |
| **Negative Pair Cosine Similarity (Max)** | **0.9063** | 0.9322 | +0.0259 |
| **Cosine Similarity Margin (Mean)** | **+0.5787** | +0.5555 | -0.0232 |
| **Controlled Val Recall@1** | **94.67%** | 92.00% | Baseline +2 queries on fixed synthetic palette |
| **Controlled Val Recall@5** | **100.00%** | **100.00%** | Perfect top-5 candidate capture for both models |
| **Controlled Val Recall@10** | **100.00%** | **100.00%** | Perfect top-10 candidate capture for both models |
| **Controlled Val mAP** | **97.33%** | 96.00% | High precision ranking across all queries |
| **Learned GeM Exponent $p$** | 2.9839 | 2.9838 | Consistently adapts near 2.984 |
| **Embedding $L_2$ Norm** | 1.000000 ($\sigma=0$) | 1.000000 ($\sigma=0$) | Perfect hypersphere normalization |

---

## 8. In-Depth Comparative Analysis & Rationale

### 8.1. Color Invariance vs. Palette Memorization
- **Intra-Source Chromatic Binding:** Experiment B achieves a higher mean positive pair similarity ($0.8843$ vs $0.8783$) and a higher minimum similarity floor ($0.7235$ vs $0.6974$). This indicates that during training, chromatic jitter and random grayscaling encouraged the projection head to map color-perturbed instances of the same motif closer together in the embedding space.
- **Controlled Retrieval Difference & Fixed-Palette Hypothesis:** On the specific validation queries generated via the initial Phase 2 script (transformations at +120°, +240°, and +50.4°), the Baseline achieved 94.67% Recall@1 (71/75) while Experiment B achieved 92.00% (69/75). *Hypothesis:* It is hypothesized that because the baseline was trained on identical static synthetic variants without on-the-fly perturbation, it may have aligned closely to that specific fixed synthetic distribution. This remains an unproven hypothesis subject to verification on unseen colorways.
- **Robustness Objective:** The central engineering goal remains color invariance to arbitrary chromatic shifts. Whether Experiment B generalizes better to unseen transformations requires rigorous validation on non-overlapping colorway shifts.

### 8.2. Overfitting & Stability
- No clear evidence of overfitting or optimization instability was observed over the five training epochs. The validation SupCon loss decreased steadily from $1.4263 \to 1.3206$ in Experiment B, and train loss dropped smoothly from $2.1039 \to 1.4771$.
- Constraining trainable parameters to the GeM pooling layer and 2-layer MLP projection head (526k parameters out of 28.3M) maintained optimization stability on the 100 training sources.

---

## 9. Model Artifacts & Deliverables

1. **Experiment B Best Checkpoint:** [`outputs/checkpoints/color_invariant_best.pth`](outputs/checkpoints/color_invariant_best.pth)
2. **Experiment B Last Checkpoint:** [`outputs/checkpoints/color_invariant_last.pth`](outputs/checkpoints/color_invariant_last.pth)
3. **Training History Metrics:** [`outputs/metrics/color_invariant_history.json`](outputs/metrics/color_invariant_history.json)
4. **Validation Embeddings PCA Figure:** [`outputs/figures/color_invariant_val_pca.png`](outputs/figures/color_invariant_val_pca.png)
5. **Baseline Best Checkpoint:** [`outputs/checkpoints/baseline_best.pth`](outputs/checkpoints/baseline_best.pth)
6. **Baseline Validation Embeddings PCA Figure:** [`outputs/figures/baseline_val_pca.png`](outputs/figures/baseline_val_pca.png)
