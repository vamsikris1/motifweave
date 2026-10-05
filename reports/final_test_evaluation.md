# Phase 5: Final Model Selection and Held-Out Test Evaluation Report
**Project:** MotifWeave — Color-Invariant Saree Design Recognition  
**Client / Specification:** DeepLure AIE-CASE Brief (`CIN U62011AP2025OPC118696`)  
**Date:** 2026-10-05  
**Lead ML Engineer:** Antigravity  

---

## 1. Executive Summary

This report documents the **first and only final evaluation** on the held-out **40-source Test split** (160 total images: 40 original photographs and 120 controlled synthetic colorway variants).

In accordance with Phase 5 instructions:
- **Experiment A (Baseline ConvNeXt-Tiny + GeM + MLP Head)** was selected as the **Provisional Final Model** based on rigorous validation evidence.
- The 40-source Test split remained **strictly sealed and untouched** throughout model exploration, architecture selection, hyperparameter tuning, and threshold calibration.
- Operating thresholds for pairwise verification were **calibrated exclusively on the Validation split** prior to test evaluation.
- All evaluation metrics are reported with precise methodological designations: **"Controlled source-identity retrieval metrics"** for Track A and **"Exploratory original-image retrieval"** for Track B.

---

## 2. Dataset & Test Protocol

The dataset partitioning protocol established in Phase 2 was strictly enforced:

$$\begin{array}{lcccc}
\hline
\textbf{Split} & \textbf{Unique Sources} & \textbf{Original Images} & \textbf{Synthetic Variants} & \textbf{Total Images} \\
\hline
\text{Train} & 100 & 100 & 300 & 400 \\
\text{Validation} & 25 & 25 & 75 & 100 \\
\textbf{Held-Out Test} & \textbf{40} & \textbf{40} & \textbf{120} & \textbf{160} \\
\hline
\textbf{Total} & \textbf{165} & \textbf{165} & \textbf{495} & \textbf{660} \\
\hline
\end{array}$$

### Test Split Independence:
1. **Zero Source Leakage:** The 40 test sources were isolated deterministically at project inception (`seed=42`).
2. **Untouched Guarantee:** Neither model weights, nor projection heads, nor decision thresholds were adjusted after unsealing the test split.
3. **No Retraining:** Test evaluation was executed as a strictly read-only benchmark.

---

## 3. Model-Selection Decision & Rationale

Prior to unlocking the test set, **Experiment A** was selected as the final model candidate based on the totality of validation evidence from Phase 4A and Phase 4B. **Experiment A was selected exclusively using validation evidence before the test split was unlocked.**

| Evaluation Dimension | Experiment A (Baseline) | Experiment B (Color-Invariant) | Decision Rationale |
| :--- | :---: | :---: | :--- |
| **Initial Validation Recall@1** | **94.67%** (71 / 75) | 92.00% (69 / 75) | Exp A retrieved 2 additional top-1 candidates |
| **Initial Validation mAP** | **97.33%** | 96.00% | Exp A maintained superior precision |
| **Initial Validation Margin** | **+0.5787** | +0.5555 | Exp A produced cleaner inter-source separation |
| **Unseen-Colorway Recall@1** | **94.67%** (71 / 75) | **94.67%** (71 / 75) | **Tied** across 75 non-overlapping unseen queries |
| **Unseen-Colorway mAP** | **97.33%** | **97.33%** | **Tied** across 75 non-overlapping unseen queries |
| **Unseen-Colorway Margin** | **+0.5770** | +0.5611 | Exp A maintained a $+0.0159$ larger margin |
| **Source-Distinct Cross Similarity** | **0.3058** | 0.3309 | Exp A suppressed false inter-source match scores |

While Experiment B demonstrated slightly higher intra-source positive similarity ($0.8921$ vs $0.8828$), it did so at the expense of higher negative cross-similarity ($0.3309$ vs $0.3058$), resulting in lower separation margins without any retrieval gain on unseen colorways. Therefore, **Experiment A was selected as the final model**.

---

## 4. Track A: Controlled Colorway Retrieval Results (Test Split)

- **Reference Gallery:** 40 original test source images (`variant_type == 'original'`).
- **Queries:** 120 controlled synthetic colorway variants (3 variants per test source).
- **Ground Truth:** `query.source_id == gallery.source_id` (exact self-matches excluded).

### Final Model (Experiment A — Baseline):
| Evaluation Metric | Test Performance | Candidates Retrieved |
| :--- | :---: | :---: |
| **Recall@1** | **97.50%** | **117 / 120** |
| **Recall@5** | **100.00%** | **120 / 120** |
| **Recall@10** | **100.00%** | **120 / 120** |
| **Mean Average Precision (mAP)** | **98.75%** | — |

### Breakdown by Controlled Colorway Variant:
- **Variant 1 (+120° Triadic Hue Shift, Sat x1.02):** Recall@1 = **97.50%** (39/40), Recall@5 = **100.00%**, mAP = **98.75%**, Mean Pos Sim = 0.8844
- **Variant 2 (+240° Triadic Hue Shift, Sat x0.98):** Recall@1 = **97.50%** (39/40), Recall@5 = **100.00%**, mAP = **98.75%**, Mean Pos Sim = 0.8831
- **Variant 3 (+50.4° Dye Shift, Sat x1.20, ValGamma 0.95):** Recall@1 = **97.50%** (39/40), Recall@5 = **100.00%**, mAP = **98.75%**, Mean Pos Sim = 0.8968

### Post-Selection Comparison with Experiment B on Test:
For complete transparency, Experiment B was evaluated once on the identical test split:
- **Experiment B:** Recall@1 = **98.33%** (118/120), Recall@5 = **100.00%**, Recall@10 = **100.00%**, mAP = **99.17%**.
- Both models demonstrate exceptional performance on held-out test data, with top-5 candidate capture reaching **100.00%**.

> [!IMPORTANT]
> **Pre-Registered Selection Integrity:**  
> Experiment A was selected exclusively using validation evidence before the test split was unlocked. Experiment B achieved marginally higher Recall@1 and mAP in the post-selection test comparison, but this result does not alter the pre-registered model selection.  
> Experiment A retained:
> - Better source-distinct negative separation (mean negative similarity $0.2928$ vs $0.3507$)
> - Higher test separation margin ($+0.5966$ vs $+0.5423$)
> - Slightly higher ROC-AUC ($0.9967$ vs $0.9963$)
> - Better validation-calibrated F1 ($0.7367$ vs $0.6944$ under $\theta_{\text{EER}}$)

> [!NOTE]
> These figures represent **controlled source-identity retrieval metrics** on synthetic colorway shifts. They do not constitute proof of real-world design recognition.

---

## 5. Pairwise Verification Results (Test Split)

Computed across all 160 test images:
- **Positive Pairs (Same Source):** $240$ pairs ($\binom{4}{2} \times 40$)
- **Negative Proxy Pairs (Distinct Sources):** $12,480$ pairs ($\binom{160}{2} - 240$)

### Similarity Distributions:
| Metric | Same-Source Positive Pairs | Source-Distinct Negative Proxy Pairs |
| :--- | :---: | :---: |
| **Mean** | **0.8894** | **0.2928** |
| **Standard Deviation** | 0.0523 | 0.2089 |
| **Minimum** | 0.6672 | -0.4601 |
| **Median** | 0.8908 | 0.2918 |
| **Maximum** | 0.9834 | 0.9569 |
| **Separation Margin ($\Delta$)** | \multicolumn{2}{c}{\textbf{+0.5966}} |

- **Test ROC-AUC:** **0.9967** (Area under the Receiver Operating Characteristic curve).
- **Test EER (Empirical Reference):** **1.31%**.

### Verification Performance at Fixed Validation-Calibrated Operating Thresholds:
The decision thresholds were calibrated **strictly on the Validation split** (where $\theta_{\text{EER}} = 0.7588$ and $\theta_{\text{F1}} = 0.8359$):

| Operating Threshold | Accuracy | Precision | Recall (TPR) | Specificity (TNR) | FAR | FRR | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation EER ($\theta = 0.7588$)** | **98.68%** | 59.05% | **97.92%** | 98.69% | 1.31% | 2.08% | **0.7367** |
| **Validation Max-F1 ($\theta = 0.8359$)** | **99.21%** | **74.91%** | 87.08% | **99.44%** | **0.56%** | 12.92% | **0.8054** |

- **Confusion Matrix at $\theta = 0.7588$:** $\text{TP} = 235$, $\text{FP} = 163$, $\text{FN} = 5$, $\text{TN} = 12,317$.
- **Interpretation:** At the validation EER threshold, the model catches **97.92% of all true design matches** while rejecting **98.69% of all impostor pairs**, yielding a False Acceptance Rate of only **1.31%**.

---

## 6. Track B: Exploratory Original-Image Retrieval

The 40 independent original test photographs were evaluated in a self-retrieval matrix ($40 \times 40$, diagonal self-matches excluded).

### 1-Nearest-Neighbor Cosine Similarity Distribution:
- **Mean:** $0.7671$
- **Standard Deviation:** $0.1026$
- **Minimum:** $0.5811$ (Query `deeplure_151` — highly distinct geometric pattern)
- **Median:** $0.7563$
- **Maximum:** $0.9422$ (Query `deeplure_100` matched to `deeplure_107`)

### Key Qualitative Observations from the 1-NN Retrieval Grid:
1. **High-Similarity Candidate Pair ($\text{Sim} > 0.90$):**  
   `deeplure_100` (`h_img_61951.jpg`) and `deeplure_107` (`h_img_642784.jpg`) match with a cosine similarity of **0.9422**. Visual inspection suggests highly similar diamond-brocade grid structures and border proportions, representing notable visual motif similarity without assigning an unverified design identity.
2. **Medium-Similarity Candidate Pair ($0.70 < \text{Sim} < 0.80$):**  
   `deeplure_051` matches `deeplure_156` with $\text{Sim} = 0.7012$. Both display visual similarity in continuous floral creeper (*bel*) border bands, alongside differing field butta density, illustrating structural motif overlap across distinct source items.
3. **Isolated Visual Pattern ($\text{Sim} < 0.60$):**  
   `deeplure_151` (`h_img_90832.jpg`) matches its nearest neighbor with only $\text{Sim} = 0.5811$, reflecting an isolated geometric layout that shares minimal visual feature overlap with other items in the test gallery.

The complete 5-query qualitative grid is saved to [`outputs/final/figures/track_b_qualitative_grid.png`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/outputs/final/figures/track_b_qualitative_grid.png).

---

## 7. Model Efficiency & Production Feasibility

Measured over 50 inference iterations on CPU with batch size 1:

- **Backbone Architecture:** `ConvNeXt-Tiny` (`ConvNeXt_Tiny_Weights.IMAGENET1K_V1`)
- **Embedding Dimension:** 256 ($L_2$ normalized)
- **Total Parameters:** 28,344,673
- **Trainable Parameters:** 526,081 (GeM exponent $p$ + 2-layer MLP projection head)
- **Frozen Parameters:** 27,818,592 (ConvNeXt feature backbone)
- **Trainable Parameter Ratio:** **1.86%**
- **Checkpoint File Size:** 117,662,491 bytes (~112.2 MB)
- **Input Resolution:** $224 \times 224 \times 3$ (Bicubic resize, ImageNet normalization)
- **CPU Inference Latency:** Measured single-image CPU latency in the reported environment: 1029.36 ± 69.95 ms/image at batch size 1.
- **Inference Runtime:** PyTorch 2.14.1+cpu on Windows x64.

---

## 8. Limitations & Scope Boundaries

1. **Absence of Real-World Design Identity Annotations:**  
   The DeepLure source corpus lacks SKU-level design identity metadata. Track A evaluates controlled color invariance using deterministic, source-preserving synthetic chromatic shifts; this demonstrates robustness to the specific controlled hue rotations and saturation shifts evaluated here, but does not establish general real-world color invariance. It does not claim fully supervised validation across human-curated real-world design catalogs.
2. **Kaggle Pattern Dataset Non-Equivalence:**  
   The external Indian Saree Patterns dataset labels (`Ikat`, `Banarasi`, `Pichwai`, `Bandhani`) represent macro craft/weave styles, not fine-grained design identities. They must not be conflated with commercial design recognition.
3. **Draping and Deformation:**  
   The current model is evaluated on catalog product photography. Full 3D draped sarees worn on human models introduce non-rigid folds, pleats, and self-occlusions that require motif segmentation or local keypoint alignment.

---

## 9. Reproducibility Statement

All code, configurations, and split definitions are fully reproducible:
- **Deterministic Split:** `seed = 42` ([`scripts/create_splits.py`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/scripts/create_splits.py))
- **Model Checkpoint:** [`outputs/checkpoints/baseline_best.pth`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/outputs/checkpoints/baseline_best.pth)
- **Evaluation Script:** [`scripts/evaluate_final_test.py`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/scripts/evaluate_final_test.py)
- **Final Metrics JSON:** [`outputs/final/metrics/final_test_metrics.json`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/outputs/final/metrics/final_test_metrics.json)

---

## 10. Final Conclusion & DeepLure Acceptance Criteria

> [!IMPORTANT]
> **Methodological Scope Distinction:**  
> The reported retrieval metrics evaluate controlled source-identity retrieval using synthetic color variants and should not be interpreted as evidence of real-world fine-grained design recognition.  
> - **Controlled Synthetic Source-Identity Retrieval (Demonstrated):** Confirms that the embedding space successfully abstracts spatial image structure and weave geometry away from specific, deterministic chromatic perturbations.  
> - **Real-World Design Recognition (Production Scope):** Requires validating against naturally occurring textile production batches, independent photographer lighting setups, human drape geometry, and authentic commercial design identity annotations.

| DeepLure Specification Requirement | Project Delivery | Status |
| :--- | :--- | :---: |
| **Color-Invariant Representation** | Same design in different colorways maps to cosine similarity $>0.88$, while different designs map to $\sim 0.29$ | **SATISFIED** |
| **Identification / Retrieval** | Track A Test Recall@1 = **97.50%**, Recall@5 = **100.00%**, mAP = **98.75%** | **SATISFIED** |
| **Pairwise Verification** | Test ROC-AUC = **0.9967**, Accuracy = **98.68%**, FAR = **1.31%** at validation threshold | **SATISFIED** |
| **Leakage-Safe Protocol** | Zero source leakage; 40-source Test split untouched until Phase 5 | **SATISFIED** |
| **Modular PyTorch Architecture** | ConvNeXt-Tiny + GeM + MLP Head; 526k trainable parameters | **SATISFIED** |
