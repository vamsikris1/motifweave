# Phase 4B: Unseen-Colorway Validation & Protocol Audit Report
**Project:** MotifWeave — Color-Invariant Saree Design Recognition  
**Client / Specification:** DeepLure AIE-CASE Brief (`CIN U62011AP2025OPC118696`)  
**Date:** 2026-10-05  
**Lead ML Engineer:** Antigravity  

---

## 1. Executive Summary

To rigorously assess whether **Experiment B (Color-Invariant Model)** provides genuine generalization across unseen chromatic variations compared to **Experiment A (Baseline)**, a dedicated **Unseen-Colorway Controlled Benchmark** was created and evaluated.

> [!IMPORTANT]
> **Strict Leakage & Governance Safeguards:**
> - The strictly held-out **40-source Test split** remains **100% untouched and unread**.
> - Neither checkpoint (`baseline_best.pth`, `color_invariant_best.pth`) was modified or retrained.
> - The evaluation was performed on newly generated, visually plausible synthetic colorways that **did not overlap** with the fixed validation palette used during Phase 4 training.
> - Metrics are strictly designated as **"unseen controlled source-identity retrieval metrics"**.

---

## 2. Audit of the Colorway Generation Protocol

Inspection of [`scripts/create_color_variants.py`](scripts/create_color_variants.py) reveals the following technical specifications:

1. **Color Space Used:** HSV (Hue in $[0, 1]$, Saturation in $[0, 1]$, Value in $[0, 1]$).
2. **LAB Transformations:** None. All chromatic rotations and scaling occur in HSV before round-tripping to standard RGB uint8.
3. **Exact Transformation Parameters by Variant:**
   - **Variant 1 (`generate_colorway_1`):**
     - Hue Transformation: `(h + 0.333) % 1.0` $\implies$ $+120.0^\circ$ rotation (Triadic complementary shift)
     - Saturation: `np.clip(s * 1.02, 0.0, 1.0)` ($+2\%$ scale)
     - Value / Brightness: Unchanged (`v` unaltered)
   - **Variant 2 (`generate_colorway_2`):**
     - Hue Transformation: `(h + 0.667) % 1.0` $\implies$ $+240.0^\circ$ rotation (Secondary triadic shift)
     - Saturation: `np.clip(s * 0.98, 0.0, 1.0)` ($-2\%$ scale)
     - Value / Brightness: Unchanged (`v` unaltered)
   - **Variant 3 (`generate_colorway_3`):**
     - Hue Transformation: `(h + 0.140) % 1.0` $\implies$ $+50.4^\circ$ rotation (Warm/cool dye shift)
     - Saturation: `np.clip(s * 1.20, 0.0, 1.0)` ($+20\%$ scale)
     - Value / Brightness: `np.clip(v ** 0.95, 0.0, 1.0)` (Gamma $0.95$ power curve)
4. **Is the Validation Palette Fixed?**  
   **YES.** All 3 variants for each of the 25 validation sources were generated once deterministically in Phase 2 and written as static JPEG files to `data/processed/variants/`.
5. **Does the Validation Transformation Overlap with Training?**  
   **YES.** The training split (`data/processed/splits/train.csv`, 100 sources) was populated using the **exact same 3 transformations** (`color_variant_1, color_variant_2, color_variant_3`). In Experiment A, the model was trained on these exact static images without further color jitter. In Experiment B, dynamic `ColorJitter` ($hue=0.3, sat=0.2, bright=0.15, contrast=0.15$) and `RandomGrayscale` ($p=0.15$) were applied on top during batch generation.

---

## 3. Unseen-Colorway Validation Benchmark Design

To test generalization beyond the fixed $+120^\circ, +240^\circ, +50.4^\circ$ rotations, 3 **new, non-overlapping** colorways were generated across the **same 25 validation source images** via [`scripts/create_unseen_val_colorways.py`](scripts/create_unseen_val_colorways.py):

| Variant Name | Hue Angle Shift | Saturation Scale | Brightness / Value Curve | Visual / Textile Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Unseen 1: Sage / Mint / Teal** | $+85.0^\circ$ ($+0.2361$) | $\times 0.85$ ($-15\%$) | $\times 1.05$ ($+5\%$ lift) | Intermediate hue between $+50^\circ$ and $+120^\circ$; soft pastel silk tone |
| **Unseen 2: Royal Indigo Inversion** | $+180.0^\circ$ ($+0.5000$) | $\times 1.10$ ($+10\%$) | $\text{pow}(v, 1.06)$ (Contrast) | Exact complementary inversion; absent from original benchmark |
| **Unseen 3: Antique Amber / Terracotta** | $-45.0^\circ / +315.0^\circ$ ($+0.8750$) | $\times 0.92$ ($-8\%$) | $\times 0.92$ ($-8\%$ antique dim) | Negative angle rotation into warm amber/copper spectrum |

- **Gallery:** 25 original validation source images (`variant_type == 'original'`).
- **Queries:** 75 unseen synthetic colorway variants (3 per source).
- **Ground Truth:** $\text{query.source\_id} == \text{gallery.source\_id}$.
- **Manifest:** [`data/processed/unseen_val_manifest.csv`](data/processed/unseen_val_manifest.csv)
- **Visual Validation Grids:** 25 four-panel comparison images persisted to [`reports/unseen_colorway_examples/`](reports/unseen_colorway_examples).

---

## 4. Unseen Controlled Source-Identity Retrieval Metrics

Both checkpoints were evaluated on the identical 75 unseen queries and 25 gallery images using standard evaluation transforms ($224 \times 224$, ImageNet normalization):

| Metric | Experiment A (Baseline) | Experiment B (Color-Invariant) | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **Recall@1** | **94.67%** (71 / 75) | **94.67%** (71 / 75) | **0.00% (Identical)** |
| **Recall@5** | **100.00%** (75 / 75) | **100.00%** (75 / 75) | **0.00% (Identical)** |
| **Recall@10** | **100.00%** (75 / 75) | **100.00%** (75 / 75) | **0.00% (Identical)** |
| **mAP** | **97.33%** | **97.33%** | **0.00% (Identical)** |

### Per-Transformation Breakdown (Recall@1):
- **Sage / Mint / Teal ($+85^\circ$):** Exp A = **92.0%** (23/25) vs Exp B = **96.0%** (24/25) $\implies$ **Exp B $+4.0\%$**
- **Royal Indigo Inversion ($+180^\circ$):** Exp A = **96.0%** (24/25) vs Exp B = **96.0%** (24/25) $\implies$ **Tied**
- **Antique Amber / Terracotta ($-45^\circ$):** Exp A = **96.0%** (24/25) vs Exp B = **92.0%** (23/25) $\implies$ **Exp A $+4.0\%$**

---

## 5. Pairwise Color-Invariance Analysis

Cosine similarities between the original gallery images and the unseen queries:

| Metric Dimension | Experiment A (Baseline) | Experiment B (Color-Invariant) | Delta ($\text{B} - \text{A}$) |
| :--- | :---: | :---: | :---: |
| **Positive Cosine Sim (Mean)** | 0.8828 | **0.8921** | $+0.0093$ |
| **Positive Cosine Sim (Std)** | 0.0584 | **0.0582** | $-0.0002$ |
| **Positive Cosine Sim (Min)** | 0.6940 | **0.7246** | **$+0.0306$ (Higher worst-case floor)** |
| **Positive Cosine Sim (Median)** | 0.8870 | **0.8995** | $+0.0125$ |
| **Positive Cosine Sim (Max)** | 0.9799 | **0.9860** | $+0.0061$ |
| **Negative Cosine Sim (Mean)** | **0.3058** | 0.3309 | $+0.0251$ (Lower is better) |
| **Negative Cosine Sim (Std)** | **0.2143** | 0.2209 | $+0.0066$ |
| **Negative Cosine Sim (Min)** | -0.3344 | -0.2850 | $+0.0494$ |
| **Negative Cosine Sim (Median)** | **0.3032** | 0.3212 | $+0.0180$ |
| **Negative Cosine Sim (Max)** | **0.9206** | 0.9321 | $+0.0115$ (Lower is better) |
| **Cosine Separation Margin** | **+0.5770** | +0.5611 | **$-0.0159$ (Exp A has higher margin)** |

---

## 6. Interpretation: Which Conclusion is Supported?

**Supported Finding: C. Results are mixed / inconclusive.**

### Rationale:
1. **Retrieval Equivalence on Unseen Palette:**  
   When evaluated on completely unseen chromatic rotations, both models achieve **identical overall performance** (Recall@1 = 94.67%, mAP = 97.33%, Recall@5 = 100.00%). Exp B does not outperform Exp A in ranking accuracy.
2. **Opposing Per-Transformation Trends:**  
   Exp B performs better on the intermediate sage/mint hue shift ($+85^\circ$, $96\%$ vs $92\%$), while Exp A performs better on the negative amber/copper shift ($-45^\circ$, $96\%$ vs $92\%$), and both tie on the $180^\circ$ inversion ($96\%$).
3. **Margin vs. Positive Alignment Trade-Off:**  
   - Exp B improves positive similarity ($0.8921$ vs $0.8828$) and raises the worst-case positive floor ($0.7246$ vs $0.6940$).
   - Exp A maintains lower negative similarity ($0.3058$ vs $0.3309$), resulting in a superior overall separation margin ($+0.5770$ vs $+0.5611$).
4. **Conclusion:**  
   Neither model can be declared decisively superior based on validation data.
