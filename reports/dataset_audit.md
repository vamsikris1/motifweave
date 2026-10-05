# Dataset Audit & Data Preparation Report — MotifWeave

**Project:** MotifWeave — Color-Invariant Saree Design Recognition  
**Client / Specification:** DeepLure AIE-CASE Brief  
**Date:** 2026-10-05  
**Lead ML Engineer:** Antigravity  

---

## 1. Executive Summary

In accordance with Sections 3, 4, 5, 6, and 28 of the DeepLure specification, an exhaustive audit and data preparation pipeline were completed before initiating model training:
1. **Preservation of Raw Files:** Raw files in `data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/` remain untouched.
2. **Zero Fabricated Design Labels:** Raw directory names and file IDs were not assumed to represent fine-grained design identities. Each authentic image is assigned a unique `source_id` (`deeplure_001` through `deeplure_165`).
3. **Deterministic Source-Level Split:** Complete source partitioning prior to variant generation ensures zero leakage.
4. **Controlled Synthetic Colorway Variants:** 3 controlled synthetic colorway variants per source image ($165 \times 3 = 495$ variants; 660 total images) enable rigorous color-invariance evaluation.

> [!IMPORTANT]
> **Methodological Disclosure:**  
> Because the supplied DeepLure corpus lacks design-level identity annotations and naturally occurring multi-colorway pairs, Track A evaluates controlled color invariance using source-preserving synthetic transformations. It does not claim to measure fully supervised real-world design identity recognition. Track B provides exploratory retrieval on independently sourced original images.

---

## 2. Dataset Directory Tree & Inventory

### DeepLure Saree Corpus
```
deep-lure-motifweave/data/
├── raw/
│   └── deeplure_corpus/
│       └── sarees_dataset/
│           ├── handloom_sarees/          [165 authentic raw images, 18.1 MB]
│           └── normal_sarees/            [0 files, EMPTY at remote source]
└── processed/
    ├── base_manifest.csv                 [165 source records]
    ├── manifest.csv                      [660 total records: 165 orig + 495 variants]
    ├── splits/
    │   ├── train_sources.csv             [100 sources]
    │   ├── val_sources.csv               [25 sources]
    │   ├── test_sources.csv              [40 sources]
    │   ├── train.csv                     [400 images]
    │   ├── val.csv                       [100 images]
    │   └── test.csv                      [160 images]
    └── variants/                         [495 controlled synthetic colorway images]
```

---

## 3. Image Dimensions, Validity & Formats

All 165 source images were verified for byte integrity and geometry:
- **Valid Images:** 165 / 165 (100.0%)
- **Corrupt / Unreadable:** 0
- **Format:** 100% JPEG
- **Color Mode:** 100% RGB (3 channels)
- **Width:** Min 226 px, Median 662 px, Max 1512 px (Mean: 814.8 px)
- **Height:** Min 226 px, Median 662 px, Max 1512 px (Mean: 821.5 px)
- **Aspect Ratio:** Mean 0.9925 (>85% are exact 1:1 square crops)

---

## 4. Duplicate & Near-Duplicate Findings

Across all $\binom{165}{2} = 13,530$ possible raw pairs:
- **Exact Duplicates (MD5):** 0
- **Perceptual Near-Duplicates (64-bit dHash $\le 6$ bits):** 0
- **Max Grayscale Cross-Correlation:** 0.5959 (`img_688109.jpg` $\leftrightarrow$ `img_746861.jpg`).
- **Conclusion:** All 165 images represent distinct individual sarees. No pre-existing multi-colorway pairs exist in the raw files.

---

## 5. Dual-Track Evaluation Protocol

To rigorously test color-invariance without fabricating real-world design labels:

### TRACK A — CONTROLLED COLOR-INVARIANCE BENCHMARK
- **Ground Truth Definition:** For any source saree $X_i$, its generated colorways $X_{i, \text{var1}}, X_{i, \text{var2}}, X_{i, \text{var3}}$ originate from the exact same source image, ensuring identical spatial structure and motif geometry.
- **Positive Pairs:** $(X_i, X_{i, \text{var1}})$, $(X_i, X_{i, \text{var2}})$, etc. Explicitly called **"controlled synthetic colorway pairs"**.
- **Negative Pairs:** Formed between distinct source designs ($X_i \leftrightarrow X_j, i \neq j$), explicitly designated as **"synthetic/source-distinct negatives"** (proxy negatives).
- **Match Criterion:** `query source_id == gallery source_id`.
- **Retrieval Metrics:** Recall@1, Recall@5, Recall@10, and mAP, formally designated as **"controlled source-identity retrieval metrics"**.
- **Verification Metrics:** ROC-AUC, EER, and optimal decision threshold tuned strictly on Validation and evaluated on Test.
- **Same-Color Control:** Trivial self-matches ($X \leftrightarrow X$) are excluded.

### TRACK B — EXPLORATORY ORIGINAL-IMAGE RETRIEVAL
- Evaluates retrieval across the 40 unseen original test sarees.
- Labeled explicitly as **"exploratory original-image retrieval"**.
- Does not assert that differing filenames guarantee different real-world design SKUs.

---

## 6. Deterministic Split & Leakage Verification

| Split | Sources | % of Sources | Total Images | Images per Source | Leakage Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | 100 | 60.6% | **400** | 1 orig + 3 variants | **Zero Leakage** |
| **VALIDATION** | 25 | 15.2% | **100** | 1 orig + 3 variants | **Zero Leakage** |
| **TEST** | 40 | 24.2% | **160** | 1 orig + 3 variants | **Zero Leakage** |
| **TOTAL** | **165** | **100.0%** | **660** | 4 images each | **100% Disjoint** |

### Formal Leakage Invariance:
For any source $S \in \text{Test}$, $\{S, S_{\text{var1}}, S_{\text{var2}}, S_{\text{var3}}\} \cap \text{Train} = \emptyset$ and $\{S, S_{\text{var1}}, S_{\text{var2}}, S_{\text{var3}}\} \cap \text{Val} = \emptyset$.
Test data remains strictly untouched until final evaluation.
