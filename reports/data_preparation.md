# Phase 2: Data Preparation & Leakage-Safe Manifest Specification
**Project:** MotifWeave — Color-Invariant Saree Design Recognition  
**Client / Specification:** DeepLure AIE-CASE Brief  
**Date:** 2026-10-05  
**Lead ML Engineer:** Antigravity  

---

## 1. Principles & Context of Data Preparation

> [!IMPORTANT]
> **Methodological Disclosure:**  
> Because the supplied DeepLure corpus lacks design-level identity annotations and naturally occurring multi-colorway pairs, Track A evaluates controlled color invariance using source-preserving synthetic transformations. It does not claim to measure fully supervised real-world design identity recognition. Track B provides exploratory retrieval on independently sourced original images.

### Key Rules Followed:
1. **Preservation of Raw Files:** Original images in `data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/` remain untouched.
2. **No Fabricated Design Labels:** Rather than asserting unverified design identities from filenames or folders, each authentic image is assigned a unique `source_id` proxy (`deeplure_001` through `deeplure_165`).
3. **Dual-Track Evaluation Architecture:**
   - **Track A (Controlled Color-Invariance Benchmark):** Evaluates pairs derived from the same underlying source image under controlled colorway transformations.
   - **Track B (Exploratory Original-Image Retrieval):** Evaluates retrieval performance across independently sourced original images without claiming distinct real-world design identities.

---

## 2. Evaluation Definitions & Protocol Formulation

### A. Evaluation Pairs & Negative Proxies
1. **SAME SOURCE / DIFFERENT SYNTHETIC COLORWAY:**
   This is a ground-truth positive pair ($X \leftrightarrow X_{\text{color\_variant\_1}}$) because both images originate from the exact same source image, ensuring identical spatial structure and motif geometry.
2. **DIFFERENT SOURCE / SYNTHETIC VARIANTS:**
   These serve as **"synthetic/source-distinct negatives"**. They are an evaluation proxy because real-world design identities are unavailable; we do not assert that different source images necessarily carry different design identities.
3. **INDEPENDENT ORIGINAL IMAGES:**
   These belong to the exploratory original-image retrieval track. Differing filenames do not prove different design identities.

### B. Track A Benchmark (Controlled Color-Invariance)
- **Gallery Set:** 40 original images from the held-out test split ($X_{i, \text{orig}}$ for $i \in \text{Test}$).
- **Query Set:** 120 controlled synthetic colorway variants ($X_{i, \text{var1}}, X_{i, \text{var2}}, X_{i, \text{var3}}$ for $i \in \text{Test}$).
- **Ground Truth Match Criterion:** 
  $$\text{query source\_id} == \text{gallery source\_id}$$
- **Identification Metrics:**
  - Recall@1, Recall@5, Recall@10, and mAP using cosine similarity.
  - Formally labeled as: **"controlled source-identity retrieval metrics"** (not presented as definitive real-world design recognition accuracy).
- **Verification Evaluation:**
  - **Positive Pairs:** Same `source_id`, different `variant_type`.
  - **Negative Pairs:** Different `source_id`, labeled as **"synthetic/source-distinct negatives"**.
  - **Decision Threshold:** Tuned strictly on the Validation split and applied to the Test split.
- **Same-Color Control Experiment Rule:** Trivial self-matches ($X \leftrightarrow X$) are strictly excluded. Any same-color control experiment must evaluate distinct instances and clearly document pair construction.

### C. Track B Benchmark (Exploratory Original-Image Retrieval)
- Evaluates retrieval across the 40 original unseen catalog images.
- Clearly designated as **"exploratory original-image retrieval"**.

---

## 3. Manifest Schema & Architecture

The manifest is located at [`data/processed/manifest.csv`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/data/processed/manifest.csv) and [`reports/manifest.csv`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/reports/manifest.csv).

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `image_path` | String (Path) | Path to image file. |
| `dataset` | String | Provenance tag: `deeplure_saree_corpus`. |
| `source_id` | String | Unique source identity (`deeplure_001` to `deeplure_165`). |
| `original_filename` | String | Source filename (e.g. `img_982056.jpg`). |
| `identity_type` | String | Explicitly marked as `source_image_proxy`. |
| `category` | String | Coarse fabric craft category: `handloom_sarees`. |
| `split` | String | Deterministic split: `train`, `val`, or `test`. |
| `width` | Integer | Image pixel width. |
| `height` | Integer | Image pixel height. |
| `format` | String | Image encoding format (`JPEG`). |
| `mode` | String | Color channels (`RGB`). |
| `quality_flag` | String | Quality validation status (`pass`). |
| `variant_type` | String | One of `original`, `color_variant_1`, `color_variant_2`, `color_variant_3`. |
| `is_synthetic` | Boolean | `False` for original files; `True` for generated color variants. |
| `transform_params` | String | Mathematical transformation parameters. |

---

## 4. Deterministic Source-Level Splitting & Leakage Guards

Splitting was executed at the source image level using fixed `seed=42`:
- **TRAIN:** 100 sources (60.6%) $\to$ 400 total images (1 orig + 3 variants)
- **VALIDATION:** 25 sources (15.2%) $\to$ 100 total images (1 orig + 3 variants)
- **TEST:** 40 sources (24.2%) $\to$ 160 total images (1 orig + 3 variants)

### Leakage Invariance:
For every source $X$, all of its controlled synthetic colorway variants inherit the exact split assigned to $X$.
$$\text{Train}_{\text{sources}} \cap \text{Val}_{\text{sources}} = \emptyset$$
$$\text{Train}_{\text{sources}} \cap \text{Test}_{\text{sources}} = \emptyset$$
$$\text{Val}_{\text{sources}} \cap \text{Test}_{\text{sources}} = \emptyset$$
Zero overlap across all three splits.
