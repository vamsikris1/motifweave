# Phase 6: Inference & Retrieval Implementation Guide
**Project:** MotifWeave — Color-Invariant Saree Design Recognition  
**Client / Specification:** DeepLure AIE-CASE Brief (`CIN U62011AP2025OPC118696`)  
**Date:** 2026-10-05  
**Lead ML Engineer:** Antigravity  

---

## 1. Overview & Pipeline Architecture

The inference and retrieval pipeline implements exact cosine-similarity ranking and pairwise verification for saree designs, strictly decoupling visual surface geometry and motifs from color palettes:

$$\text{Query Image} \xrightarrow{\text{Resize (224, 224) + Norm}} \text{ConvNeXt-Tiny (Frozen)} \xrightarrow{\text{GeM Pooling}} \text{MLP Projection} \xrightarrow{L_2 \text{ Norm}} \mathbf{z}_q \in \mathbb{R}^{256}$$

### Downstream Operations:
1. **Identification / Top-K Retrieval:**
   $$\text{Cosine Similarity Vector: } \mathbf{s} = \mathbf{z}_q \cdot \mathbf{Z}_{\text{gallery}}^T \in [-1, 1]^N \implies \text{Rank descending}$$
2. **Pairwise Verification:**
   $$\text{Cosine Similarity: } s = \mathbf{z}_a \cdot \mathbf{z}_b \in [-1, 1] \implies \begin{cases} \text{SAME}, & s \ge \theta_{\text{calib}} \\ \text{DIFFERENT}, & s < \theta_{\text{calib}} \end{cases}$$

---

## 2. Core Implementation Modules

| File | Purpose | Key Functions / Classes |
| :--- | :--- | :--- |
| [`src/inference.py`](src/inference.py) | Standalone embedding extraction | `load_model()`, `embed_image()`, `embed_images()`, `get_inference_transform()` |
| [`src/retrieval.py`](src/retrieval.py) | Gallery search and pairwise verification | `build_gallery()`, `load_gallery()`, `retrieve()`, `verify_pair()` |
| [`scripts/build_gallery.py`](scripts/build_gallery.py) | CLI to construct and serialize reference gallery | Arguments: `--manifest`, `--split`, `--variant-type`, `--output` |
| [`scripts/query_gallery.py`](scripts/query_gallery.py) | CLI to query gallery for Top-K candidates | Arguments: `--query`, `--gallery`, `--top-k`, `--checkpoint` |
| [`scripts/verify_pair.py`](scripts/verify_pair.py) | CLI for pairwise design verification | Arguments: `--image-a`, `--image-b`, `--threshold`, `--checkpoint` |
| [`scripts/test_inference_pipeline.py`](scripts/test_inference_pipeline.py) | Automated sanity and regression test suite | Verifies 9 essential pipeline invariance properties |

---

## 3. Specifications & Preprocessing

- **Selected Final Checkpoint:** [`outputs/checkpoints/baseline_best.pth`](outputs/checkpoints/baseline_best.pth) (117,662,491 bytes)
- **Architecture:** ConvNeXt-Tiny (`IMAGENET1K_V1`, frozen) $\to$ GeM Pooling ($p=2.9839$) $\to$ Linear(768 $\to$ 512) $\to$ LayerNorm $\to$ GELU $\to$ Dropout(0.1) $\to$ Linear(512 $\to$ 256) $\to$ $L_2$ Normalization.
- **Embedding Format:** Unit-normalized float32 vector, shape `(256,)` ($\|\mathbf{z}\|_2 = 1.000000 \pm 0.000001$).
- **Image Preprocessing:**
  - Resize: `(224, 224)` (Bicubic interpolation)
  - Color space: RGB (3 channels)
  - Normalization: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
- **Operating Decision Threshold:**
  - **$\theta = 0.7588$**
  - **Strict Calibration Protocol:** Calibrated **exclusively on the Validation split** prior to test evaluation at the Equal Error Rate (EER) operating point ($\text{FAR} = \text{FRR} = 2.63\%$). Frozen before unlocking the test split.

---

## 4. Verification & Sanity-Check Results

Execution of [`scripts/test_inference_pipeline.py`](scripts/test_inference_pipeline.py) verified:

1. **Model Loading:** Clean loading in `eval()` mode with all 28.3M parameters set to `requires_grad=False`.
2. **Embedding Shape & Norm:** Extracted embeddings have exact shape `(256,)` and L2 norm `1.000000`.
3. **Determinism:** Repeated inference on the same input produces exact zero difference ($0.0 \times 10^0$).
4. **Batch Consistency:** `embed_images()` matches single-image inference perfectly across batches.
5. **Gallery Indexing:** Gallery `.npz` index builds, serializes, and deserializes without data loss.
6. **Top-K Search:** Returns ranked candidates sorted in descending cosine similarity.
7. **Pairwise Verification:**
   - Same-Source Colorway Pair: Similarity = **0.8648** $\ge 0.7588 \implies$ **SAME (Match)**
   - Distinct-Source Pair: Similarity = **0.6374** $< 0.7588 \implies$ **DIFFERENT (No Match)**
8. **Immutability:** SHA-256 hash of all model weight tensors is identical before and after inference passes.
9. **Execution Latency:**
   - Single-image embedding extraction: $1592.34 \pm 352.94$ ms (measured on local CPU, batch size 1).
   - Gallery search latency: **0.0066 ms** for $N=40$ gallery (embedding-to-gallery similarity computation only, excluding image loading, preprocessing, and neural-network embedding inference).
   - *Deployment note:* GPU or optimized deployment may reduce inference latency, but such acceleration was not measured in this evaluation.

---

## 5. Gallery Terminology & Scope Boundaries

### Gallery Categorization:
1. **Evaluation Gallery:**  
   The 40 original held-out test source images used strictly for the controlled Track A evaluation.
2. **Exploratory / Demo Gallery:**  
   A reference gallery of original images used for exploratory visual retrieval demonstrations.

> [!IMPORTANT]
> **No Unverified Design Identity Claims:**  
> Because the DeepLure corpus lacks verified fine-grained commercial design identities, arbitrary original images must NOT be described as "known designs" or assumed to represent verified commercial design identities.  
> Precise technical terms must be used at all times:
> - *"reference images"*
> - *"gallery images"*
> - *"source images"*
> - *"exploratory visual retrieval"*

### Controlled Retrieval vs. Real-World Recognition:
- **Track A (Controlled):** Validates that synthetic color transformations of the same source image match with cosine similarity $>0.86$.
- **Track B (Exploratory):** Top-K ranking across independent photographs provides visual similarity retrieval based on surface motifs, not verified commercial SKU recognition.

### Data Privacy Protection:
In compliance with DeepLure NDAs, proprietary images in `data/` and model checkpoints in `outputs/checkpoints/` are strictly excluded from version control via `.gitignore`.
