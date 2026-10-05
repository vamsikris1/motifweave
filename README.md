# MotifWeave

Color-invariant textile surface-design recognition and retrieval for handloom sarees. MotifWeave maps saree photographs to a 256-dimensional unit hypersphere where pairwise distance is governed by weave geometry, brocade motifs, and border structures rather than surface dye colorways.

Primary technical submission artifact: [`MotifWeave_Submission.ipynb`](MotifWeave_Submission.ipynb)

---

## Problem

In Indian ethnic textile manufacturing and e-commerce, sarees featuring identical weave structures (jacquard brocades, floral buttas, zari borders, and geometric pallu layouts) are regularly produced across varied seasonal colorways (e.g., ruby red body with gold zari versus peacock blue body with silver zari).

Conventional computer vision backbones conflate dominant chromatic dye absorption with textile identity. The objective of MotifWeave is **surface-design retrieval independent of color palette**:

$$\text{SAME DESIGN} + \text{DIFFERENT COLORWAY} \longrightarrow \textbf{MATCH}$$
$$\text{DIFFERENT DESIGN} + \text{SAME COLORWAY} \longrightarrow \textbf{DO NOT MATCH}$$

---

## Method

### Architecture
MotifWeave uses a lightweight metric-learning architecture with 98.14% frozen parameters:
1. **Feature Backbone:** ConvNeXt-Tiny (Torchvision `IMAGENET1K_V1`, 27,818,592 parameters, **fully frozen**). Extracts spatial feature maps $(B, 768, 7, 7)$ preserving weave edge geometry.
2. **Pooling Layer:** Generalized Mean (GeM) Pooling with learnable exponent $p = 2.9839$ (1 trainable parameter). Focuses representation on salient high-contrast structural textures rather than uniform dye fields.
3. **Projection Head:** 2-Layer MLP (526,080 trainable parameters):
   - Linear $(768 \to 512)$
   - LayerNorm $(512)$
   - GELU activation
   - Dropout $(p = 0.1)$
   - Linear $(512 \to 256)$
4. **Unit Normalization:** Exact $L_2$ vector normalization mapping outputs to $\mathbb{S}^{255} \subset \mathbb{R}^{256}$ ($\|z\|_2 = 1.000000$), enabling exact dot-product cosine similarity retrieval.

**Parameter Breakdown:**
- Total Parameters: 28,344,673
- Trainable Parameters: 526,081 (1.86%)
- Frozen Parameters: 27,818,592 (98.14%)
- Checkpoint: [`outputs/checkpoints/baseline_best.pth`](outputs/checkpoints/baseline_best.pth) (~112.2 MB)

### Training Method & Objective
- **Loss Objective:** Supervised Contrastive Loss (SupCon) with temperature $\tau = 0.07$.
- **Batch Sampler:** `SourceAwareBatchSampler` enforcing $P = 8$ distinct source designs and $K = 4$ variants per source ($P \times K = 32$ images/mini-batch), ensuring consistent positive and negative contrastive anchors in every mini-batch.
- **Optimization:** AdamW ($\text{lr} = 5 \times 10^{-4}$, $\text{weight\_decay} = 10^{-4}$) scheduled via CosineAnnealingLR over 5 epochs ($\text{min\_lr} = 5 \times 10^{-6}$).
- **Augmentation:** Bicubic resize to $224 \times 224$, random horizontal flips, and ImageNet normalization.
- **Reproducibility Note:** Checkpoint inspection confirms that `baseline_best.pth` was trained with Supervised Contrastive Loss ($\tau = 0.07$, $P = 8, K = 4$), resolving any ambiguity regarding batch-hard triplet loss formulations.

---

## Dataset

### DeepLure Corpus
The provided corpus consists of 165 JPEG RGB photographs under `handloom_sarees/` (0 duplicate files, 0 corrupt images).

### Kaggle Indian Saree Patterns Context
Public Kaggle saree datasets categorize images by broad regional or craft traditions (`Banarasi`, `Ikat`, `Chanderi`, `Kanjeevaram`). These represent **macro craft categories**, not fine-grained design SKU identities. Treating macro labels as class identities induces severe false positive conflation across distinct weave motifs.

### Controlled Synthetic Colorway Benchmark
Because the corpus lacks verified commercial multi-colorway SKU annotations, 3 deterministic, non-trivial colorway variants were generated for each source image (+120° triadic hue shift, +240° triadic hue shift, and +50.4° dye shift), preserving 100% of spatial image geometry, zari highlights, and weave edges. Total dataset size: 660 images.

### Source-Level Splitting Protocol
To prevent data leakage, splitting was performed strictly at the **source image level**:
- **Train Split:** 100 sources (100 originals + 300 variants = 400 images)
- **Validation Split:** 25 sources (25 originals + 75 variants = 100 images)
- **Held-Out Test Split:** 40 sources (40 originals + 120 variants = 160 images)

*Isolation Guarantee:* All synthetic variants inherit the split assignment of their parent source. The 40-source test split remained sealed until final evaluation.

---

## Results

Metrics confirmed against the authoritative final checkpoint (`outputs/checkpoints/baseline_best.pth`) evaluated once on the sealed 40-source held-out test split:

### Retrieval Performance (120 Queries $\to$ 40 Gallery Originals)
- **Recall@1:** **97.50%** (117 / 120 queries correctly matched at rank 1)
- **Recall@5:** **100.00%** (120 / 120 queries correctly matched in top 5)
- **Recall@10:** **100.00%** (120 / 120 queries correctly matched in top 10)
- **Mean Average Precision (mAP):** **98.75%**

### Breakdown by Transformation:
- Variant 1 (+120° Triadic Hue Shift): Recall@1 = 97.50%, Recall@5 = 100.00%, mAP = 98.75%
- Variant 2 (+240° Triadic Hue Shift): Recall@1 = 97.50%, Recall@5 = 100.00%, mAP = 98.75%
- Variant 3 (+50.4° Dye Shift): Recall@1 = 97.50%, Recall@5 = 100.00%, mAP = 98.75%

### Pairwise Verification Performance (240 Positive Pairs, 12,480 Negative Proxy Pairs)
Operating thresholds were calibrated **strictly on the Validation split** with zero test tuning:

| Metric | Validation EER Operating Point | Validation Max-F1 Operating Point |
| :--- | :---: | :---: |
| **Decision Threshold ($\theta$)** | **0.7588** | **0.8359** |
| **Accuracy** | **98.68%** | **99.21%** |
| **Precision** | 59.05% | **74.91%** |
| **Recall (TPR)** | **97.92%** | 87.08% |
| **False Acceptance Rate (FAR)** | **1.31%** | **0.56%** |
| **False Rejection Rate (FRR)** | **2.08%** | 12.92% |
| **F1-Score** | **0.7367** | **0.8054** |

- **Verification ROC-AUC:** **0.9967**
- **Test EER (Reference):** **1.31%**
- **Separation Margin ($\Delta$):** **+0.5966** (Mean Positive: $0.8894 \pm 0.0523$, Mean Negative: $0.2928 \pm 0.2089$)

> **Important Scope Clarification:**  
> These metrics reflect performance on the controlled source-preserving synthetic colorway benchmark. They do **NOT** prove universal real-world color invariance or verified commercial design recognition across uncontrolled real-world illumination, fabric draping, or chemical dye variations.

---

## Installation

Create a virtual environment and install the required dependencies:

```bash
python -m venv .venv
```

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Install Requirements:**
```bash
pip install -r requirements.txt
```

---

## Inference

Reviewers can verify the pipeline using the provided scripts:

### 1. Run 9-Point Verification Suite
Executes end-to-end verification of model loading, L2 normalization, determinism, batch extraction, gallery indexing, top-K retrieval, pairwise verification, parameter immutability, and latency:
```bash
python scripts/test_inference_pipeline.py
```

### 2. Query Reference Gallery
Query the 40-image reference gallery with a sample colorway variant:
```bash
python scripts/query_gallery.py \
    --query data/processed/variants/deeplure_002_variant1.jpg \
    --gallery outputs/final/galleries/test_gallery.npz \
    --top-k 5
```

### 3. Pairwise Design Verification
Verify whether two images share the same surface design using the frozen validation threshold ($\theta = 0.7588$):

**Same Design / Different Colorway:**
```bash
python scripts/verify_pair.py \
    --image-a data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/h_img_149526.jpg \
    --image-b data/processed/variants/deeplure_002_variant1.jpg
```

**Distinct Designs:**
```bash
python scripts/verify_pair.py \
    --image-a data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/h_img_149526.jpg \
    --image-b data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/img_635895.jpg
```

### 4. Interactive Web Application (Optional)
Launch the local web dashboard for interactive visual similarity search and live pairwise verification:
```bash
python app.py
```
Open [http://127.0.0.1:5050](http://127.0.0.1:5050) in any web browser.

---

## Limitations

1. **Lack of Verified Fine-Grained Design IDs:** The raw dataset lacks SKU-level design annotations or confirmed multi-colorway production pairs. True identification accuracy is therefore evaluated on synthetic transformations.
2. **Controlled Synthetic Color Transformations:** Synthetic shifts preserve exact edge coordinates and luminance gradients. Real-world batch dyeing involves physical chemical variations, uneven dye uptake, fiber sheen differences, and metallic zari oxidation.
3. **Exploratory Track B Scope:** Matches between independent catalog photographs indicate structural textural affinity, not verified commercial identity.
4. **Catalog Photography Assumption:** The pipeline assumes planar flat-lay product captures. Sarees draped on human models introduce non-rigid 3D pleating and occlusions that require specialized motif localization before global embedding extraction.

---

## Primary Artifacts

- **Official Technical Submission Notebook:** [`MotifWeave_Submission.ipynb`](MotifWeave_Submission.ipynb)
- **Baseline Configuration:** [`configs/baseline.yaml`](configs/baseline.yaml)
- **Winning Model Checkpoint:** [`outputs/checkpoints/baseline_best.pth`](outputs/checkpoints/baseline_best.pth)
- **40-Source Reference Gallery:** [`outputs/final/galleries/test_gallery.npz`](outputs/final/galleries/test_gallery.npz)
- **Test Metrics Record:** [`outputs/final/metrics/final_test_metrics.json`](outputs/final/metrics/final_test_metrics.json)
