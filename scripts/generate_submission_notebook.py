"""
Generates and executes MotifWeave_Submission.ipynb with real, reproducible outputs.
"""
import os
import sys
import io
import json
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_and_execute_notebook():
    cells = []
    global_env = {
        "__file__": str(PROJECT_ROOT / "MotifWeave_Submission.ipynb"),
        "PROJECT_ROOT": PROJECT_ROOT
    }
    
    # Ensure project root is in sys.path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })
        
    def add_code(code_str):
        cell_idx = len([c for c in cells if c["cell_type"] == "code"]) + 1
        
        # Execute code in global_env and capture stdout/stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        error_occurred = False
        
        # Switch working directory to PROJECT_ROOT during execution
        old_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code_str, global_env)
        except Exception as e:
            error_occurred = True
            stderr_buf.write(traceback.format_exc())
        finally:
            os.chdir(old_cwd)
            
        outputs = []
        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()
        
        if stdout_text:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_text.splitlines()]
            })
        if stderr_text:
            outputs.append({
                "name": "stderr",
                "output_type": "stream",
                "text": [line + "\n" for line in stderr_text.splitlines()]
            })
            
        if error_occurred:
            print(f"Error in cell {cell_idx}:\n{stderr_text}")
            raise RuntimeError(f"Cell {cell_idx} failed execution!")
            
        cells.append({
            "cell_type": "code",
            "execution_count": cell_idx,
            "metadata": {},
            "outputs": outputs,
            "source": [line + "\n" for line in code_str.strip().split("\n")]
        })

    # =========================================================================
    # 1. TITLE & SUBMISSION HEADER
    # =========================================================================
    add_md("""# MotifWeave — Color-Invariant Saree Surface-Design Retrieval
### Official Technical Submission Notebook | DeepLure AIE-CASE Project Brief
**Client Organization:** DeepLure (`CIN U62011AP2025OPC118696`)  
**Lead ML Systems Engineer:** Antigravity  
**Specification Target:** Color-Invariant Saree Motif Recognition, Fine-Grained Retrieval, and Pairwise Verification  
**Primary Artifact:** `MotifWeave_Submission.ipynb`  
**Winning Checkpoint:** `outputs/checkpoints/baseline_best.pth` (112.2 MB)  
**Governance Status:** Evaluated Once on Sealed 40-Source Test Split | Calibrated Exclusively on Validation Split
""")

    # =========================================================================
    # 2. EXECUTIVE SUMMARY
    # =========================================================================
    add_md("""## 1. Executive Summary

In Indian ethnic textile manufacturing and e-commerce, sarees are frequently manufactured with identical weave structures (borders, jacquard brocades, pallu arrangements, buttas) produced in distinct seasonal colorways (e.g., ruby red body with gold zari versus peacock blue body with silver zari). Conventional computer vision models conflate chromatic dominance with design identity, leading to high false negatives across different colorways of the same design, and high false positives across different designs sharing similar dominant dyes.

**MotifWeave** solves this challenge by mapping saree images to a compact **256-dimensional unit hypersphere** where distance is governed purely by motif geometry, weave layouts, and structural boundaries rather than surface color palettes:

$$\\text{SAME DESIGN} + \\text{DIFFERENT COLOR} \\longrightarrow \\text{HIGH SIMILARITY (MATCH)}$$
$$\\text{DIFFERENT DESIGN} + \\text{SAME COLOR} \\longrightarrow \\text{LOW SIMILARITY (NO MATCH)}$$

### Key Achievements on Held-Out Test Split (40 Sealed Sources):
- **Track A Top-1 Identification (Recall@1):** **97.50%** (117 / 120 queries correctly identified)
- **Track A Top-5 Identification (Recall@5):** **100.00%** (120 / 120 queries retrieved)
- **Mean Average Precision (mAP):** **98.75%**
- **Pairwise Verification ROC-AUC:** **0.9967** across 12,720 test pairs
- **Cosine Separation Margin ($\\Delta$):** **+0.5966** (Mean Pos = $0.8894$, Mean Neg = $0.2928$)
- **Equal Error Rate (EER):** **1.31%** (Operating Accuracy **98.68%** at validation-calibrated threshold $\\theta = 0.7588$)
- **Compact & Fast:** 28.34M total parameters (98.14% frozen backbone, only 526K trainable parameters), 256-D float32 embeddings (1 KB/image), and sub-millisecond gallery search.
""")

    # =========================================================================
    # 3. ENVIRONMENT & DEPENDENCY VERIFICATION
    # =========================================================================
    add_md("""## 2. Environment & Dependency Verification

To ensure full reproducibility, the following code cell verifies the runtime environment, active PyTorch version, torchvision build, and GPU/CPU availability:
""")

    # Code cell 1: Environment check
    code_env = '''import sys
import platform
import numpy as np
import torch
import torchvision

print("=" * 70)
print("MOTIFWEAVE RUNTIME ENVIRONMENT VERIFICATION")
print("=" * 70)
print(f"Operating System:     {platform.system()} {platform.release()} ({platform.architecture()[0]})")
print(f"Python Executable:    {sys.executable}")
print(f"Python Version:       {platform.python_version()}")
print(f"PyTorch Version:      {torch.__version__}")
print(f"Torchvision Version:  {torchvision.__version__}")
print(f"NumPy Version:        {np.__version__}")
print(f"Compute Device:       {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
print(f"Torch CPU Threads:    {torch.get_num_threads()}")
print("=" * 70)
'''
    add_code(code_env)

    # =========================================================================
    # 4. PROBLEM FORMULATION
    # =========================================================================
    add_md("""## 3. Problem Formulation

Let $\\mathcal{X}$ denote the space of textile surface photographs. Each saree image $x \\in \\mathcal{X}$ contains:
1. **Structural Design Elements ($\\mathcal{D}$):** Motifs, jacquard weaves, buttas, border geometric patterns, and pallu layouts.
2. **Chromatic Attributes ($\\mathcal{C}$):** Base fabric dye, warp/weft hue, zari reflectance, and lighting conditions.

We formulate the task as learning a metric embedding function:
$$f_\\theta: \\mathcal{X} \\longrightarrow \\mathcal{S}^{D-1} \\subset \\mathbb{R}^D, \\quad \\|f_\\theta(x)\\|_2 = 1, \\quad D=256$$

such that the pairwise cosine similarity:
$$s(x_i, x_j) = \\langle f_\\theta(x_i), f_\\theta(x_j) \\rangle = \\sum_{k=1}^D f_\\theta(x_i)_k \\cdot f_\\theta(x_j)_k$$

satisfies:
$$s(x_i, x_j) \\ge \\theta \\iff \\text{Design}(x_i) = \\text{Design}(x_j)$$
$$s(x_i, x_j) < \\theta \\iff \\text{Design}(x_i) \\neq \\text{Design}(x_j)$$

independent of the colorway variations $\\mathcal{C}(x_i)$ and $\\mathcal{C}(x_j)$.
""")

    # =========================================================================
    # 5. 500-CHARACTER APPROACH NOTE
    # =========================================================================
    add_md("""## 4. 500-Character Approach Note (DeepLure Submission Format)

As requested by the DeepLure Job Description, below is the concise approach note summarizing the actual winning architecture, training strategy, and loss function:
""")

    # Code cell 2: Approach note & verification
    code_approach_note = '''approach_note = (
    "MotifWeave recognizes saree designs invariantly to color palettes. Architecture: frozen ConvNeXt-Tiny "
    "extracts visual textures, pooled via learnable GeM (p=3.0) and mapped by a 2-layer MLP to a 256-D "
    "L2-normalized hypersphere. Preprocessing: 224x224 bicubic resize with ImageNet normalization. Strategy: "
    "source-aware PxK batch sampling (P=8, K=4) trained with Supervised Contrastive Loss (tau=0.07) and AdamW "
    "across controlled synthetic colorway shifts and spatial flips."
)

char_count = len(approach_note)
print(f"Authoritative Approach Note:\\n\\\"{approach_note}\\\"\\n")
print(f"Character Count: {char_count} characters (Strictly <= 500 character limit: {char_count <= 500})")
assert char_count <= 500, f"Approach note exceeds 500 characters: {char_count}"
'''
    add_code(code_approach_note)

    # =========================================================================
    # 6. DATASET & GOVERNANCE
    # =========================================================================
    add_md("""## 5. Dataset Audit & Data Governance

### 5.1 Corpus Composition
The provided corpus located at `data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/` contains **exactly 165 JPEG RGB photographs**:
- Resolution range: $225 \\times 225$ to $1000 \\times 1000$ pixels.
- Exact file integrity: 165 valid images, 0 corrupt files, 0 exact byte/perceptual duplicates.

### 5.2 Why Kaggle Macro Labels Are NOT Fine-Grained Design Identities
The public Kaggle Indian Saree Patterns dataset labels images into broad regional craft categories:
- `Ikat` (tie-dye technique)
- `Banarasi` (brocade weaving region)
- `Bandhani` (tie-dye dot patterns)
- `Pichwai` (devotional painted motifs)

**Critical Methodological Principle:** Macro craft categories are **not** fine-grained design SKU identities. Two completely different `Banarasi` sarees do **not** share the same motif or weave pattern. Treating macro labels as identical designs would introduce massive false-positive label noise and destroy metric-learning representation quality.

### 5.3 Data Privacy & Confidentiality
In compliance with DeepLure corporate confidentiality (`CIN U62011AP2025OPC118696`):
- Raw customer photographs are strictly excluded from version control via `.gitignore`.
- Trained `.pth` binary checkpoints remain strictly local and are not distributed to public repositories.
""")

    # =========================================================================
    # 7. DATASET AUDIT & LEAKAGE-SAFE SPLITS
    # =========================================================================
    add_md("""## 6. Leakage-Safe Dataset Split Protocol

To evaluate color invariance without fabricating design labels, 3 controlled synthetic colorway variants were deterministically generated for each source image (+120° triadic shift, +240° triadic shift, and +50.4° dye shift), preserving spatial image structure, motif locations, and weave boundaries while modifying chromatic appearance.

### Source-Level Deterministic Split (`seed=42`):
| Split | Unique Sources | Original Images | Synthetic Variants | Total Images | Usage Policy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 100 | 100 | 300 | 400 | Model optimization |
| **Validation** | 25 | 25 | 75 | 100 | Checkpointing & threshold calibration |
| **Held-Out Test** | **40** | **40** | **120** | **160** | **Sealed; evaluated once in Phase 5** |
| **Total** | **165** | **165** | **495** | **660** | Complete dataset |

**Leakage Protection:** All synthetic variants inherit the exact split assignment of their parent source photograph. There is **zero source overlap** across splits.
""")

    # =========================================================================
    # 8. METHOD & ARCHITECTURE
    # =========================================================================
    add_md("""## 7. Model Architecture

The MotifWeave architecture employs a modern modular structure specifically tailored for transfer learning from small, high-value textile corpora:

```text
Input Saree Image (3 x 224 x 224)
        │
        ▼
ConvNeXt-Tiny Feature Backbone (Torchvision IMAGENET1K_V1, Frozen)
        │   Output: Feature Maps (B, 768, 7, 7)
        ▼
Generalized Mean (GeM) Pooling (Learnable Exponent p = 2.9839)
        │   Output: Salient Structural Vectors (B, 768)
        ▼
2-Layer MLP Projection Head:
    Linear(768 -> 512)
    LayerNorm(512)
    GELU()
    Dropout(p = 0.1)
    Linear(512 -> 256)
        │   Output: Unconstrained Embeddings (B, 256)
        ▼
Exact L2 Normalization (z / ||z||_2)
        │
        ▼
Unit Embedding Vector z in R^256  (||z||_2 = 1.000000)
```

### Parameter Accounting:
| Component | Layer Type | Output Dimension | Total Parameters | Trainable Parameters | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Backbone** | ConvNeXt-Tiny (`IMAGENET1K_V1`) | $768 \\times 7 \\times 7$ | 27,818,592 | 0 | **Frozen** (98.14%) |
| **Pooling** | GeM Pooling ($p=2.9839$) | 768 | 1 | 1 | **Trainable** (<0.01%) |
| **Projection** | Linear $768 \\to 512$ | 512 | 393,728 | 393,728 | **Trainable** (1.39%) |
| | LayerNorm(512) | 512 | 1,024 | 1,024 | **Trainable** (<0.01%) |
| | Linear $512 \\to 256$ | 256 | 131,328 | 131,328 | **Trainable** (0.46%) |
| **Total** | **Full MotifWeave Pipeline** | **256-D** | **28,344,673** | **526,081** | **1.86% Trainable** |
""")

    # =========================================================================
    # 9. TRAINING STRATEGY & REPRODUCIBILITY AUDIT
    # =========================================================================
    add_md("""## 8. Training Strategy & Reproducibility Audit (CASE A Confirmed)

### 8.1 Authoritative Training Configuration
The submitted checkpoint `outputs/checkpoints/baseline_best.pth` was trained under **Experiment A** with the following audited parameters:
- **Metric Loss:** **Supervised Contrastive Loss (SupCon)** (Khosla et al., NeurIPS 2020) with temperature $\\tau = 0.07$.
- **Batch Sampler:** `SourceAwareBatchSampler` guaranteeing $P=8$ distinct sources and $K=4$ variants per source ($P \\times K = 32$ images/mini-batch).
- **Augmentation Pipeline:** Standard ImageNet augmentations (`Resize(256)`, `RandomResizedCrop(224, scale=(0.85, 1.0))`, `RandomHorizontalFlip(p=0.5)`, ImageNet Normalization). No aggressive color jitter was applied to Experiment A.
- **Optimizer:** AdamW with initial learning rate $\\eta = 5 \\times 10^{-4}$ and weight decay $\\lambda = 10^{-4}$.
- **Learning Rate Scheduler:** CosineAnnealingLR over 5 epochs with minimum learning rate $\\eta_{\\min} = 5 \\times 10^{-6}$.
- **Random Seed:** 42.

### 8.2 Resolution of Methodological Questions
- **Supervised Contrastive vs Triplet Loss:** Inspection of `configs/baseline.yaml`, `src/losses.py`, and `outputs/checkpoints/baseline_best.pth` confirms that `baseline_best.pth` was trained with **Supervised Contrastive Loss** ($\\tau=0.07$), not Triplet Loss.
- **Ablation Comparison:** In Phase 4B, an ablation model (Experiment B) with on-the-fly `ColorJitter` and `RandomGrayscale` was trained. Experiment A was pre-registered and selected as the superior final model because it achieved a **higher validation cosine margin (+0.5787 vs +0.5555)**, higher initial Recall@1 (94.67% vs 92.00%), identical unseen-colorway retrieval (94.67%), and significantly lower false positive cross-source similarity (0.3058 vs 0.3309).
""")

    # Code cell 3: Inspect Checkpoint directly
    code_inspect_ckpt = '''import torch
from pathlib import Path

ckpt_path = Path("outputs/checkpoints/baseline_best.pth")
assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

ckpt = torch.load(ckpt_path, map_location="cpu")
config = ckpt.get("config", {})

print("=" * 70)
print("AUDITED CHECKPOINT METADATA: baseline_best.pth")
print("=" * 70)
print(f"Epoch:                {ckpt.get('epoch')}")
print(f"Validation Loss:      {ckpt.get('val_loss')}")
print(f"Separation Margin:    {ckpt.get('sim_margin')}")
print(f"Loss Objective:       {config.get('loss')} (tau={config.get('temperature')})")
print(f"Batch Sampler:        P={config.get('p_sources')} sources, K={config.get('k_variants')} variants (Batch size={config.get('batch_size')})")
print(f"Augmentation:         {config.get('augmentation')}")
print(f"Optimizer:            {config.get('optimizer')} (lr={config.get('learning_rate')}, wd={config.get('weight_decay')})")
print(f"Scheduler:            {config.get('scheduler')} (min_lr={config.get('min_lr')})")
print(f"Total Parameters:     {ckpt.get('total_params'):,}")
print(f"Trainable Parameters: {ckpt.get('trainable_params'):,}")
print(f"GeM Pooling Exp p:    {ckpt['model_state_dict']['pool.p'].item():.4f}")
print("=" * 70)
'''
    add_code(code_inspect_ckpt)

    # =========================================================================
    # 10. INFERENCE PIPELINE
    # =========================================================================
    add_md("""## 9. Inference Pipeline

The inference pipeline maps any input saree image (RGB PIL Image or file path) to a 256-D unit-normalized vector:
1. Load image and convert to RGB.
2. Bicubic resize to $224 \\times 224$.
3. ImageNet channel-wise normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
4. Forward pass through frozen ConvNeXt backbone, GeM pooling, and MLP head.
5. Exact $L_2$ normalization: $z = \\frac{v}{\\|v\\|_2}$.
""")

    # Code cell 4: Test inference functions
    code_inference = '''import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model, get_inference_transform, embed_image

# Load submitted checkpoint in evaluation mode
model, config, device = load_model("outputs/checkpoints/baseline_best.pth", device=torch.device('cpu'))
transform = get_inference_transform(config.get('image_size', 224))

# Test embedding extraction on a sample image
sample_img = "data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/h_img_149526.jpg"
emb = embed_image(sample_img, model=model, transform=transform, device=device)

print(f"Extracted Embedding Shape:    {emb.shape}")
print(f"Embedding L2 Norm:            {np.linalg.norm(emb):.6f}")
print(f"Embedding Dtype:              {emb.dtype}")
print(f"Embedding First 5 Dimensions: {np.round(emb[:5], 4)}")
assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5), "Embedding is not unit-normalized!"
'''
    add_code(code_inference)

    # =========================================================================
    # 11. RETRIEVAL PIPELINE
    # =========================================================================
    add_md("""## 10. Gallery Retrieval Pipeline

The retrieval pipeline supports fast Top-$K$ retrieval against pre-indexed reference galleries:
1. **Offline Gallery Indexing:** Embeddings for all reference gallery images are extracted and serialized into compressed `.npz` archives along with metadata (`source_id`, `image_path`).
2. **Online Query Processing:** For query vector $q \\in \\mathbb{R}^{256}$ and gallery matrix $G \\in \\mathbb{R}^{N \\times 256}$, cosine similarities are computed via dot product:
   $$S = G \\cdot q \\in [-1, 1]^N$$
3. **Top-$K$ Selection:** Candidate indices are sorted in descending order of similarity.
""")

    # Code cell 5: Query gallery
    code_retrieval = '''from src.retrieval import load_gallery, retrieve

gallery = load_gallery("outputs/final/galleries/test_gallery.npz")
print(f"Loaded Reference Gallery: {gallery['num_items']} images (Index dimension: {gallery['embeddings'].shape})")

# Query with synthetic colorway variant 1 of source deeplure_002
query_img = "data/processed/variants/deeplure_002_variant1.jpg"
matches = retrieve(query_img, gallery, top_k=5, model=model, transform=transform, device=device)

print("\\n" + "=" * 75)
print("TOP-5 GALLERY RETRIEVAL RESULTS FOR QUERY: deeplure_002_variant1.jpg")
print("=" * 75)
print(f"{'Rank':<6} | {'Similarity':<10} | {'Source ID':<15} | {'Gallery Image'}")
print("-" * 75)
for m in matches:
    print(f"{m['rank']:<6} | {m['similarity']:<10.4f} | {m['source_id']:<15} | {Path(m['image_path']).name}")
print("=" * 75)
'''
    add_code(code_retrieval)

    # =========================================================================
    # 12. VERIFICATION PIPELINE
    # =========================================================================
    add_md("""## 11. Pairwise Design Verification

Given two saree photographs $x_A$ and $x_B$, MotifWeave computes their embedding cosine similarity and classifies whether they share the same surface design:

$$\\hat{y} = \\begin{cases} \\textbf{SAME DESIGN}, & \\text{if } \\langle f_\\theta(x_A), f_\\theta(x_B) \\rangle \\ge \\theta_{\\text{calibrated}} \\\\[4pt] \\textbf{DIFFERENT DESIGN}, & \\text{if } \\langle f_\\theta(x_A), f_\\theta(x_B) \\rangle < \\theta_{\\text{calibrated}} \\end{cases}$$

### Calibration Governance (Zero Test Leakage):
The decision threshold was calibrated **strictly on the 25-source Validation split**:
- **Operating Threshold ($\\theta_{\\text{EER}}$):** **0.7588** (selected where validation FAR = FRR)
- **High-Precision Threshold ($\\theta_{\\text{F1}}$):** **0.8359** (selected to maximize validation F1-score)
""")

    # Code cell 6: Verify pairs
    code_verify = '''from src.retrieval import verify_pair

img_source = "data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/h_img_149526.jpg"
img_color_variant = "data/processed/variants/deeplure_002_variant1.jpg"
img_different_design = "data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/img_635895.jpg"

print("=" * 70)
print("TEST 1: SAME DESIGN / DIFFERENT SYNTHETIC COLORWAY")
res1 = verify_pair(img_source, img_color_variant, threshold=0.7588, model=model, transform=transform, device=device)
print(f"Cosine Similarity:  {res1['similarity']:.4f}")
print(f"Decision Threshold: {res1['threshold']:.4f}")
print(f"Classification:     {res1['prediction']} (Expected: SAME)")
print("-" * 70)

print("TEST 2: DIFFERENT SOURCE DESIGNS")
res2 = verify_pair(img_source, img_different_design, threshold=0.7588, model=model, transform=transform, device=device)
print(f"Cosine Similarity:  {res2['similarity']:.4f}")
print(f"Decision Threshold: {res2['threshold']:.4f}")
print(f"Classification:     {res2['prediction']} (Expected: DIFFERENT)")
print("=" * 70)
'''
    add_code(code_verify)

    # =========================================================================
    # 13. EVALUATION PROTOCOL
    # =========================================================================
    add_md("""## 12. Evaluation Protocol: Dual-Track Design

To balance scientific rigor with honest empirical reporting, evaluation was divided into two distinct tracks:

### Track A: Controlled Source-Identity Retrieval (Ground-Truth Identification)
- **Gallery:** 40 original held-out test source photographs.
- **Queries:** 120 synthetic colorway variants (3 non-trivial chromatic transforms per source).
- **Ground Truth:** Positive match if `query.source_id == gallery.source_id`.
- **Constraint:** Exact self-matches (querying the original image against itself) are strictly excluded.
- **Metrics:** Recall@1, Recall@5, Recall@10, Mean Average Precision (mAP).

### Track B: Exploratory Original-Image Retrieval (Visual Association Analysis)
- **Dataset:** All 40 independent original photographs in the test set.
- **Protocol:** Compute pairwise similarity matrix across independent real-world photos.
- **Honest Caveat:** The dataset does **not** contain verified commercial multi-colorway SKU pairs. Track B is an exploratory visual similarity analysis and must **not** be presented as ground-truth design recognition.
""")

    # =========================================================================
    # 14. FINAL TEST RESULTS
    # =========================================================================
    add_md("""## 13. Authoritative Final Test Results

The results below were obtained from the final evaluation on the sealed 40-source test split (`outputs/final/metrics/final_test_metrics.json`):

### Track A: Identification Performance (120 Queries $\\to$ 40 Gallery Originals)
| Metric | Test Split Result | Successful Queries / Total |
| :--- | :---: | :---: |
| **Recall@1** | **97.50%** | **117 / 120** |
| **Recall@5** | **100.00%** | **120 / 120** |
| **Recall@10** | **100.00%** | **120 / 120** |
| **Mean Average Precision (mAP)** | **98.75%** | — |

### Breakdown by Synthetic Variant Type:
- **Variant 1 (+120° Triadic Hue Shift):** Recall@1 = **97.50%**, Recall@5 = **100.00%**, mAP = **98.75%**
- **Variant 2 (+240° Triadic Hue Shift):** Recall@1 = **97.50%**, Recall@5 = **100.00%**, mAP = **98.75%**
- **Variant 3 (+50.4° Secondary Dye Shift):** Recall@1 = **97.50%**, Recall@5 = **100.00%**, mAP = **98.75%**

### Pairwise Verification Performance (240 Positive Pairs, 12,480 Negative Proxy Pairs):
- **Verification ROC-AUC:** **0.9967**
- **Cosine Separation Margin ($\\Delta$):** **+0.5966**
- **Test EER (Reference):** **1.31%**

| Operating Point | Threshold ($\\theta$) | Accuracy | Precision | Recall (TPR) | Specificity (TNR) | FAR | FRR | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation EER Point** | **0.7588** | **98.68%** | 59.05% | **97.92%** | 98.69% | **1.31%** | **2.08%** | **0.7367** |
| **Validation Max-F1 Point** | **0.8359** | **99.21%** | **74.91%** | 87.08% | **99.44%** | **0.56%** | 12.92% | **0.8054** |
""")

    # Code cell 7: Load and verify test metrics from JSON
    code_metrics = '''import json
from pathlib import Path

metrics_path = Path("outputs/final/metrics/final_test_metrics.json")
with open(metrics_path, "r") as f:
    test_metrics = json.load(f)

print("=" * 70)
print("PROGRAMMATIC VERIFICATION OF HELD-OUT TEST METRICS")
print("=" * 70)
track_a = test_metrics["track_a_test_retrieval"]
print(f"Track A Recall@1:    {track_a['Recall@1']:.2f}%")
print(f"Track A Recall@5:    {track_a['Recall@5']:.2f}%")
print(f"Track A Recall@10:   {track_a['Recall@10']:.2f}%")
print(f"Track A mAP:         {track_a['mAP']:.2f}%")

pair_stats = test_metrics["pairwise_verification_test"]
print(f"Verification ROC-AUC:{pair_stats['roc_auc']:.4f}")
print(f"Separation Margin:   {pair_stats['similarity_margin']:+.4f}")
print(f"Test EER Reference:  {pair_stats['test_eer_reference']*100:.2f}%")

cal_eer = pair_stats["at_calibrated_eer_threshold"]
print(f"At Val EER Threshold ({cal_eer['threshold']:.4f}):")
print(f"  Accuracy:          {cal_eer['accuracy']*100:.2f}%")
print(f"  FAR:               {cal_eer['FAR']*100:.2f}%")
print(f"  FRR:               {cal_eer['FRR']*100:.2f}%")
print(f"  F1-Score:          {cal_eer['f1_score']:.4f}")
print("=" * 70)
'''
    add_code(code_metrics)

    # =========================================================================
    # 15. VERIFICATION STATISTICS
    # =========================================================================
    add_md("""## 14. Verification Statistics & Separation Margin

The distribution of pairwise cosine similarities confirms clean bimodal separation between identical designs across synthetic colorways and distinct designs:

| Distribution Category | Pair Count | Mean Similarity | Std Dev | Min Similarity | Median Similarity | Max Similarity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Same-Source Pairs (Positive)** | 240 | **0.8894** | 0.0523 | 0.6672 | 0.8908 | 0.9834 |
| **Different-Source Pairs (Negative)** | 12,480 | **0.2928** | 0.2089 | -0.4601 | 0.2918 | 0.9569 |
| **Separation Margin ($\\Delta$)** | — | **+0.5966** | — | — | — | — |

- **True Positive Retention:** At threshold $\\theta = 0.7588$, 235 out of 240 positive pairs are correctly matched (TPR = **97.92%**).
- **False Positive Suppression:** Out of 12,480 negative proxy pairs, only 163 fall above the threshold (FAR = **1.31%**).
""")

    # =========================================================================
    # 16. EFFICIENCY & LATENCY
    # =========================================================================
    add_md("""## 15. Computational Efficiency & Latency Profile

### Model Footprint:
- **Parameter Count:** 28,344,673 parameters total; only 526,081 trainable (1.86%).
- **Embedding Dimension:** 256 float32 values ($1,024$ bytes / $1$ KB per image).
- **Checkpoint Footprint:** 112.2 MB (`baseline_best.pth`).
- **Memory Footprint at Inference:** <450 MB RAM on CPU.

### Measured Latency Profile (Standard CPU Environment):
- **Single-Image Embedding Inference:** **$618.92 \\pm 44.52$ ms/image**  
  *(Includes image loading, bicubic resize to 224x224, ImageNet normalization, neural-network forward pass, and exact L2 normalization).*
- **Gallery Search Latency ($N=40$):** **0.0066 ms**  
  *(Embedding-to-gallery matrix dot-product and top-$K$ sorting only).*

> **Deployment Note:** GPU or optimized deployment engines (e.g. TensorRT, ONNX Runtime) may reduce inference latency, but such acceleration was not measured in this evaluation.
""")

    # =========================================================================
    # 17. REPRODUCIBILITY & ENVIRONMENT
    # =========================================================================
    add_md("""## 16. Reproducibility & Environment Setup

MotifWeave is designed for zero-friction reproducibility in standard Python environments:

### Environment Specifications:
- **Python Version:** 3.12 (Miniconda / Anaconda)
- **PyTorch Version:** 2.14.1+cpu (also compatible with GPU versions)
- **Torchvision Version:** 0.29.1+cpu
- **NumPy Version:** 2.x
- **Random Seed:** Locked to `42` across NumPy, PyTorch, and Python random.

### Setup Instructions:
```bash
# Clone and enter directory
cd deep-lure-motifweave

# Install dependencies
pip install torch torchvision numpy pandas pillow matplotlib pyyaml
```
""")

    # =========================================================================
    # 18. REVIEWER QUICK START
    # =========================================================================
    add_md("""## 17. Reviewer Quick Start Guide

Technical reviewers can verify the entire pipeline using the pre-built CLI tools:

### 1. Run the 9-Point Inference Sanity Check Suite
Verifies model loading, L2 normalization, determinism, gallery indexing, top-$K$ retrieval, pairwise verification, parameter immutability, and latency:
```bash
python scripts/test_inference_pipeline.py
```

### 2. Query the Reference Gallery
```bash
python scripts/query_gallery.py \\
    --query data/processed/variants/deeplure_002_variant1.jpg \\
    --gallery outputs/final/galleries/test_gallery.npz \\
    --top-k 5
```

### 3. Pairwise Verification Demo (Same Design / Different Colorway)
```bash
python scripts/verify_pair.py \\
    --image-a data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/h_img_149526.jpg \\
    --image-b data/processed/variants/deeplure_002_variant1.jpg
```

### 4. Pairwise Verification Demo (Distinct Designs)
```bash
python scripts/verify_pair.py \\
    --image-a data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/h_img_149526.jpg \\
    --image-b data/raw/deeplure_corpus/sarees_dataset/handloom_sarees/img_635895.jpg
```
""")

    # =========================================================================
    # 19. LIMITATIONS & FAILURE MODES
    # =========================================================================
    add_md("""## 18. Limitations & Failure Modes

To ensure production integrity, the following limitations are explicitly noted:
1. **Lack of Commercial SKU Labels:** The available raw corpus does not include commercial catalog SKU identity tags or ground-truth multi-colorway pairs. Ground-truth evaluation is performed using controlled synthetic colorways.
2. **Synthetic vs Real Dyes:** While synthetic colorways perturb chromatic coordinates and dye values, they do not simulate physical chemical variations such as metallic sheen changes, uneven dye penetration, or variable weave shrinkage.
3. **Exploratory Nature of Track B:** Matches found between independent original photographs (e.g. `deeplure_100` and `deeplure_107`, similarity = $0.9422$) reflect strong structural and weave visual affinity, but cannot be guaranteed to originate from the same weaver master.
4. **Photography Bias:** The model is optimized for catalog/flat product photographs with relatively even lighting. Extreme fold occlusions, perspective warping, or low-resolution social media captures may degrade motif boundary recognition.
""")

    # =========================================================================
    # 20. SUBMISSION CHECKLIST
    # =========================================================================
    add_md("""## 19. DeepLure Submission Checklist (13/13 Requirements PASS)

| ID | Requirement Area | DeepLure JD Specification | Implementation Status | Audit Result |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Approach Note** | Concise summary of approach (strictly <= 500 characters) | Programmatically verified at 471 characters | **PASS** |
| 2 | **Architecture** | Pretrained feature extractor + pooling + projection head | Frozen ConvNeXt-Tiny + learnable GeM + 2-layer MLP | **PASS** |
| 3 | **Normalization** | Exact unit vector output for cosine similarity | Exact L2 normalization ($D=256$, $\\|z\\|=1.000000$) | **PASS** |
| 4 | **Loss Function** | Metric-learning objective | Supervised Contrastive Loss ($\\tau=0.07$) | **PASS** |
| 5 | **Batch Sampler** | Mini-batch balancing positives and negatives | `SourceAwareBatchSampler` ($P=8$, $K=4$, $B=32$) | **PASS** |
| 6 | **Training Code** | Complete reproducible training pipeline | [`scripts/train.py`](scripts/train.py) with early stopping & logging | **PASS** |
| 7 | **Inference Code** | Clean inference module for embedding extraction | [`src/inference.py`](src/inference.py) with full parameter freezing | **PASS** |
| 8 | **Retrieval Code** | Gallery building & Top-K query retrieval | [`src/retrieval.py`](src/retrieval.py) & [`scripts/query_gallery.py`](scripts/query_gallery.py) | **PASS** |
| 9 | **Verification Code**| Pairwise verification with threshold decision | [`scripts/verify_pair.py`](scripts/verify_pair.py) with frozen threshold $\\theta=0.7588$ | **PASS** |
| 10 | **Data Splits** | Leakage-free source-level partition | 100 Train / 25 Val / 40 Test with 0 source leakage | **PASS** |
| 11 | **Test Evaluation** | Held-out test identification & verification metrics | Track A R@1=97.5%, mAP=98.75%, ROC-AUC=0.9967 | **PASS** |
| 12 | **Threshold Integrity**| Decision thresholds calibrated strictly on validation data | Calibrated on Val split at EER; zero test tuning | **PASS** |
| 13 | **Data Privacy** | Exclude confidential customer images and weights | Comprehensive `.gitignore` protecting client assets | **PASS** |
""")

    # =========================================================================
    # 21. CONCLUSION
    # =========================================================================
    add_md("""## 20. Conclusion

MotifWeave delivers a technically sound, empirically validated, and production-ready solution for color-invariant saree surface-design retrieval. By freezing 98.14% of a modern ConvNeXt backbone and fine-tuning a learnable GeM pooling layer and projection head using Supervised Contrastive Loss, the model achieves **97.50% Recall@1**, **98.75% mAP**, and **0.9967 ROC-AUC** while maintaining sub-millisecond search latencies and strict data governance.

---
*MotifWeave Technical Submission Artifact | Prepared for DeepLure (`CIN U62011AP2025OPC118696`)*
""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5,
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    output_path = PROJECT_ROOT / "MotifWeave_Submission.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1)
        
    print(f"Successfully generated and executed notebook with {len(cells)} cells at: {output_path}")

if __name__ == "__main__":
    build_and_execute_notebook()
