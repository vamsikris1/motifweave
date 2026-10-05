# Approach Note: MotifWeave
**Project:** Color-Invariant Saree Design Recognition  
**Client / Specification:** DeepLure AIE-CASE Brief (`CIN U62011AP2025OPC118696`)  
**Lead ML Engineer:** Antigravity  

---

### Authoritative Approach Note (471 Characters):

```text
MotifWeave recognizes saree designs invariantly to color palettes. Architecture: frozen ConvNeXt-Tiny extracts visual textures, pooled via learnable GeM (p=3.0) and mapped by a 2-layer MLP to a 256-D L2-normalized hypersphere. Preprocessing: 224x224 bicubic resize with ImageNet normalization. Strategy: source-aware PxK batch sampling (P=8, K=4) trained with Supervised Contrastive Loss (tau=0.07) and AdamW across controlled synthetic colorway shifts and spatial flips.
```

### Character Accounting Breakdown:
- **Total Character Count (including spaces):** Exactly 471 characters (Strictly below the 500-character JD limit).
- **Core Elements Covered:**
  1. *Architecture:* ConvNeXt-Tiny backbone + learnable GeM pooling + 2-layer MLP projection head (256-D $L_2$ normalized).
  2. *Rationale:* Decoupling surface weave and motif geometry from chromatic variations.
  3. *Preprocessing:* $224 \times 224$ bicubic resize with standard ImageNet normalization.
  4. *Training Strategy & Sampling:* Source-aware $P \times K$ batch sampling ($P=8$ sources, $K=4$ variants/source) with AdamW optimizer.
  5. *Loss Function:* Supervised Contrastive Loss ($\tau = 0.07$) with self-contrast masking.
  6. *Augmentation:* Controlled synthetic colorway transformations and random horizontal flips.
