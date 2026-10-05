# Phase 2: Controlled Color Transformation Report
**Project:** MotifWeave — Color-Invariant Saree Design Recognition  
**Client / Specification:** DeepLure AIE-CASE Brief  
**Date:** 2026-10-05  
**Lead ML Engineer:** Antigravity  

---

## 1. Objective of the Controlled Color Transformation Pipeline

Because the supplied DeepLure corpus lacks design-level identity annotations and naturally occurring multi-colorway pairs, we implement a **controlled synthetic colorway pipeline**. 

The goal is to generate controlled synthetic colorway variants from known source images to rigorously benchmark whether an embedding model can recognize identical textile design motifs when rendered across altered color palettes.

The transformation preserves spatial image structure and motif locations while modifying chromatic appearance.

---

## 2. Selected Transformations & Implementation Details

All transformations operate deterministically on RGB arrays mapped to HSV color space:

| Variant Name | Mathematical Specification | Color Space | Transformation Characteristics |
| :--- | :--- | :--- | :--- |
| **Original** | Identity: $X$ | RGB | Base authentic e-commerce catalog image. |
| **Color Variant 1** | $H \leftarrow (H + 0.333) \bmod 1.0$<br>$S \leftarrow \text{clip}(S \times 1.02, 0, 1)$ | HSV | **Controlled Palette Perturbation (+120°):** Rotates the primary hue palette by $120^\circ$ on the normalized color wheel. Modifies chromatic coordinates while preserving spatial image structure and motif locations. |
| **Color Variant 2** | $H \leftarrow (H + 0.667) \bmod 1.0$<br>$S \leftarrow \text{clip}(S \times 0.98, 0, 1)$ | HSV | **Controlled Palette Perturbation (+240°):** Shifts hue by an orthogonal offset ($240^\circ$), generating an alternate chromatic appearance from the same source image. |
| **Color Variant 3** | $H \leftarrow (H + 0.140) \bmod 1.0$<br>$S \leftarrow \text{clip}(S \times 1.20, 0, 1)$<br>$V \leftarrow \text{clip}(V^{0.95}, 0, 1)$ | HSV | **Chromatic & Contrast Perturbation (+50°, +20% Sat):** Combines a $+50^\circ$ hue shift with a $+20\%$ saturation scaling and subtle power-law value curve ($V^{0.95}$), testing robustness to saturation and intensity variations. |

---

## 3. Spatial Structure Preservation & Transformation Scope

### Supported Properties:
- **Spatial Alignment:** Pixel coordinate $(x, y)$ in any synthetic variant directly corresponds to pixel $(x, y)$ in the original image.
- **Edge & Texture Geometry:** Fine structural boundaries (borders, zari brocade outlines, butta patterns) are geometrically identical between the source image and its variants because no spatial resampling, warping, or blurring is introduced.
- **Controlled Scope:** These images are explicitly designated as **"controlled synthetic colorway variants"**. They are not claimed to be physically realistic alternate photographs or newly dyed fabrics.

---

## 4. Visual Inspection of Generated Colorways

A total of 25 side-by-side comparison grids have been saved under:
[`reports/color_augmentation_examples/`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/reports/color_augmentation_examples/)

Each grid displays:
`ORIGINAL | COLOR VARIANT 1 | COLOR VARIANT 2 | COLOR VARIANT 3`

- [`example_01_deeplure_001.jpg`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/reports/color_augmentation_examples/example_01_deeplure_001.jpg)
- [`example_02_deeplure_002.jpg`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/reports/color_augmentation_examples/example_02_deeplure_002.jpg)
- [`example_03_deeplure_003.jpg`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/reports/color_augmentation_examples/example_03_deeplure_003.jpg)
- [`example_04_deeplure_004.jpg`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/reports/color_augmentation_examples/example_04_deeplure_004.jpg)
- [`example_05_deeplure_005.jpg`](file:///C:/Users/PandraVamsi/.gemini/antigravity/scratch/deep-lure-motifweave/reports/color_augmentation_examples/example_05_deeplure_005.jpg)
*(... through `example_25_deeplure_025.jpg`)*

Inspection validates that the transformation preserves spatial image structure and motif locations while modifying chromatic appearance.
