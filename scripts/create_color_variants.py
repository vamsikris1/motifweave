import os
import colorsys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def rgb_to_hsv_np(rgb_arr):
    # rgb_arr: float32 in [0, 1], shape (H, W, 3)
    r = rgb_arr[:, :, 0]
    g = rgb_arr[:, :, 1]
    b = rgb_arr[:, :, 2]
    
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    
    deltac = maxc - minc
    s = np.zeros_like(v)
    mask = maxc > 1e-6
    s[mask] = deltac[mask] / maxc[mask]
    
    # Hue calculation
    h = np.zeros_like(v)
    rc = (maxc - r) / (deltac + 1e-6)
    gc = (maxc - g) / (deltac + 1e-6)
    bc = (maxc - b) / (deltac + 1e-6)
    
    mask_r = (r == maxc) & mask
    mask_g = (g == maxc) & mask
    mask_b = (b == maxc) & mask
    
    h[mask_r] = bc[mask_r] - gc[mask_r]
    h[mask_g] = 2.0 + rc[mask_g] - bc[mask_g]
    h[mask_b] = 4.0 + gc[mask_b] - rc[mask_b]
    
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)

def hsv_to_rgb_np(hsv_arr):
    # hsv_arr: float32, H in [0, 1], S in [0, 1], V in [0, 1]
    h = hsv_arr[:, :, 0]
    s = hsv_arr[:, :, 1]
    v = hsv_arr[:, :, 2]
    
    i = np.floor(h * 6.0).astype(np.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    i_mod = i % 6
    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)
    
    m0 = (i_mod == 0)
    r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
    
    m1 = (i_mod == 1)
    r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
    
    m2 = (i_mod == 2)
    r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
    
    m3 = (i_mod == 3)
    r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
    
    m4 = (i_mod == 4)
    r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
    
    m5 = (i_mod == 5)
    r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]
    
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)

def generate_colorway_1(img_pil):
    """
    Variant 1: Complementary / Triadic Hue Shift (+120 degrees / +0.33)
    Rotates primary color palette while preserving exact luminance, zari highlights, and weave edges.
    """
    arr = np.array(img_pil.convert('RGB'), dtype=np.float32) / 255.0
    hsv = rgb_to_hsv_np(arr)
    hsv[:, :, 0] = (hsv[:, :, 0] + 0.333) % 1.0
    # Mild saturation calibration
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.02, 0.0, 1.0)
    rgb = hsv_to_rgb_np(hsv)
    return Image.fromarray((rgb * 255.0).astype(np.uint8))

def generate_colorway_2(img_pil):
    """
    Variant 2: Secondary Triadic Hue Shift (+240 degrees / +0.67)
    Creates an orthogonal textile palette (e.g., Ruby Red -> Emerald Green -> Sapphire Blue).
    """
    arr = np.array(img_pil.convert('RGB'), dtype=np.float32) / 255.0
    hsv = rgb_to_hsv_np(arr)
    hsv[:, :, 0] = (hsv[:, :, 0] + 0.667) % 1.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.98, 0.0, 1.0)
    rgb = hsv_to_rgb_np(hsv)
    return Image.fromarray((rgb * 255.0).astype(np.uint8))

def generate_colorway_3(img_pil):
    """
    Variant 3: Realistic Antique Dye / Jewel Tone Shift
    Combines warm/cool dye shift (+50 degrees / +0.14) with saturation intensification (+20%)
    and subtle contrast curve, simulating authentic batch dye variations on silk.
    """
    arr = np.array(img_pil.convert('RGB'), dtype=np.float32) / 255.0
    hsv = rgb_to_hsv_np(arr)
    hsv[:, :, 0] = (hsv[:, :, 0] + 0.14) % 1.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.20, 0.0, 1.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] ** 0.95, 0.0, 1.0)
    rgb = hsv_to_rgb_np(hsv)
    return Image.fromarray((rgb * 255.0).astype(np.uint8))

def process_and_generate_variants(base_manifest_path, variants_dir, examples_dir, final_manifest_path):
    base_df = pd.read_csv(base_manifest_path)
    variants_dir = Path(variants_dir)
    examples_dir = Path(examples_dir)
    variants_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating controlled colorway variants for {len(base_df)} source images...")
    
    all_records = []
    
    for idx, row in base_df.iterrows():
        source_id = row['source_id']
        orig_path = Path(row['image_path'])
        split = row['split']
        
        # 1. Keep original record
        orig_record = row.to_dict()
        all_records.append(orig_record)
        
        # Load image
        with Image.open(orig_path) as img:
            img_rgb = img.convert('RGB')
            w, h = img_rgb.size
            
            # Generate 3 variants
            v1 = generate_colorway_1(img_rgb)
            v2 = generate_colorway_2(img_rgb)
            v3 = generate_colorway_3(img_rgb)
            
            v1_path = variants_dir / f"{source_id}_variant1.jpg"
            v2_path = variants_dir / f"{source_id}_variant2.jpg"
            v3_path = variants_dir / f"{source_id}_variant3.jpg"
            
            v1.save(v1_path, "JPEG", quality=95)
            v2.save(v2_path, "JPEG", quality=95)
            v3.save(v3_path, "JPEG", quality=95)
            
            # Add records (STRICT LEAKAGE CONTROL: Split matches source_id exactly!)
            all_records.append({
                'image_path': str(v1_path.resolve()),
                'dataset': 'deeplure_saree_corpus',
                'source_id': source_id,
                'original_filename': f"{source_id}_variant1.jpg",
                'identity_type': 'source_image_proxy',
                'category': 'handloom_sarees',
                'split': split,  # SAME SPLIT
                'width': w,
                'height': h,
                'format': 'JPEG',
                'mode': 'RGB',
                'quality_flag': 'pass',
                'variant_type': 'color_variant_1',
                'is_synthetic': True,
                'transform_params': 'HSV_Hue_+120deg_Sat_x1.02'
            })
            
            all_records.append({
                'image_path': str(v2_path.resolve()),
                'dataset': 'deeplure_saree_corpus',
                'source_id': source_id,
                'original_filename': f"{source_id}_variant2.jpg",
                'identity_type': 'source_image_proxy',
                'category': 'handloom_sarees',
                'split': split,  # SAME SPLIT
                'width': w,
                'height': h,
                'format': 'JPEG',
                'mode': 'RGB',
                'quality_flag': 'pass',
                'variant_type': 'color_variant_2',
                'is_synthetic': True,
                'transform_params': 'HSV_Hue_+240deg_Sat_x0.98'
            })
            
            all_records.append({
                'image_path': str(v3_path.resolve()),
                'dataset': 'deeplure_saree_corpus',
                'source_id': source_id,
                'original_filename': f"{source_id}_variant3.jpg",
                'identity_type': 'source_image_proxy',
                'category': 'handloom_sarees',
                'split': split,  # SAME SPLIT
                'width': w,
                'height': h,
                'format': 'JPEG',
                'mode': 'RGB',
                'quality_flag': 'pass',
                'variant_type': 'color_variant_3',
                'is_synthetic': True,
                'transform_params': 'HSV_Hue_+50deg_Sat_x1.20_ValGamma_0.95'
            })
            
            # Save visual comparison for the first 25 examples
            if idx < 25:
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(img_rgb)
                axes[0].set_title(f"Original ({source_id})\n[{split.upper()}]", fontsize=11, fontweight='bold')
                axes[0].axis('off')
                
                axes[1].imshow(v1)
                axes[1].set_title("Color Variant 1\n(Hue +120° Triadic)", fontsize=11)
                axes[1].axis('off')
                
                axes[2].imshow(v2)
                axes[2].set_title("Color Variant 2\n(Hue +240° Triadic)", fontsize=11)
                axes[2].axis('off')
                
                axes[3].imshow(v3)
                axes[3].set_title("Color Variant 3\n(Dye Shift +50°, Sat +20%)", fontsize=11)
                axes[3].axis('off')
                
                plt.tight_layout()
                example_fig_path = examples_dir / f"example_{idx+1:02d}_{source_id}.jpg"
                plt.savefig(example_fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
    final_df = pd.DataFrame(all_records)
    final_df.to_csv(final_manifest_path, index=False)
    
    # Also save to reports directory for convenient reference
    reports_manifest = Path(r"\reports\manifest.csv")
    final_df.to_csv(reports_manifest, index=False)
    
    # Save split files
    splits_dir = Path(r"\data\processed\splits")
    for s_name in ['train', 'val', 'test']:
        s_df = final_df[final_df['split'] == s_name]
        s_df.to_csv(splits_dir / f"{s_name}.csv", index=False)
        print(f"Split '{s_name}' saved: {len(s_df)} images ({len(s_df['source_id'].unique())} unique sources)")
        
    print(f"\nFinal manifest saved: {len(final_df)} total records at {final_manifest_path}")
    print(f"Visualization examples saved to: {examples_dir.resolve()}")
    return final_df

if __name__ == '__main__':
    base_m = r"\data\processed\base_manifest.csv"
    var_d = r"\data\processed\variants"
    ex_d = r"\reports\color_augmentation_examples"
    fin_m = r"\data\processed\manifest.csv"
    process_and_generate_variants(base_m, var_d, ex_d, fin_m)
