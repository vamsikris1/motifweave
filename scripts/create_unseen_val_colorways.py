import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.create_color_variants import rgb_to_hsv_np, hsv_to_rgb_np

def generate_unseen_colorway_1(img_pil):
    """
    Unseen Variant 1: Sage / Mint / Teal Dye Shift (Intermediate hue)
    Hue shift: +85 degrees (+0.2361)
    Saturation: x0.85 (soft pastel desaturation)
    Value: x1.05 (subtle brightness lift)
    """
    arr = np.array(img_pil.convert('RGB'), dtype=np.float32) / 255.0
    hsv = rgb_to_hsv_np(arr)
    hsv[:, :, 0] = (hsv[:, :, 0] + (85.0 / 360.0)) % 1.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.85, 0.0, 1.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0.0, 1.0)
    rgb = hsv_to_rgb_np(hsv)
    return Image.fromarray((rgb * 255.0).astype(np.uint8))

def generate_unseen_colorway_2(img_pil):
    """
    Unseen Variant 2: Royal Indigo / Deep Complementary Inversion
    Hue shift: +180 degrees (+0.5000)
    Saturation: x1.10 (+10% saturation)
    Value: gamma 1.06 (mild shadow contrast deepening)
    """
    arr = np.array(img_pil.convert('RGB'), dtype=np.float32) / 255.0
    hsv = rgb_to_hsv_np(arr)
    hsv[:, :, 0] = (hsv[:, :, 0] + (180.0 / 360.0)) % 1.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0.0, 1.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] ** 1.06, 0.0, 1.0)
    rgb = hsv_to_rgb_np(hsv)
    return Image.fromarray((rgb * 255.0).astype(np.uint8))

def generate_unseen_colorway_3(img_pil):
    """
    Unseen Variant 3: Antique Amber / Terracotta Shift (Negative hue angle)
    Hue shift: -45 / +315 degrees (+0.8750)
    Saturation: x0.92 (-8% saturation)
    Value: x0.92 (-8% brightness, simulating aged unbleached ground)
    """
    arr = np.array(img_pil.convert('RGB'), dtype=np.float32) / 255.0
    hsv = rgb_to_hsv_np(arr)
    hsv[:, :, 0] = (hsv[:, :, 0] + (315.0 / 360.0)) % 1.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.92, 0.0, 1.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.92, 0.0, 1.0)
    rgb = hsv_to_rgb_np(hsv)
    return Image.fromarray((rgb * 255.0).astype(np.uint8))

def build_unseen_validation_benchmark():
    val_csv = PROJECT_ROOT / "data/processed/splits/val.csv"
    val_df = pd.read_csv(val_csv)
    
    # Filter strictly to original source images in validation split
    orig_val_df = val_df[val_df['variant_type'] == 'original'].copy()
    assert len(orig_val_df) == 25, f"Expected 25 validation sources, found {len(orig_val_df)}"
    print(f"Loaded {len(orig_val_df)} original validation source images.")
    
    output_dir = PROJECT_ROOT / "data/processed/unseen_val_variants"
    examples_dir = PROJECT_ROOT / "reports/unseen_colorway_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    for idx, (_, row) in enumerate(orig_val_df.iterrows()):
        source_id = row['source_id']
        orig_img_path = Path(row['image_path'])
        
        with Image.open(orig_img_path) as img:
            img_rgb = img.convert('RGB')
            w, h = img_rgb.size
            
            u1 = generate_unseen_colorway_1(img_rgb)
            u2 = generate_unseen_colorway_2(img_rgb)
            u3 = generate_unseen_colorway_3(img_rgb)
            
            u1_path = output_dir / f"{source_id}_unseen_variant1.jpg"
            u2_path = output_dir / f"{source_id}_unseen_variant2.jpg"
            u3_path = output_dir / f"{source_id}_unseen_variant3.jpg"
            
            u1.save(u1_path, "JPEG", quality=95)
            u2.save(u2_path, "JPEG", quality=95)
            u3.save(u3_path, "JPEG", quality=95)
            
            records.append({
                'image_path': str(u1_path.resolve()),
                'dataset': 'deeplure_saree_corpus',
                'source_id': source_id,
                'original_filename': f"{source_id}_unseen_variant1.jpg",
                'identity_type': 'source_image_proxy',
                'category': 'handloom_sarees',
                'split': 'val',
                'width': w,
                'height': h,
                'format': 'JPEG',
                'mode': 'RGB',
                'quality_flag': 'pass',
                'variant_type': 'unseen_color_variant_1',
                'is_synthetic': True,
                'transform_name': 'Sage_Mint_Teal',
                'transform_params': 'HSV_Hue_+85deg_Sat_x0.85_Val_x1.05'
            })
            
            records.append({
                'image_path': str(u2_path.resolve()),
                'dataset': 'deeplure_saree_corpus',
                'source_id': source_id,
                'original_filename': f"{source_id}_unseen_variant2.jpg",
                'identity_type': 'source_image_proxy',
                'category': 'handloom_sarees',
                'split': 'val',
                'width': w,
                'height': h,
                'format': 'JPEG',
                'mode': 'RGB',
                'quality_flag': 'pass',
                'variant_type': 'unseen_color_variant_2',
                'is_synthetic': True,
                'transform_name': 'Royal_Indigo_Complementary',
                'transform_params': 'HSV_Hue_+180deg_Sat_x1.10_ValGamma_1.06'
            })
            
            records.append({
                'image_path': str(u3_path.resolve()),
                'dataset': 'deeplure_saree_corpus',
                'source_id': source_id,
                'original_filename': f"{source_id}_unseen_variant3.jpg",
                'identity_type': 'source_image_proxy',
                'category': 'handloom_sarees',
                'split': 'val',
                'width': w,
                'height': h,
                'format': 'JPEG',
                'mode': 'RGB',
                'quality_flag': 'pass',
                'variant_type': 'unseen_color_variant_3',
                'is_synthetic': True,
                'transform_name': 'Antique_Amber_Terracotta',
                'transform_params': 'HSV_Hue_+315deg_Sat_x0.92_Val_x0.92'
            })
            
            # Save visual comparison figure for all 25 sources
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img_rgb)
            axes[0].set_title(f"Original Source: {source_id}\n[VAL GALLERY]", fontsize=10, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(u1)
            axes[1].set_title("Unseen Variant 1\n(Hue +85°, Sat x0.85, Val x1.05)", fontsize=10)
            axes[1].axis('off')
            
            axes[2].imshow(u2)
            axes[2].set_title("Unseen Variant 2\n(Hue +180° Inversion, Sat x1.10)", fontsize=10)
            axes[2].axis('off')
            
            axes[3].imshow(u3)
            axes[3].set_title("Unseen Variant 3\n(Hue -45°/315°, Sat x0.92, Val x0.92)", fontsize=10)
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(examples_dir / f"unseen_val_example_{idx+1:02d}_{source_id}.jpg", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
    unseen_df = pd.DataFrame(records)
    manifest_path = PROJECT_ROOT / "data/processed/unseen_val_manifest.csv"
    unseen_df.to_csv(manifest_path, index=False)
    print(f"Generated {len(unseen_df)} unseen validation queries (25 sources x 3 variants).")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Visual validation examples saved to: {examples_dir}")
    return unseen_df

if __name__ == '__main__':
    build_unseen_validation_benchmark()
