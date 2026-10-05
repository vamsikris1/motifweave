import os
import glob
from pathlib import Path
import pandas as pd
from PIL import Image

def prepare_base_manifest(raw_dir, output_manifest_path):
    raw_dir = Path(raw_dir)
    print(f"Reading raw images from: {raw_dir.resolve()}")
    
    # Strictly preserve original files - only read them
    files = sorted([f for f in raw_dir.glob("*.jpg")])
    print(f"Found {len(files)} raw images.")
    
    records = []
    for idx, f in enumerate(files, 1):
        source_id = f"deeplure_{idx:03d}"
        with Image.open(f) as img:
            w, h = img.size
            fmt = img.format
            mode = img.mode
            
        record = {
            'image_path': str(f.resolve()),
            'dataset': 'deeplure_saree_corpus',
            'source_id': source_id,
            'original_filename': f.name,
            'identity_type': 'source_image_proxy',  # Explicitly documented proxy, NOT fabricated design SKU
            'category': 'handloom_sarees',          # Coarse craft category
            'split': 'unassigned',                  # Will be populated deterministically by create_splits.py
            'width': w,
            'height': h,
            'format': fmt,
            'mode': mode,
            'quality_flag': 'pass',
            'variant_type': 'original',
            'is_synthetic': False,
            'transform_params': 'none'
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
    df.to_csv(output_manifest_path, index=False)
    print(f"Base manifest created with {len(df)} records at: {output_manifest_path}")
    return df

if __name__ == '__main__':
    raw_path = r"\data\raw\deeplure_corpus\sarees_dataset\handloom_sarees"
    out_manifest = r"\data\processed\base_manifest.csv"
    prepare_base_manifest(raw_path, out_manifest)
