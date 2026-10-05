import os
import sys
import hashlib
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

def compute_md5(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(image, hash_size=8):
    # Difference Hash: robust to color changes, captures intensity gradients/structure
    img_gray = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(img_gray)
    # Compare adjacent pixels
    diff = pixels[:, 1:] > pixels[:, :-1]
    return ''.join(['1' if b else '0' for b in diff.flatten()])

def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def audit_directory(data_root):
    data_root = Path(data_root)
    records = []
    
    print(f"Auditing directory: {data_root.resolve()}")
    all_files = list(data_root.rglob('*'))
    file_list = [f for f in all_files if f.is_file()]
    dir_list = [d for d in all_files if d.is_dir()]
    
    print(f"Total directories found: {len(dir_list)}")
    for d in dir_list:
        sub_files = [f for f in d.iterdir() if f.is_file()]
        print(f"  - Directory: {d.relative_to(data_root)} | File count: {len(sub_files)}")
    
    print(f"\nTotal files found: {len(file_list)}")
    
    for f in file_list:
        rel_path = f.relative_to(data_root)
        folder = f.parent.name
        file_size = f.stat().st_size
        record = {
            'file_name': f.name,
            'rel_path': str(rel_path),
            'folder': folder,
            'file_size_bytes': file_size,
            'is_valid_image': False,
            'corrupt_error': None,
            'format': None,
            'mode': None,
            'width': None,
            'height': None,
            'aspect_ratio': None,
            'md5': None,
            'dhash': None,
            'mean_r': None,
            'mean_g': None,
            'mean_b': None,
            'apparent_identity': None,
            'colorway': None,
        }
        
        # Check image validity
        try:
            with Image.open(f) as img:
                img.verify()
            with Image.open(f) as img:
                record['is_valid_image'] = True
                record['format'] = img.format
                record['mode'] = img.mode
                record['width'] = img.width
                record['height'] = img.height
                record['aspect_ratio'] = round(img.width / img.height, 4) if img.height > 0 else 0
                record['md5'] = compute_md5(f)
                record['dhash'] = compute_dhash(img)
                
                # RGB statistics
                rgb_img = img.convert('RGB')
                arr = np.array(rgb_img)
                record['mean_r'] = round(float(np.mean(arr[:, :, 0])), 2)
                record['mean_g'] = round(float(np.mean(arr[:, :, 1])), 2)
                record['mean_b'] = round(float(np.mean(arr[:, :, 2])), 2)
        except Exception as e:
            record['is_valid_image'] = False
            record['corrupt_error'] = str(e)
            record['md5'] = compute_md5(f)
            
        records.append(record)
        
    df = pd.DataFrame(records)
    return df

def analyze_duplicates_and_clusters(df):
    print("\n--- Duplicate Analysis ---")
    valid_df = df[df['is_valid_image'] == True].copy()
    
    # Exact duplicates via MD5
    exact_dups = valid_df[valid_df.duplicated(subset=['md5'], keep=False)]
    print(f"Exact duplicates (identical MD5): {len(exact_dups)}")
    if len(exact_dups) > 0:
        for md5_val, group in exact_dups.groupby('md5'):
            print(f"  MD5 {md5_val}: {list(group['file_name'])}")
    else:
        print("  None. All image files have unique MD5 checksums.")
        
    # Perceptual near-duplicates via dHash (Hamming distance <= 4)
    print("\n--- Near-Duplicate Analysis (Structural dHash) ---")
    hashes = valid_df[['file_name', 'dhash']].to_dict('records')
    near_dups = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            h1 = hashes[i]['dhash']
            h2 = hashes[j]['dhash']
            dist = hamming_distance(h1, h2)
            if dist <= 6:  # very close structural match
                near_dups.append((hashes[i]['file_name'], hashes[j]['file_name'], dist))
                
    print(f"Near-duplicate / structurally close pairs (dHash distance <= 6 out of 64 bits): {len(near_dups)}")
    for f1, f2, d in near_dups[:20]:
        print(f"  {f1} <-> {f2} (distance = {d})")
        
    return exact_dups, near_dups

if __name__ == '__main__':
    data_path = r"C:\Users\PandraVamsi\.gemini\antigravity\scratch\deep-lure-motifweave\data\raw\deeplure_corpus\sarees_dataset"
    df = audit_directory(data_path)
    
    # Save manifest
    output_dir = r"C:\Users\PandraVamsi\.gemini\antigravity\scratch\deep-lure-motifweave\reports"
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "deeplure_manifest.csv")
    df.to_csv(manifest_path, index=False)
    print(f"\nManifest successfully saved to: {manifest_path}")
    
    # Statistical summary
    print("\n================ DATASET AUDIT SUMMARY ================")
    print(f"Total files audited: {len(df)}")
    print(f"Valid images: {df['is_valid_image'].sum()}")
    print(f"Corrupt images: {(~df['is_valid_image']).sum()}")
    print("\nFormat distribution:")
    print(df['format'].value_counts(dropna=False))
    print("\nColor Mode distribution:")
    print(df['mode'].value_counts(dropna=False))
    print("\nImage Dimensions summary:")
    print(df[['width', 'height', 'aspect_ratio']].describe())
    
    analyze_duplicates_and_clusters(df)
