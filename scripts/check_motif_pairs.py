import os
import glob
import numpy as np
from PIL import Image

def analyze_structural_similarity(img_dir):
    files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"Analyzing {len(files)} images for potential shared design motifs...")
    
    # Load and convert to small normalized grayscale edge/gradient maps
    gray_thumbnails = []
    names = []
    for f in files:
        with Image.open(f) as img:
            # Resize to standardized resolution
            g = img.convert('L').resize((128, 128), Image.Resampling.LANCZOS)
            arr = np.array(g, dtype=np.float32)
            # Normalize zero mean, unit variance
            arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-6)
            gray_thumbnails.append(arr)
            names.append(os.path.basename(f))
            
    # Compute normalized cross-correlation across all pairs
    top_correlations = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            corr = np.mean(gray_thumbnails[i] * gray_thumbnails[j])
            top_correlations.append((corr, names[i], names[j]))
            
    top_correlations.sort(reverse=True)
    print("\nTop 15 structurally most correlated image pairs (Grayscale Normalized Cross-Correlation):")
    for corr, f1, f2 in top_correlations[:15]:
        print(f"  Corr = {corr:0.4f}: {f1} <---> {f2}")
        
    print("\nLowest 5 correlated image pairs:")
    for corr, f1, f2 in top_correlations[-5:]:
        print(f"  Corr = {corr:0.4f}: {f1} <---> {f2}")

if __name__ == '__main__':
    data_path = r"C:\Users\PandraVamsi\.gemini\antigravity\scratch\deep-lure-motifweave\data\raw\deeplure_corpus\sarees_dataset\handloom_sarees"
    analyze_structural_similarity(data_path)
