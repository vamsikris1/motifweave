import os
from pathlib import Path
import pandas as pd
import numpy as np

def create_deterministic_splits(base_manifest_path, splits_dir, seed=42):
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(base_manifest_path)
    total_sources = len(df)
    print(f"Assigning deterministic splits for {total_sources} unique source images (Seed={seed})...")
    
    # Unique source IDs
    source_ids = df['source_id'].unique()
    
    # Fix deterministic random state
    rng = np.random.RandomState(seed)
    shuffled_sources = rng.permutation(source_ids)
    
    # Justified split sizes:
    # Train: 100 (60.6%) - sufficient diversity for metric learning
    # Val:    25 (15.2%) - for validation loss & threshold tuning (4,950 candidate pairs)
    # Test:   40 (24.2%) - strictly held-out test gallery & query benchmark
    n_train = 100
    n_val = 25
    n_test = 40
    assert n_train + n_val + n_test == total_sources, "Split counts must sum to total sources"
    
    train_sources = shuffled_sources[:n_train]
    val_sources = shuffled_sources[n_train:n_train + n_val]
    test_sources = shuffled_sources[n_train + n_val:]
    
    # Verification of zero leakage
    train_set = set(train_sources)
    val_set = set(val_sources)
    test_set = set(test_sources)
    
    assert len(train_set.intersection(val_set)) == 0, "LEAKAGE ERROR: Train and Val overlap!"
    assert len(train_set.intersection(test_set)) == 0, "LEAKAGE ERROR: Train and Test overlap!"
    assert len(val_set.intersection(test_set)) == 0, "LEAKAGE ERROR: Val and Test overlap!"
    assert len(train_set) + len(val_set) + len(test_set) == total_sources, "Source count mismatch!"
    
    print("Leakage Verification: PASSED (Zero source overlap between Train, Val, and Test).")
    print(f"  - Train sources: {len(train_sources)} ({len(train_sources)/total_sources*100:.1f}%)")
    print(f"  - Val sources:   {len(val_sources)} ({len(val_sources)/total_sources*100:.1f}%)")
    print(f"  - Test sources:  {len(test_sources)} ({len(test_sources)/total_sources*100:.1f}%)")
    
    # Assign split to dataframe
    split_map = {}
    for s in train_sources:
        split_map[s] = 'train'
    for s in val_sources:
        split_map[s] = 'val'
    for s in test_sources:
        split_map[s] = 'test'
        
    df['split'] = df['source_id'].map(split_map)
    df.to_csv(base_manifest_path, index=False)
    
    # Save individual split source lists
    pd.DataFrame({'source_id': train_sources}).to_csv(splits_dir / "train_sources.csv", index=False)
    pd.DataFrame({'source_id': val_sources}).to_csv(splits_dir / "val_sources.csv", index=False)
    pd.DataFrame({'source_id': test_sources}).to_csv(splits_dir / "test_sources.csv", index=False)
    
    print(f"Split source lists saved to: {splits_dir.resolve()}")
    return df, train_sources, val_sources, test_sources

if __name__ == '__main__':
    base_manifest = r"\data\processed\base_manifest.csv"
    splits_output = r"\data\processed\splits"
    create_deterministic_splits(base_manifest, splits_output, seed=42)
