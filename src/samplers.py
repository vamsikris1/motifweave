import math
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Sampler

class SourceAwareBatchSampler(Sampler):
    """
    Batch sampler that ensures each batch contains P distinct source IDs,
    with K variants per source ID (Batch Size = P * K).
    
    This guarantees that every anchor has (K - 1) positive pairs and
    (P - 1) * K negative pairs within the same mini-batch.
    """
    def __init__(self, dataset, p_sources=8, k_variants=4, seed=42):
        self.dataset = dataset
        self.p_sources = p_sources
        self.k_variants = k_variants
        self.batch_size = p_sources * k_variants
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Group dataset indices by source_label
        self.source_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            row = dataset.df.iloc[idx]
            label = row['source_label']
            self.source_to_indices[label].append(idx)
            
        self.unique_sources = list(self.source_to_indices.keys())
        self.num_sources = len(self.unique_sources)
        
        # Calculate number of batches per epoch
        # Total images / batch_size
        self.num_batches = max(1, len(dataset) // self.batch_size)
        
    def __iter__(self):
        # Deterministically shuffle for each epoch
        current_sources = self.unique_sources.copy()
        self.rng.shuffle(current_sources)
        
        for _ in range(self.num_batches):
            # Select P sources
            if len(current_sources) < self.p_sources:
                current_sources = self.unique_sources.copy()
                self.rng.shuffle(current_sources)
                
            selected_sources = [current_sources.pop() for _ in range(self.p_sources)]
            
            batch_indices = []
            for src in selected_sources:
                available_indices = self.source_to_indices[src]
                # Sample K variants from this source (with replacement if len < K, without if len >= K)
                replace = len(available_indices) < self.k_variants
                chosen = self.rng.choice(available_indices, size=self.k_variants, replace=replace)
                batch_indices.extend(chosen.tolist())
                
            yield batch_indices

    def __len__(self):
        return self.num_batches
