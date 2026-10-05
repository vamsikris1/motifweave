import os
import random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint

def compute_pairwise_validation_statistics(embeddings, labels):
    """
    Computes pairwise cosine similarity statistics on the validation set.
    Excludes trivial self-matches (similarity with itself).
    
    Args:
        embeddings: Tensor of shape (N, D), L2-normalized.
        labels: Tensor of shape (N,) containing source_id labels.
    Returns:
        dict containing:
            - mean_pos_sim: mean similarity between same-source pairs (excluding self)
            - mean_neg_sim: mean similarity between source-distinct pairs
            - sim_margin: mean_pos_sim - mean_neg_sim
            - std_pos_sim: standard deviation of positive similarities
            - std_neg_sim: standard deviation of negative similarities
    """
    # Ensure numpy arrays
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        
    n = len(labels)
    # Cosine similarity matrix (embeddings are L2 normalized)
    sim_matrix = np.dot(embeddings, embeddings.T)
    
    pos_sims = []
    neg_sims = []
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if labels[i] == labels[j]:
                pos_sims.append(sim)
            else:
                neg_sims.append(sim)
                
    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)
    
    mean_pos = float(np.mean(pos_sims)) if len(pos_sims) > 0 else 0.0
    mean_neg = float(np.mean(neg_sims)) if len(neg_sims) > 0 else 0.0
    
    return {
        'num_pos_pairs': len(pos_sims),
        'num_neg_pairs': len(neg_sims),
        'mean_pos_sim': round(mean_pos, 4),
        'mean_neg_sim': round(mean_neg, 4),
        'sim_margin': round(mean_pos - mean_neg, 4),
        'std_pos_sim': round(float(np.std(pos_sims)), 4) if len(pos_sims) > 0 else 0.0,
        'std_neg_sim': round(float(np.std(neg_sims)), 4) if len(neg_sims) > 0 else 0.0,
        'min_pos_sim': round(float(np.min(pos_sims)), 4) if len(pos_sims) > 0 else 0.0,
        'max_neg_sim': round(float(np.max(neg_sims)), 4) if len(neg_sims) > 0 else 0.0,
    }

def plot_pca_embeddings(embeddings, labels, output_path, title="Validation Embeddings PCA (Sanity Check)"):
    """
    2D PCA visualization of validation embeddings to inspect source grouping.
    Uses SVD in numpy.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        
    # Center embeddings
    centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    # SVD
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T  # Project to first 2 principal components
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab20', alpha=0.85, s=60, edgecolors='black', linewidth=0.5)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel(f"PC 1 ({s[0]**2 / np.sum(s**2)*100:.1f}% variance)")
    plt.ylabel(f"PC 2 ({s[1]**2 / np.sum(s**2)*100:.1f}% variance)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.colorbar(scatter, label="Source ID Index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"PCA visualization saved to: {output_path}")
