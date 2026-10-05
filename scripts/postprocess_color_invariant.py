import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import SareeDataset, get_baseline_transforms
from src.model import build_model
from src.utils import compute_pairwise_validation_statistics, plot_pca_embeddings
from scripts.evaluate_val_retrieval import evaluate_controlled_val_retrieval

def postprocess():
    best_ckpt_path = PROJECT_ROOT / "outputs/checkpoints/color_invariant_best.pth"
    last_ckpt_path = PROJECT_ROOT / "outputs/checkpoints/color_invariant_last.pth"
    history_path = PROJECT_ROOT / "outputs/metrics/color_invariant_history.json"
    pca_fig_path = PROJECT_ROOT / "outputs/figures/color_invariant_val_pca.png"
    val_csv_path = PROJECT_ROOT / "data/processed/splits/val.csv"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(best_ckpt_path, map_location=device)
    config = ckpt['config']
    history = ckpt['history']
    
    # 1. Save history JSON
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to: {history_path}")
    
    # 2. Duplicate best checkpoint to last checkpoint
    if last_ckpt_path.exists():
        os.remove(last_ckpt_path)
    shutil.copyfile(best_ckpt_path, last_ckpt_path)
    print(f"Last checkpoint created cleanly at: {last_ckpt_path}")
    
    # 3. Extract validation embeddings
    model = build_model(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    val_transform = get_baseline_transforms(image_size=config.get('image_size', 224), is_training=False)
    val_dataset = SareeDataset(val_csv_path, transform=val_transform)
    
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for i in range(len(val_dataset)):
            img, label, _, _ = val_dataset[i]
            emb = model(img.unsqueeze(0).to(device))
            all_embeddings.append(emb.cpu())
            all_labels.append(label)
            
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.tensor(all_labels)
    
    # Check embedding norms
    norms = torch.norm(all_embeddings, p=2, dim=1)
    print(f"Embedding Norm check: Mean={norms.mean().item():.6f}, Std={norms.std().item():.6f}")
    
    # Pairwise stats
    stats = compute_pairwise_validation_statistics(all_embeddings, all_labels)
    stats['val_loss'] = ckpt['val_loss']
    print("\nPairwise Statistics for Experiment B (Validation Only):")
    for k, v in stats.items():
        print(f"  {k}: {v}")
        
    # 4. Generate PCA figure
    plot_pca_embeddings(all_embeddings, all_labels, str(pca_fig_path), title="Color-Invariant ConvNeXt-Tiny Validation Embeddings PCA")
    
    # 5. Evaluate controlled retrieval on Validation
    print("\nEvaluating Controlled Colorway Retrieval on Validation Split:")
    retrieval_metrics = evaluate_controlled_val_retrieval(best_ckpt_path, val_csv_path)
    
    return stats, retrieval_metrics

if __name__ == '__main__':
    postprocess()
