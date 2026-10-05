import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import SareeDataset, get_baseline_transforms
from src.model import build_model
from src.utils import load_checkpoint

def evaluate_controlled_val_retrieval(checkpoint_path, val_csv_path=None):
    """
    Evaluates controlled source-identity retrieval on the VALIDATION split:
    Query: 75 controlled synthetic colorway variants
    Gallery: 25 original source images
    Ground truth match: query source_id == gallery source_id
    """
    if val_csv_path is None:
        val_csv_path = PROJECT_ROOT / "data/processed/splits/val.csv"
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt['config']
    
    model = build_model(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load dataset
    transform = get_baseline_transforms(image_size=config.get('image_size', 224), is_training=False)
    dataset = SareeDataset(val_csv_path, transform=transform)
    df = dataset.df
    
    # Separate gallery and query
    gallery_indices = df[df['variant_type'] == 'original'].index.tolist()
    query_indices = df[df['variant_type'] != 'original'].index.tolist()
    
    assert len(gallery_indices) == 25, f"Expected 25 gallery items, found {len(gallery_indices)}"
    assert len(query_indices) == 75, f"Expected 75 query items, found {len(query_indices)}"
    
    # Extract embeddings
    all_embeddings = []
    with torch.no_grad():
        for i in range(len(dataset)):
            img_tensor, _, _, _ = dataset[i]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            emb = model(img_tensor)
            all_embeddings.append(emb.cpu())
            
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy() # (100, 256)
    
    gallery_embeds = all_embeddings[gallery_indices] # (25, 256)
    gallery_sources = df.iloc[gallery_indices]['source_id'].values
    
    query_embeds = all_embeddings[query_indices] # (75, 256)
    query_sources = df.iloc[query_indices]['source_id'].values
    query_variants = df.iloc[query_indices]['variant_type'].values
    
    # Cosine similarity matrix between queries and gallery (75, 25)
    sim_matrix = np.dot(query_embeds, gallery_embeds.T)
    
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    average_precisions = []
    
    num_queries = len(query_indices)
    
    for q_idx in range(num_queries):
        true_source = query_sources[q_idx]
        sims = sim_matrix[q_idx]
        
        # Rank gallery items in descending similarity
        ranked_indices = np.argsort(-sims)
        ranked_sources = gallery_sources[ranked_indices]
        
        # Match mask
        matches = (ranked_sources == true_source)
        
        # Recall@K
        if np.any(matches[:1]):
            top1_correct += 1
        if np.any(matches[:5]):
            top5_correct += 1
        if np.any(matches[:10]):
            top10_correct += 1
            
        # Average Precision (AP)
        # In this 1-to-1 gallery setting, AP = 1.0 / (rank + 1)
        rank = np.where(matches)[0][0]
        ap = 1.0 / (rank + 1)
        average_precisions.append(ap)
        
    r1 = top1_correct / num_queries
    r5 = top5_correct / num_queries
    r10 = top10_correct / num_queries
    m_ap = float(np.mean(average_precisions))
    
    results = {
        'num_queries': num_queries,
        'num_gallery': len(gallery_indices),
        'recall_at_1': round(r1 * 100, 2),
        'recall_at_5': round(r5 * 100, 2),
        'recall_at_10': round(r10 * 100, 2),
        'mAP': round(m_ap * 100, 2)
    }
    
    print(f"\n--- Controlled Source-Identity Retrieval Metrics (Validation Only) ---")
    print(f"Model Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Queries: {num_queries} (controlled synthetic variants) | Gallery: {len(gallery_indices)} (original sources)")
    print(f"  Recall@1:  {results['recall_at_1']:.2f}%")
    print(f"  Recall@5:  {results['recall_at_5']:.2f}%")
    print(f"  Recall@10: {results['recall_at_10']:.2f}%")
    print(f"  mAP:       {results['mAP']:.2f}%")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    evaluate_controlled_val_retrieval(args.checkpoint)
