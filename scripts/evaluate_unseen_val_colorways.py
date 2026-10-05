import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model

def get_eval_transform(image_size=224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

def extract_embeddings(model, image_paths, transform, device):
    embeddings = []
    with torch.no_grad():
        for path in image_paths:
            with Image.open(path) as img:
                img_rgb = img.convert('RGB')
                tensor = transform(img_rgb).unsqueeze(0).to(device)
                emb = model(tensor)
                embeddings.append(emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Embeddings are not unit L2 normalized!"
    return embeddings

def evaluate_model_on_unseen(ckpt_path, gallery_df, query_df, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    
    model = build_model(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    transform = get_eval_transform(config.get('image_size', 224))
    
    # 1. Extract Gallery Embeddings (25 original images)
    gallery_paths = gallery_df['image_path'].tolist()
    gallery_sources = gallery_df['source_id'].tolist()
    gallery_embs = extract_embeddings(model, gallery_paths, transform, device) # (25, 256)
    
    # 2. Extract Query Embeddings (75 unseen variants)
    query_paths = query_df['image_path'].tolist()
    query_sources = query_df['source_id'].tolist()
    query_transforms = query_df['transform_name'].tolist()
    query_embs = extract_embeddings(model, query_paths, transform, device) # (75, 256)
    
    # 3. Compute Cosine Similarity Matrix (Query x Gallery)
    # Both are unit L2 normalized, so dot product is cosine similarity
    sim_matrix = torch.matmul(query_embs, gallery_embs.T).numpy() # (75, 25)
    
    # 4. Pairwise Analysis (original source vs unseen variants)
    pos_sims = []
    neg_sims = []
    
    ranks = []
    reciprocal_ranks = []
    per_query_results = []
    
    for q_idx in range(len(query_sources)):
        q_src = query_sources[q_idx]
        q_trans = query_transforms[q_idx]
        sims = sim_matrix[q_idx]
        
        # Ground truth gallery index
        true_g_idx = gallery_sources.index(q_src)
        
        # Positive similarity (original vs this unseen variant)
        pos_sim = float(sims[true_g_idx])
        pos_sims.append(pos_sim)
        
        # Source-distinct negatives
        for g_idx in range(len(gallery_sources)):
            if g_idx != true_g_idx:
                neg_sims.append(float(sims[g_idx]))
                
        # Rank: argsort descending
        sorted_indices = np.argsort(-sims)
        rank = int(np.where(sorted_indices == true_g_idx)[0][0]) + 1 # 1-indexed
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
        
        per_query_results.append({
            'query_idx': q_idx,
            'source_id': q_src,
            'transform': q_trans,
            'pos_sim': pos_sim,
            'rank': rank,
            'top1_match': bool(rank == 1),
            'top5_match': bool(rank <= 5),
            'top10_match': bool(rank <= 10)
        })
        
    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)
    
    # Aggregate Metrics
    n_queries = len(query_sources)
    r1 = float(np.mean([r <= 1 for r in ranks]) * 100.0)
    r5 = float(np.mean([r <= 5 for r in ranks]) * 100.0)
    r10 = float(np.mean([r <= 10 for r in ranks]) * 100.0)
    mAP = float(np.mean(reciprocal_ranks) * 100.0)
    
    pairwise_stats = {
        'positive_similarity': {
            'mean': float(np.mean(pos_sims)),
            'std': float(np.std(pos_sims)),
            'min': float(np.min(pos_sims)),
            'median': float(np.median(pos_sims)),
            'max': float(np.max(pos_sims))
        },
        'negative_similarity': {
            'mean': float(np.mean(neg_sims)),
            'std': float(np.std(neg_sims)),
            'min': float(np.min(neg_sims)),
            'median': float(np.median(neg_sims)),
            'max': float(np.max(neg_sims))
        },
        'similarity_margin': float(np.mean(pos_sims) - np.mean(neg_sims))
    }
    
    # Per-transform breakdown
    transform_breakdown = {}
    for t_name in sorted(list(set(query_transforms))):
        t_indices = [i for i, t in enumerate(query_transforms) if t == t_name]
        t_ranks = [ranks[i] for i in t_indices]
        t_pos = pos_sims[t_indices]
        transform_breakdown[t_name] = {
            'count': len(t_indices),
            'Recall@1': float(np.mean([r <= 1 for r in t_ranks]) * 100.0),
            'Recall@5': float(np.mean([r <= 5 for r in t_ranks]) * 100.0),
            'mAP': float(np.mean([1.0 / r for r in t_ranks]) * 100.0),
            'mean_pos_sim': float(np.mean(t_pos)),
            'min_pos_sim': float(np.min(t_pos))
        }
        
    return {
        'checkpoint': Path(ckpt_path).name,
        'num_queries': n_queries,
        'num_gallery': len(gallery_sources),
        'Recall@1': r1,
        'Recall@5': r5,
        'Recall@10': r10,
        'mAP': mAP,
        'pairwise_stats': pairwise_stats,
        'transform_breakdown': transform_breakdown,
        'per_query_results': per_query_results
    }

def main():
    val_csv = PROJECT_ROOT / "data/processed/splits/val.csv"
    unseen_csv = PROJECT_ROOT / "data/processed/unseen_val_manifest.csv"
    
    val_df = pd.read_csv(val_csv)
    gallery_df = val_df[val_df['variant_type'] == 'original'].reset_index(drop=True)
    query_df = pd.read_csv(unseen_csv).reset_index(drop=True)
    
    print(f"Gallery count (original validation sources): {len(gallery_df)}")
    print(f"Query count (new unseen colorway variants):  {len(query_df)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    baseline_ckpt = PROJECT_ROOT / "outputs/checkpoints/baseline_best.pth"
    color_inv_ckpt = PROJECT_ROOT / "outputs/checkpoints/color_invariant_best.pth"
    
    print("\n--- Evaluating Experiment A (Baseline) on Unseen Colorways ---")
    res_a = evaluate_model_on_unseen(baseline_ckpt, gallery_df, query_df, device)
    
    print("\n--- Evaluating Experiment B (Color-Invariant) on Unseen Colorways ---")
    res_b = evaluate_model_on_unseen(color_inv_ckpt, gallery_df, query_df, device)
    
    comparison = {
        'benchmark_info': {
            'gallery_count': len(gallery_df),
            'query_count': len(query_df),
            'ground_truth': 'query source_id == gallery source_id',
            'metric_label': 'unseen controlled source-identity retrieval metrics'
        },
        'experiment_a_baseline': res_a,
        'experiment_b_color_invariant': res_b
    }
    
    output_json = PROJECT_ROOT / "outputs/metrics/unseen_val_retrieval_comparison.json"
    with open(output_json, 'w') as f:
        json.dump(comparison, f, indent=2)
        
    print(f"\nSaved comparison metrics to: {output_json}")
    
    # Print formatted summary table
    print("\n" + "=" * 75)
    print("UNSEEN CONTROLLED SOURCE-IDENTITY RETRIEVAL METRICS (VALIDATION ONLY)")
    print("=" * 75)
    print(f"{'Metric':<30} | {'Experiment A (Baseline)':<20} | {'Experiment B (Color-Inv)':<20}")
    print("-" * 75)
    print(f"{'Recall@1':<30} | {res_a['Recall@1']:>19.2f}% | {res_b['Recall@1']:>19.2f}%")
    print(f"{'Recall@5':<30} | {res_a['Recall@5']:>19.2f}% | {res_b['Recall@5']:>19.2f}%")
    print(f"{'Recall@10':<30} | {res_a['Recall@10']:>19.2f}% | {res_b['Recall@10']:>19.2f}%")
    print(f"{'mAP':<30} | {res_a['mAP']:>19.2f}% | {res_b['mAP']:>19.2f}%")
    print("-" * 75)
    print("PAIRWISE COLOR-INVARIANCE ANALYSIS:")
    p_a = res_a['pairwise_stats']['positive_similarity']
    p_b = res_b['pairwise_stats']['positive_similarity']
    n_a = res_a['pairwise_stats']['negative_similarity']
    n_b = res_b['pairwise_stats']['negative_similarity']
    print(f"{'Pos Sim (Mean)':<30} | {p_a['mean']:>20.4f} | {p_b['mean']:>20.4f}")
    print(f"{'Pos Sim (Std)':<30} | {p_a['std']:>20.4f} | {p_b['std']:>20.4f}")
    print(f"{'Pos Sim (Min)':<30} | {p_a['min']:>20.4f} | {p_b['min']:>20.4f}")
    print(f"{'Pos Sim (Median)':<30} | {p_a['median']:>20.4f} | {p_b['median']:>20.4f}")
    print(f"{'Pos Sim (Max)':<30} | {p_a['max']:>20.4f} | {p_b['max']:>20.4f}")
    print(f"{'Neg Sim (Mean)':<30} | {n_a['mean']:>20.4f} | {n_b['mean']:>20.4f}")
    print(f"{'Similarity Margin':<30} | {res_a['pairwise_stats']['similarity_margin']:>20.4f} | {res_b['pairwise_stats']['similarity_margin']:>20.4f}")
    print("=" * 75)
    
    print("\nPER-TRANSFORMATION BREAKDOWN (Recall@1):")
    for t_name in sorted(res_a['transform_breakdown'].keys()):
        tb_a = res_a['transform_breakdown'][t_name]
        tb_b = res_b['transform_breakdown'][t_name]
        print(f"  {t_name:<30}: Exp A = {tb_a['Recall@1']:>5.1f}% | Exp B = {tb_b['Recall@1']:>5.1f}%")

if __name__ == '__main__':
    main()
