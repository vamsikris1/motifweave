import os
import sys
import time
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
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

def extract_embeddings_batch(model, image_paths, transform, device, batch_size=32):
    model.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            tensors = []
            for p in batch_paths:
                with Image.open(p) as img:
                    tensors.append(transform(img.convert('RGB')))
            batch_tensor = torch.stack(tensors).to(device)
            embs = model(batch_tensor)
            all_embs.append(embs.cpu())
    all_embs = torch.cat(all_embs, dim=0)
    norms = torch.norm(all_embs, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Embeddings not unit L2 normalized!"
    return all_embs

def compute_roc_curve(y_true, y_scores, num_thresholds=1000):
    # Pure numpy ROC curve computation
    thresholds = np.linspace(-1.0, 1.0, num_thresholds)
    tpr_list = []
    fpr_list = []
    
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    
    for th in thresholds:
        pred_pos = (y_scores >= th)
        tp = np.sum(pred_pos & pos_mask)
        fp = np.sum(pred_pos & neg_mask)
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    
    # Sort by FPR ascending for trapezoidal AUC integration
    sort_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sort_idx]
    tpr_sorted = tpr_arr[sort_idx]
    if hasattr(np, 'trapezoid'):
        auc_score = float(np.trapezoid(tpr_sorted, fpr_sorted))
    else:
        auc_score = float(np.sum((tpr_sorted[:-1] + tpr_sorted[1:]) * np.diff(fpr_sorted)) / 2.0)
    
    return fpr_arr, tpr_arr, thresholds, auc_score

def calibrate_threshold_on_validation(model, val_csv, transform, device):
    print("--- 1. Calibrating Verification Thresholds on VALIDATION Split Only ---")
    val_df = pd.read_csv(val_csv)
    image_paths = val_df['image_path'].tolist()
    source_ids = val_df['source_id'].tolist()
    
    embs = extract_embeddings_batch(model, image_paths, transform, device)
    sim_mat = torch.matmul(embs, embs.T).numpy()
    
    pos_sims = []
    neg_sims = []
    
    n = len(val_df)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_mat[i, j])
            if source_ids[i] == source_ids[j]:
                pos_sims.append(s)
            else:
                neg_sims.append(s)
                
    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)
    
    print(f"Validation Positive Pairs: {len(pos_sims)} | Negative Pairs: {len(neg_sims)}")
    
    # Grid search threshold
    thresholds = np.linspace(-0.5, 1.0, 1500)
    best_eer_diff = float('inf')
    theta_eer = 0.0
    val_eer = 0.0
    
    best_f1 = -1.0
    theta_f1 = 0.0
    val_best_f1_metrics = {}
    
    for th in thresholds:
        far = np.mean(neg_sims >= th)
        frr = np.mean(pos_sims < th)
        
        diff = abs(far - frr)
        if diff < best_eer_diff:
            best_eer_diff = diff
            theta_eer = float(th)
            val_eer = float((far + frr) / 2.0)
            
        tp = np.sum(pos_sims >= th)
        fp = np.sum(neg_sims >= th)
        fn = np.sum(pos_sims < th)
        tn = np.sum(neg_sims < th)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            theta_f1 = float(th)
            val_best_f1_metrics = {
                'f1': float(f1),
                'precision': float(prec),
                'recall': float(rec),
                'accuracy': float((tp + tn) / (tp + tn + fp + fn))
            }
            
    print(f"Validation EER Threshold (theta_EER): {theta_eer:.4f} (Val EER: {val_eer*100:.2f}%)")
    print(f"Validation Max-F1 Threshold (theta_F1):  {theta_f1:.4f} (Val F1: {val_best_f1_metrics['f1']:.4f})")
    
    return {
        'theta_eer': theta_eer,
        'val_eer': val_eer,
        'theta_f1': theta_f1,
        'val_f1_metrics': val_best_f1_metrics,
        'val_pos_mean': float(np.mean(pos_sims)),
        'val_neg_mean': float(np.mean(neg_sims))
    }

def evaluate_test_track_a(model, test_csv, transform, device):
    print("\n--- 2. Evaluating Track A: Controlled Source-Identity Retrieval (Held-Out Test Split) ---")
    test_df = pd.read_csv(test_csv)
    
    gallery_df = test_df[test_df['variant_type'] == 'original'].reset_index(drop=True)
    query_df = test_df[test_df['variant_type'] != 'original'].reset_index(drop=True)
    
    assert len(gallery_df) == 40, f"Expected 40 test gallery images, found {len(gallery_df)}"
    assert len(query_df) == 120, f"Expected 120 test query images, found {len(query_df)}"
    
    gallery_paths = gallery_df['image_path'].tolist()
    gallery_sources = gallery_df['source_id'].tolist()
    gallery_embs = extract_embeddings_batch(model, gallery_paths, transform, device)
    
    query_paths = query_df['image_path'].tolist()
    query_sources = query_df['source_id'].tolist()
    query_variants = query_df['variant_type'].tolist()
    query_embs = extract_embeddings_batch(model, query_paths, transform, device)
    
    # Cosine similarity matrix (120 queries x 40 gallery)
    sim_mat = torch.matmul(query_embs, gallery_embs.T).numpy()
    
    ranks = []
    reciprocal_ranks = []
    pos_sims = []
    
    for q_idx in range(len(query_sources)):
        q_src = query_sources[q_idx]
        sims = sim_mat[q_idx]
        true_g_idx = gallery_sources.index(q_src)
        
        pos_sims.append(float(sims[true_g_idx]))
        
        # Rank descending
        sorted_indices = np.argsort(-sims)
        rank = int(np.where(sorted_indices == true_g_idx)[0][0]) + 1
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
        
    r1 = float(np.mean([r <= 1 for r in ranks]) * 100.0)
    r5 = float(np.mean([r <= 5 for r in ranks]) * 100.0)
    r10 = float(np.mean([r <= 10 for r in ranks]) * 100.0)
    mAP = float(np.mean(reciprocal_ranks) * 100.0)
    
    # Breakdown by variant type
    variant_breakdown = {}
    for v_name in sorted(list(set(query_variants))):
        v_indices = [i for i, v in enumerate(query_variants) if v == v_name]
        v_ranks = [ranks[i] for i in v_indices]
        v_pos = [pos_sims[i] for i in v_indices]
        variant_breakdown[v_name] = {
            'count': len(v_indices),
            'Recall@1': float(np.mean([r <= 1 for r in v_ranks]) * 100.0),
            'Recall@5': float(np.mean([r <= 5 for r in v_ranks]) * 100.0),
            'mAP': float(np.mean([1.0 / r for r in v_ranks]) * 100.0),
            'mean_pos_sim': float(np.mean(v_pos))
        }
        
    print(f"Track A Overall Test Metrics (120 queries -> 40 gallery):")
    print(f"  Recall@1:  {r1:.2f}% ({sum(r <= 1 for r in ranks)} / 120)")
    print(f"  Recall@5:  {r5:.2f}% ({sum(r <= 5 for r in ranks)} / 120)")
    print(f"  Recall@10: {r10:.2f}% ({sum(r <= 10 for r in ranks)} / 120)")
    print(f"  mAP:       {mAP:.2f}%")
    
    return {
        'gallery_count': len(gallery_sources),
        'query_count': len(query_sources),
        'Recall@1': r1,
        'Recall@5': r5,
        'Recall@10': r10,
        'mAP': mAP,
        'ranks': ranks,
        'variant_breakdown': variant_breakdown
    }

def evaluate_test_pairwise_verification(model, test_csv, calib_info, transform, device, figures_dir):
    print("\n--- 3. Evaluating Pairwise Verification on Held-Out Test Split ---")
    test_df = pd.read_csv(test_csv)
    image_paths = test_df['image_path'].tolist()
    source_ids = test_df['source_id'].tolist()
    
    embs = extract_embeddings_batch(model, image_paths, transform, device)
    sim_mat = torch.matmul(embs, embs.T).numpy()
    
    pos_sims = []
    neg_sims = []
    y_true = []
    y_scores = []
    
    n = len(test_df)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_mat[i, j])
            is_pos = (source_ids[i] == source_ids[j])
            if is_pos:
                pos_sims.append(s)
                y_true.append(1)
            else:
                neg_sims.append(s)
                y_true.append(0)
            y_scores.append(s)
            
    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    print(f"Test Positive Pairs: {len(pos_sims)} | Test Negative Proxy Pairs: {len(neg_sims)}")
    
    pos_stats = {
        'mean': float(np.mean(pos_sims)),
        'std': float(np.std(pos_sims)),
        'min': float(np.min(pos_sims)),
        'median': float(np.median(pos_sims)),
        'max': float(np.max(pos_sims))
    }
    
    neg_stats = {
        'mean': float(np.mean(neg_sims)),
        'std': float(np.std(neg_sims)),
        'min': float(np.min(neg_sims)),
        'median': float(np.median(neg_sims)),
        'max': float(np.max(neg_sims))
    }
    
    sim_margin = pos_stats['mean'] - neg_stats['mean']
    
    print(f"Positive Similarity: Mean={pos_stats['mean']:.4f} (Min={pos_stats['min']:.4f}, Median={pos_stats['median']:.4f}, Max={pos_stats['max']:.4f})")
    print(f"Negative Similarity: Mean={neg_stats['mean']:.4f} (Min={neg_stats['min']:.4f}, Median={neg_stats['median']:.4f}, Max={neg_stats['max']:.4f})")
    print(f"Similarity Margin:   {sim_margin:+.4f}")
    
    # Compute ROC Curve & AUC on Test Pairs
    fpr_arr, tpr_arr, thresholds, roc_auc = compute_roc_curve(y_true, y_scores)
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    
    # Evaluate at Fixed Validation-Calibrated Operating Thresholds
    th_eer = calib_info['theta_eer']
    th_f1 = calib_info['theta_f1']
    
    def evaluate_at_threshold(th):
        pred_pos = (y_scores >= th)
        tp = int(np.sum(pred_pos & (y_true == 1)))
        fp = int(np.sum(pred_pos & (y_true == 0)))
        fn = int(np.sum((~pred_pos) & (y_true == 1)))
        tn = int(np.sum((~pred_pos) & (y_true == 0)))
        
        tpr = tp / len(pos_sims) if len(pos_sims) > 0 else 0.0
        tnr = tn / len(neg_sims) if len(neg_sims) > 0 else 0.0
        far = fp / len(neg_sims) if len(neg_sims) > 0 else 0.0
        frr = fn / len(pos_sims) if len(pos_sims) > 0 else 0.0
        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * tpr) / (prec + tpr) if (prec + tpr) > 0 else 0.0
        
        return {
            'threshold': float(th),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall_tpr': float(tpr),
            'specificity_tnr': float(tnr),
            'FAR': float(far),
            'FRR': float(frr),
            'f1_score': float(f1),
            'confusion_matrix': {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}
        }
        
    metrics_at_calib_eer = evaluate_at_threshold(th_eer)
    metrics_at_calib_f1 = evaluate_at_threshold(th_f1)
    
    # Find empirical Test EER for reporting
    test_eer_idx = np.argmin(np.abs(fpr_arr - (1.0 - tpr_arr)))
    test_eer = float((fpr_arr[test_eer_idx] + (1.0 - tpr_arr[test_eer_idx])) / 2.0)
    
    print(f"\nVerification Performance on Test using VALIDATION-CALIBRATED Operating Point (theta_EER = {th_eer:.4f}):")
    print(f"  Accuracy:    {metrics_at_calib_eer['accuracy']*100:.2f}%")
    print(f"  Precision:   {metrics_at_calib_eer['precision']*100:.2f}%")
    print(f"  Recall (TPR):{metrics_at_calib_eer['recall_tpr']*100:.2f}%")
    print(f"  FAR:         {metrics_at_calib_eer['FAR']*100:.2f}%")
    print(f"  FRR:         {metrics_at_calib_eer['FRR']*100:.2f}%")
    print(f"  F1-Score:    {metrics_at_calib_eer['f1_score']:.4f}")
    
    # --- Generate Figures ---
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Similarity Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(neg_sims, bins=50, density=True, alpha=0.6, color='steelblue', label=f'Source-Distinct Negatives (N={len(neg_sims)})')
    ax.hist(pos_sims, bins=30, density=True, alpha=0.6, color='crimson', label=f'Same-Source Positives (N={len(pos_sims)})')
    ax.axvline(th_eer, color='black', linestyle='--', linewidth=2, label=f'Val Calibrated Threshold ({th_eer:.2f})')
    ax.set_title("MotifWeave Test Pairwise Cosine Similarity Distribution", fontsize=12, fontweight='bold')
    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    dist_fig_path = figures_dir / "test_similarity_distributions.png"
    plt.savefig(dist_fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 2. ROC Curve Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_arr, tpr_arr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    ax.scatter([metrics_at_calib_eer['FAR']], [metrics_at_calib_eer['recall_tpr']], color='red', s=70, zorder=5, 
               label=f'Val Operating Point (FAR={metrics_at_calib_eer["FAR"]*100:.1f}%, TPR={metrics_at_calib_eer["recall_tpr"]*100:.1f}%)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Acceptance Rate (FAR)', fontsize=11)
    ax.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=11)
    ax.set_title('Test ROC Curve (Pairwise Verification)', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_fig_path = figures_dir / "test_roc_curve.png"
    plt.savefig(roc_fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return {
        'positive_stats': pos_stats,
        'negative_stats': neg_stats,
        'similarity_margin': float(sim_margin),
        'roc_auc': float(roc_auc),
        'test_eer_reference': float(test_eer),
        'at_calibrated_eer_threshold': metrics_at_calib_eer,
        'at_calibrated_f1_threshold': metrics_at_calib_f1
    }

def evaluate_test_track_b(model, test_csv, transform, device, figures_dir):
    print("\n--- 4. Evaluating Track B: Exploratory Original-Image Retrieval (Test Split) ---")
    test_df = pd.read_csv(test_csv)
    orig_df = test_df[test_df['variant_type'] == 'original'].reset_index(drop=True)
    assert len(orig_df) == 40, f"Expected 40 original test images, found {len(orig_df)}"
    
    paths = orig_df['image_path'].tolist()
    sources = orig_df['source_id'].tolist()
    embs = extract_embeddings_batch(model, paths, transform, device)
    
    # 40 x 40 pairwise cosine similarities
    sim_mat = torch.matmul(embs, embs.T).numpy()
    
    # Zero out diagonal
    np.fill_diagonal(sim_mat, -1e9)
    
    nn_sims = []
    top_k_records = []
    
    for i in range(len(sources)):
        sims = sim_mat[i]
        sorted_indices = np.argsort(-sims)
        
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1]
        top3_idx = sorted_indices[2]
        
        nn_sims.append(float(sims[top1_idx]))
        
        top_k_records.append({
            'query_source': sources[i],
            'query_path': paths[i],
            'top1_source': sources[top1_idx],
            'top1_sim': float(sims[top1_idx]),
            'top1_path': paths[top1_idx],
            'top2_source': sources[top2_idx],
            'top2_sim': float(sims[top2_idx]),
            'top2_path': paths[top2_idx],
            'top3_source': sources[top3_idx],
            'top3_sim': float(sims[top3_idx]),
            'top3_path': paths[top3_idx]
        })
        
    nn_sims = np.array(nn_sims)
    nn_stats = {
        'mean': float(np.mean(nn_sims)),
        'std': float(np.std(nn_sims)),
        'min': float(np.min(nn_sims)),
        'median': float(np.median(nn_sims)),
        'max': float(np.max(nn_sims))
    }
    
    print(f"Track B 1-NN Similarity Distribution across 40 Original Images:")
    print(f"  Mean:   {nn_stats['mean']:.4f}")
    print(f"  Std:    {nn_stats['std']:.4f}")
    print(f"  Min:    {nn_stats['min']:.4f}")
    print(f"  Median: {nn_stats['median']:.4f}")
    print(f"  Max:    {nn_stats['max']:.4f}")
    
    # Generate PCA visualization of the 40 original images via SVD
    embs_np = embs.numpy()
    centered = embs_np - np.mean(embs_np, axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    embs_pca = centered @ vt[:2].T
    var_exp = (s**2) / np.sum(s**2) * 100.0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(embs_pca[:, 0], embs_pca[:, 1], c='teal', s=60, edgecolors='black', alpha=0.85)
    for i, src in enumerate(sources):
        # Annotate selected points
        if i % 4 == 0:
            ax.annotate(src.replace('deeplure_', ''), (embs_pca[i, 0]+0.015, embs_pca[i, 1]+0.015), fontsize=8, alpha=0.8)
    ax.set_title("Track B: Exploratory 2D PCA of 40 Original Test Sarees", fontsize=12, fontweight='bold')
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=10)
    plt.tight_layout()
    pca_fig_path = Path(figures_dir) / "track_b_original_pca.png"
    plt.savefig(pca_fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Generate Qualitative Retrieval Grid (5 diverse query examples)
    # Select queries at percentiles of 1-NN similarity: lowest, 25th, 50th, 75th, highest
    sorted_order = np.argsort(nn_sims)
    selected_indices = [
        sorted_order[0],                      # Minimum NN similarity
        sorted_order[len(sorted_order) // 4], # 25th percentile
        sorted_order[len(sorted_order) // 2], # 50th percentile
        sorted_order[3 * len(sorted_order) // 4], # 75th percentile
        sorted_order[-1]                      # Maximum NN similarity
    ]
    
    fig, axes = plt.subplots(5, 4, figsize=(14, 15))
    for row_idx, q_i in enumerate(selected_indices):
        rec = top_k_records[q_i]
        
        # Col 0: Query
        with Image.open(rec['query_path']) as img:
            axes[row_idx, 0].imshow(img.convert('RGB'))
        axes[row_idx, 0].set_title(f"Query: {rec['query_source']}", fontsize=9, fontweight='bold', color='navy')
        axes[row_idx, 0].axis('off')
        
        # Col 1: Top 1
        with Image.open(rec['top1_path']) as img:
            axes[row_idx, 1].imshow(img.convert('RGB'))
        axes[row_idx, 1].set_title(f"1st: {rec['top1_source']}\nSim = {rec['top1_sim']:.4f}", fontsize=9)
        axes[row_idx, 1].axis('off')
        
        # Col 2: Top 2
        with Image.open(rec['top2_path']) as img:
            axes[row_idx, 2].imshow(img.convert('RGB'))
        axes[row_idx, 2].set_title(f"2nd: {rec['top2_source']}\nSim = {rec['top2_sim']:.4f}", fontsize=9)
        axes[row_idx, 2].axis('off')
        
        # Col 3: Top 3
        with Image.open(rec['top3_path']) as img:
            axes[row_idx, 3].imshow(img.convert('RGB'))
        axes[row_idx, 3].set_title(f"3rd: {rec['top3_source']}\nSim = {rec['top3_sim']:.4f}", fontsize=9)
        axes[row_idx, 3].axis('off')
        
    plt.suptitle("Track B: Exploratory Nearest-Neighbor Retrieval on Original Test Images", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    qual_grid_path = Path(figures_dir) / "track_b_qualitative_grid.png"
    plt.savefig(qual_grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Track B Qualitative Retrieval Grid saved to: {qual_grid_path}")
    
    return {
        'nn_similarity_stats': nn_stats,
        'sample_top_k_records': [top_k_records[idx] for idx in selected_indices]
    }

def measure_efficiency_and_latency(model, transform, device):
    print("\n--- 5. Measuring Inference Latency & Model Efficiency ---")
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)
            
    # Measure latency over 50 passes
    num_runs = 50
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0) # in ms
        
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    fps = float(1000.0 / mean_lat)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params
    
    print(f"Parameters: Total={total_params:,} | Trainable={trainable_params:,} | Frozen={frozen_params:,}")
    print(f"CPU Latency (batch=1): {mean_lat:.2f} ms +/- {std_lat:.2f} ms ({fps:.1f} img/sec)")
    
    return {
        'embedding_dim': 256,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': frozen_params,
        'mean_latency_ms': mean_lat,
        'std_latency_ms': std_lat,
        'throughput_fps': fps,
        'device': str(device),
        'torch_version': torch.__version__,
        'platform': platform.platform(),
        'processor': platform.processor()
    }

def main():
    val_csv = PROJECT_ROOT / "data/processed/splits/val.csv"
    test_csv = PROJECT_ROOT / "data/processed/splits/test.csv"
    baseline_ckpt = PROJECT_ROOT / "outputs/checkpoints/baseline_best.pth"
    color_inv_ckpt = PROJECT_ROOT / "outputs/checkpoints/color_invariant_best.pth"
    
    output_dir = PROJECT_ROOT / "outputs/final"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference Device: {device}")
    
    # 1. Load Provisional Final Model (Experiment A - Baseline)
    ckpt_a = torch.load(baseline_ckpt, map_location=device)
    config_a = ckpt_a['config']
    model_a = build_model(config_a).to(device)
    model_a.load_state_dict(ckpt_a['model_state_dict'])
    model_a.eval()
    transform = get_eval_transform(config_a.get('image_size', 224))
    
    # 2. Validation Calibration ONLY (Leakage-Safe)
    calib_info = calibrate_threshold_on_validation(model_a, val_csv, transform, device)
    
    # 3. Track A: Controlled Colorway Retrieval on Test Split
    track_a_results = evaluate_test_track_a(model_a, test_csv, transform, device)
    
    # 4. Pairwise Verification on Test Split
    verification_results = evaluate_test_pairwise_verification(model_a, test_csv, calib_info, transform, device, figures_dir)
    
    # 5. Track B: Exploratory Original-Image Retrieval on Test Split
    track_b_results = evaluate_test_track_b(model_a, test_csv, transform, device, figures_dir)
    
    # 6. Efficiency & Latency Benchmarks
    efficiency_results = measure_efficiency_and_latency(model_a, transform, device)
    
    # 7. Optional Post-Selection Test Comparison (Experiment B on Test)
    print("\n--- 6. Post-Selection Comparative Evaluation: Experiment B on Held-Out Test ---")
    ckpt_b = torch.load(color_inv_ckpt, map_location=device)
    config_b = ckpt_b['config']
    model_b = build_model(config_b).to(device)
    model_b.load_state_dict(ckpt_b['model_state_dict'])
    model_b.eval()
    
    # Calibrate B on validation
    calib_info_b = calibrate_threshold_on_validation(model_b, val_csv, transform, device)
    track_a_b = evaluate_test_track_a(model_b, test_csv, transform, device)
    
    # Pairwise for B
    figures_dir_b = figures_dir / "experiment_b_comparison"
    verification_b = evaluate_test_pairwise_verification(model_b, test_csv, calib_info_b, transform, device, figures_dir_b)
    
    final_output = {
        'metadata': {
            'project': 'MotifWeave',
            'dataset_split': 'held_out_test_40_sources',
            'total_test_images': 160,
            'original_test_images': 40,
            'synthetic_test_queries': 120,
            'seed': 42
        },
        'selection_decision': {
            'selected_final_model': 'Experiment A (ConvNeXt-Tiny Baseline)',
            'checkpoint_path': str(baseline_ckpt),
            'rationale': 'Superior validation cosine margin (+0.5787 vs +0.5555), higher initial validation Recall@1 (94.67% vs 92.00%), identical unseen-colorway retrieval (94.67% R@1, 97.33% mAP), and lower source-distinct negative cross-similarity.'
        },
        'validation_threshold_calibration': calib_info,
        'track_a_test_retrieval': track_a_results,
        'pairwise_verification_test': verification_results,
        'track_b_exploratory_retrieval': track_b_results,
        'efficiency_and_latency': efficiency_results,
        'post_selection_experiment_b_comparison': {
            'checkpoint_path': str(color_inv_ckpt),
            'validation_threshold_calibration': calib_info_b,
            'track_a_test_retrieval': track_a_b,
            'pairwise_verification_test': verification_b
        }
    }
    
    metrics_file = metrics_dir / "final_test_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"\nFinal Test Metrics saved cleanly to: {metrics_file}")
    print("=" * 80)
    print("PHASE 5 TEST EVALUATION COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
