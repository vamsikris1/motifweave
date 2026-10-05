import os
import sys
import argparse
import json
import yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Maximize CPU core utilization on CPU machines
torch.set_num_threads(os.cpu_count() or 4)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import SareeDataset, get_baseline_transforms, get_color_invariant_transforms
from src.samplers import SourceAwareBatchSampler
from src.model import build_model
from src.losses import SupervisedContrastiveLoss
from src.utils import set_seed, save_checkpoint, load_checkpoint, compute_pairwise_validation_statistics, plot_pca_embeddings

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, labels, _, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        
        loss.backward()
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(1, num_batches)

@torch.no_grad()
def evaluate_validation(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_embeddings = []
    all_labels = []
    
    for images, labels, _, _ in dataloader:
        images = images.to(device)
        labels_device = labels.to(device)
        
        embeddings = model(images)
        loss = criterion(embeddings, labels_device)
        
        total_loss += loss.item()
        num_batches += 1
        
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())
        
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    avg_loss = total_loss / max(1, num_batches)
    pairwise_stats = compute_pairwise_validation_statistics(all_embeddings, all_labels)
    pairwise_stats['val_loss'] = round(avg_loss, 4)
    
    return pairwise_stats, all_embeddings, all_labels

def run_training(config_path, resume_checkpoint=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup paths and naming
    exp_name = config.get('experiment_name', 'baseline')
    checkpoint_dir = PROJECT_ROOT / config.get('checkpoint_dir', 'outputs/checkpoints')
    log_dir = PROJECT_ROOT / config.get('log_dir', 'outputs/metrics')
    figures_dir = PROJECT_ROOT / config.get('figures_dir', 'outputs/figures')
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Deterministic Seeding
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"[{exp_name}] Random seed locked to: {seed}", flush=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{exp_name}] Using compute device: {device} (Torch Threads: {torch.get_num_threads()})", flush=True)
    
    # 2. Datasets & Source-Aware Sampling
    train_csv = PROJECT_ROOT / config['train_csv']
    val_csv = PROJECT_ROOT / config['val_csv']
    
    print(f"Loading TRAIN split from: {train_csv}", flush=True)
    print(f"Loading VALIDATION split from: {val_csv}", flush=True)
    print("NOTE: Test split is strictly excluded from training and validation.", flush=True)
    
    image_size = config.get('image_size', 224)
    aug_type = config.get('augmentation', 'standard_imagenet')
    
    if aug_type == 'color_invariant':
        print(f"Applying Color-Invariant Training Augmentations (Hue={config.get('hue_jitter', 0.3)}, Sat={config.get('saturation_jitter', 0.2)}, Bright={config.get('brightness_jitter', 0.15)}, Grayscale_p={config.get('p_grayscale', 0.15)})", flush=True)
        train_transform = get_color_invariant_transforms(
            image_size=image_size,
            is_training=True,
            hue=config.get('hue_jitter', 0.30),
            saturation=config.get('saturation_jitter', 0.20),
            brightness=config.get('brightness_jitter', 0.15),
            contrast=config.get('contrast_jitter', 0.15),
            p_grayscale=config.get('p_grayscale', 0.15)
        )
    else:
        print("Applying Standard ImageNet Training Augmentations (Baseline)", flush=True)
        train_transform = get_baseline_transforms(image_size=image_size, is_training=True)
        
    val_transform = get_baseline_transforms(image_size=image_size, is_training=False)
    
    train_dataset = SareeDataset(train_csv, transform=train_transform)
    val_dataset = SareeDataset(val_csv, transform=val_transform)
    
    print(f"Train samples: {len(train_dataset)} ({len(train_dataset.label_encoder)} unique sources)", flush=True)
    print(f"Val samples:   {len(val_dataset)} ({len(val_dataset.label_encoder)} unique sources)", flush=True)
    
    # Batch Sampler for Train
    p_sources = config.get('p_sources', 8)
    k_variants = config.get('k_variants', 4)
    batch_sampler = SourceAwareBatchSampler(train_dataset, p_sources=p_sources, k_variants=k_variants, seed=seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=config.get('num_workers', 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    # 3. Model Architecture
    model = build_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\nModel Initialized: {config.get('backbone')} with {config.get('pooling')} pooling", flush=True)
    print(f"Total Parameters:     {total_params:,}", flush=True)
    print(f"Trainable Parameters: {trainable_params:,}", flush=True)
    print(f"Frozen Parameters:    {frozen_params:,}", flush=True)
    print(f"Embedding Dimension:  {config.get('embedding_dim', 256)}", flush=True)
    
    # 4. Objective, Optimizer & Scheduler
    temperature = config.get('temperature', 0.07)
    criterion = SupervisedContrastiveLoss(temperature=temperature).to(device)
    
    lr = float(config.get('learning_rate', 5e-4))
    wd = float(config.get('weight_decay', 1e-4))
    epochs = config.get('epochs', 5)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    min_lr = float(config.get('min_lr', 5e-6))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    
    # Check for resume
    start_epoch = 1
    best_val_loss = float('inf')
    best_margin = -float('inf')
    history = []
    
    # Determine checkpoint filenames
    # If exp_name is baseline_convnext_tiny -> baseline_best.pth
    # If exp_name is color_invariant_convnext_tiny -> color_invariant_best.pth
    prefix = "color_invariant" if "color" in exp_name else "baseline"
    best_ckpt_path = checkpoint_dir / f"{prefix}_best.pth"
    last_ckpt_path = checkpoint_dir / f"{prefix}_last.pth"
    history_path = log_dir / f"{prefix}_history.json"
    pca_fig_path = figures_dir / f"{prefix}_val_pca.png"
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from checkpoint: {resume_checkpoint}", flush=True)
        ckpt = load_checkpoint(resume_checkpoint, model, optimizer, scheduler)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        history = ckpt.get('history', [])
        
    print(f"\nStarting Training Schedule ({prefix.upper()})...", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Val Loss':^10} | {'Pos Sim':^9} | {'Neg Sim':^9} | {'Margin':^9} | {'LR':^9}", flush=True)
    print("-" * 80, flush=True)
    
    best_val_embeddings = None
    best_val_labels = None
    
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_stats, val_embeddings, val_labels = evaluate_validation(model, val_loader, criterion, device)
        val_loss = val_stats['val_loss']
        pos_sim = val_stats['mean_pos_sim']
        neg_sim = val_stats['mean_neg_sim']
        margin = val_stats['sim_margin']
        current_lr = scheduler.get_last_lr()[0]
        
        scheduler.step()
        
        epoch_record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss': val_loss,
            'mean_pos_sim': pos_sim,
            'mean_neg_sim': neg_sim,
            'sim_margin': margin,
            'lr': current_lr
        }
        history.append(epoch_record)
        
        print(f"{epoch:^7d} | {train_loss:^11.4f} | {val_loss:^10.4f} | {pos_sim:^9.4f} | {neg_sim:^9.4f} | {margin:^9.4f} | {current_lr:^9.2e}", flush=True)
        
        # Save best model checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_margin = margin
            best_val_embeddings = val_embeddings.clone()
            best_val_labels = val_labels.clone()
            
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'sim_margin': margin,
                'config': config,
                'history': history,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'embedding_dim': config.get('embedding_dim', 256)
            }, best_ckpt_path)
            
    # Save final checkpoint
    save_checkpoint({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'sim_margin': margin,
        'config': config,
        'history': history
    }, last_ckpt_path)
    
    # Save training history JSON
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}", flush=True)
    print(f"Best checkpoint saved to:   {best_ckpt_path}", flush=True)
    
    # 5. Sanity Checks & Visual Embeddings Plot
    print("\n" + "=" * 50, flush=True)
    print(f"RUNNING {prefix.upper()} MODEL SANITY CHECKS", flush=True)
    print("=" * 50, flush=True)
    
    # Re-verify best model on validation
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()
    
    val_stats, final_embeddings, final_labels = evaluate_validation(model, val_loader, criterion, device)
    
    # Check 1: Finite values
    has_nan = torch.isnan(final_embeddings).any().item()
    has_inf = torch.isinf(final_embeddings).any().item()
    print(f"1. Embeddings Finite (No NaN/Inf): {'PASSED' if (not has_nan and not has_inf) else 'FAILED'}", flush=True)
    assert not has_nan and not has_inf, "Embeddings contain NaN or Inf values!"
    
    # Check 2: Embedding Dimension
    emb_dim = final_embeddings.shape[1]
    expected_dim = config.get('embedding_dim', 256)
    print(f"2. Embedding Dimension: {emb_dim} (Expected: {expected_dim}) -> {'PASSED' if emb_dim == expected_dim else 'FAILED'}", flush=True)
    assert emb_dim == expected_dim, f"Embedding dimension mismatch: {emb_dim} != {expected_dim}"
    
    # Check 3: L2 Norm == 1.0
    norms = torch.norm(final_embeddings, p=2, dim=1)
    norm_mean = norms.mean().item()
    norm_std = norms.std().item()
    print(f"3. L2 Norm Check: Mean = {norm_mean:.6f}, Std = {norm_std:.6f} -> {'PASSED' if abs(norm_mean - 1.0) < 1e-4 else 'FAILED'}", flush=True)
    assert abs(norm_mean - 1.0) < 1e-4, f"Embeddings not unit normalized: {norm_mean}"
    
    # Check 4: Positive vs Negative Margin
    print(f"4. Validation Pairwise Statistics (VALIDATION ONLY):", flush=True)
    print(f"   - Positive Pairs Evaluated: {val_stats['num_pos_pairs']:,}", flush=True)
    print(f"   - Negative Pairs Evaluated: {val_stats['num_neg_pairs']:,}", flush=True)
    print(f"   - Mean Same-Source Similarity:     {val_stats['mean_pos_sim']:.4f} (std={val_stats['std_pos_sim']:.4f})", flush=True)
    print(f"   - Mean Source-Distinct Similarity: {val_stats['mean_neg_sim']:.4f} (std={val_stats['std_neg_sim']:.4f})", flush=True)
    print(f"   - Similarity Margin (Pos - Neg):   {val_stats['sim_margin']:.4f}", flush=True)
    
    # Check 5: Save 2D PCA Visualization
    plot_pca_embeddings(final_embeddings, final_labels, str(pca_fig_path), title=f"{prefix.upper()} Validation Embeddings PCA")
    print(f"5. 2D PCA Embeddings Figure generated successfully at {pca_fig_path}", flush=True)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'embedding_dim': emb_dim,
        'best_val_loss': best_val_loss,
        'val_stats': val_stats,
        'best_checkpoint_path': str(best_ckpt_path),
        'history_path': str(history_path),
        'pca_fig_path': str(pca_fig_path)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MotifWeave Training")
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', help='Path to configuration YAML')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()
    
    cfg_file = PROJECT_ROOT / args.config if not os.path.isabs(args.config) else Path(args.config)
    run_training(cfg_file, resume_checkpoint=args.resume)
