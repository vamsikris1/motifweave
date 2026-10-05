import os
import sys
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model, embed_image, embed_images, get_inference_transform
from src.retrieval import build_gallery, load_gallery, retrieve, verify_pair, FROZEN_VAL_EER_THRESHOLD

def get_model_param_hash(model):
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()

def run_sanity_checks():
    print("=" * 75)
    print("PHASE 6: INFERENCE & RETRIEVAL SANITY CHECKS")
    print("=" * 75)
    
    checkpoint_path = PROJECT_ROOT / "outputs/checkpoints/baseline_best.pth"
    val_csv = PROJECT_ROOT / "data/processed/splits/val.csv"
    val_df = pd.read_csv(val_csv)
    
    # Select sample images
    orig_samples = val_df[val_df['variant_type'] == 'original'].head(5)
    var_samples = val_df[val_df['variant_type'] == 'color_variant_1'].head(5)
    
    sample_img_1 = Path(orig_samples.iloc[0]['image_path'])
    sample_img_1_var = Path(val_df[(val_df['source_id'] == orig_samples.iloc[0]['source_id']) & 
                                   (val_df['variant_type'] == 'color_variant_1')].iloc[0]['image_path'])
    sample_img_2 = Path(orig_samples.iloc[1]['image_path'])
    
    # 1. Model Loading Check
    print("\n1. Testing Model Loading...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config, device = load_model(checkpoint_path, device=device)
    assert not model.training, "Model is in training mode instead of eval mode!"
    for p in model.parameters():
        assert not p.requires_grad, "Model parameter has requires_grad == True!"
    print(f"   Model loaded cleanly in eval mode. All parameters frozen: PASSED")
    
    initial_param_hash = get_model_param_hash(model)
    
    # 2. Embedding Shape and L2 Norm Check
    print("\n2. Testing Embedding Shape and L2 Normalization...")
    emb_1 = embed_image(sample_img_1, model, device=device)
    assert emb_1.shape == (256,), f"Expected shape (256,), got {emb_1.shape}"
    norm_1 = np.linalg.norm(emb_1)
    assert np.isclose(norm_1, 1.0, atol=1e-5), f"Expected norm 1.0, got {norm_1}"
    print(f"   Shape: {emb_1.shape} | L2 Norm: {norm_1:.6f}: PASSED")
    
    # 3. Determinism Check (Repeated Inference)
    print("\n3. Testing Determinism across Repeated Inferences...")
    emb_1_repeat = embed_image(sample_img_1, model, device=device)
    diff = np.max(np.abs(emb_1 - emb_1_repeat))
    assert diff == 0.0, f"Inference is non-deterministic! Max diff: {diff}"
    print(f"   Max absolute difference between runs: {diff:.1e}: PASSED")
    
    # 4. Batch Embedding Check
    print("\n4. Testing Batch Embedding Extraction...")
    sample_paths = [Path(p) for p in orig_samples['image_path'].tolist()]
    batch_embs = embed_images(sample_paths, model, device=device, batch_size=2)
    assert batch_embs.shape == (len(sample_paths), 256), f"Unexpected batch shape: {batch_embs.shape}"
    batch_norms = np.linalg.norm(batch_embs, axis=1)
    assert np.allclose(batch_norms, 1.0, atol=1e-5), "Batch embeddings not L2 normalized!"
    print(f"   Batch shape: {batch_embs.shape} | All row norms == 1.0: PASSED")
    
    # 5. Gallery Build and Load Check
    print("\n5. Testing Gallery Build, Persistence, and Loading...")
    gallery_out = PROJECT_ROOT / "outputs/final/galleries/test_gallery.npz"
    gallery = build_gallery(
        image_paths=sample_paths,
        model=model,
        device=device,
        source_ids=orig_samples['source_id'].tolist(),
        save_path=gallery_out
    )
    loaded_gallery = load_gallery(gallery_out)
    assert loaded_gallery['num_items'] == len(sample_paths), "Gallery item count mismatch!"
    assert np.allclose(gallery['embeddings'], loaded_gallery['embeddings']), "Loaded embeddings mismatch!"
    print(f"   Gallery built and verified ({loaded_gallery['num_items']} items): PASSED")
    
    # 6. Top-K Retrieval Check
    print("\n6. Testing Exact Cosine Top-K Retrieval...")
    retrieval_results = retrieve(sample_img_1, loaded_gallery, top_k=3, model=model, device=device)
    assert len(retrieval_results) == 3, f"Expected 3 results, got {len(retrieval_results)}"
    assert retrieval_results[0]['rank'] == 1, "Top result rank != 1"
    # Query is in the gallery, so top match should be itself with sim ~ 1.0
    assert np.isclose(retrieval_results[0]['similarity'], 1.0, atol=1e-4), f"Self-similarity != 1.0: {retrieval_results[0]['similarity']}"
    print(f"   Top-1 Match: {Path(retrieval_results[0]['image_path']).name} (Similarity: {retrieval_results[0]['similarity']:.4f}): PASSED")
    
    # 7. Pairwise Verification Check
    print("\n7. Testing Pairwise Verification with Frozen Validation Threshold...")
    print(f"   Using Frozen Threshold: {FROZEN_VAL_EER_THRESHOLD:.4f}")
    
    # Positive pair: Original vs its own color variant
    res_pos = verify_pair(sample_img_1, sample_img_1_var, threshold=FROZEN_VAL_EER_THRESHOLD, model=model, device=device)
    print(f"   Same-Source Pair: Similarity={res_pos['similarity']:.4f} -> Prediction={res_pos['prediction']}")
    assert res_pos['prediction'] == 'SAME', "Expected SAME prediction for same-source pair!"
    
    # Negative pair: Original 1 vs Original 2 (distinct sources)
    res_neg = verify_pair(sample_img_1, sample_img_2, threshold=FROZEN_VAL_EER_THRESHOLD, model=model, device=device)
    print(f"   Distinct-Source Pair: Similarity={res_neg['similarity']:.4f} -> Prediction={res_neg['prediction']}")
    assert res_neg['prediction'] == 'DIFFERENT', "Expected DIFFERENT prediction for distinct sources!"
    print("   Pairwise verification checks: PASSED")
    
    # 8. Model Parameter Immutability Check
    print("\n8. Testing Parameter Immutability (No Training Side-Effects)...")
    final_param_hash = get_model_param_hash(model)
    assert initial_param_hash == final_param_hash, "Model parameters were altered during inference!"
    print("   Model parameter hash before and after inference matches identically: PASSED")
    
    # 9. Latency Benchmarking (CPU Environment)
    print("\n9. Measuring Inference Pipeline Latencies...")
    transform = get_inference_transform()
    
    # Single-image embedding latency (10 warmup, 30 timed iterations)
    for _ in range(5):
        _ = embed_image(sample_img_1, model, transform, device)
        
    num_runs = 20
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = embed_image(sample_img_1, model, transform, device)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
    mean_emb_lat = np.mean(latencies)
    std_emb_lat = np.std(latencies)
    print(f"   Single-Image Embedding Latency: {mean_emb_lat:.2f} +/- {std_emb_lat:.2f} ms/image")
    
    # Cosine search latency across gallery (40 items)
    query_vec = emb_1
    search_latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = np.dot(loaded_gallery['embeddings'], query_vec)
        t1 = time.perf_counter()
        search_latencies.append((t1 - t0) * 1000.0)
        
    mean_search_lat = np.mean(search_latencies)
    print(f"   Exact Cosine Retrieval Search Latency (N={loaded_gallery['num_items']}): {mean_search_lat:.4f} ms (embedding-to-gallery similarity computation only, excluding image loading, preprocessing, and neural-network embedding inference)")
    
    print("\n" + "=" * 75)
    print("ALL 9 INFERENCE AND RETRIEVAL SANITY CHECKS PASSED PERFECTLY!")
    print("=" * 75)
    
    return {
        'single_image_latency_ms': float(mean_emb_lat),
        'single_image_latency_std_ms': float(std_emb_lat),
        'gallery_search_latency_ms': float(mean_search_lat),
        'gallery_num_items': int(loaded_gallery['num_items']),
        'threshold_used': float(FROZEN_VAL_EER_THRESHOLD)
    }

if __name__ == '__main__':
    run_sanity_checks()
