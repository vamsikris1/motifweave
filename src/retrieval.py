import os
import sys
from pathlib import Path
from typing import Union, List, Dict, Any
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model, embed_image, embed_images, get_inference_transform

# Frozen Validation-Calibrated Operating Thresholds (Calibrated exclusively in Phase 5)
FROZEN_VAL_EER_THRESHOLD = 0.7588  # Operating point where Val FAR == Val FRR (2.63%)
FROZEN_VAL_MAX_F1_THRESHOLD = 0.8359 # Operating point where Val F1 is maximized (0.7284)

def build_gallery(image_paths: List[Union[str, Path]],
                  model: torch.nn.Module,
                  transform: Any = None,
                  device: torch.device = None,
                  source_ids: List[str] = None,
                  metadata: Dict[str, Any] = None,
                  save_path: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Builds a reference gallery index of textile source images.
    Extracts 256-D L2-normalized embeddings for all gallery images.
    Stores paths, source IDs, embeddings, and preprocessing metadata.
    Does NOT duplicate proprietary image bytes.
    """
    if transform is None:
        transform = get_inference_transform()
    if device is None:
        device = next(model.parameters()).device
        
    print(f"Building gallery index for {len(image_paths)} images...")
    embeddings = embed_images(image_paths, model, transform, device)
    
    gallery = {
        'embeddings': embeddings.astype(np.float32),
        'image_paths': np.array([str(Path(p).resolve()) for p in image_paths]),
        'source_ids': np.array(source_ids if source_ids is not None else [''] * len(image_paths)),
        'embedding_dim': 256,
        'image_size': 224,
        'num_items': len(image_paths),
        'normalization': 'L2_unit_sphere',
        'model_name': 'ConvNeXt-Tiny-GeM-MLP'
    }
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            embeddings=gallery['embeddings'],
            image_paths=gallery['image_paths'],
            source_ids=gallery['source_ids'],
            embedding_dim=gallery['embedding_dim'],
            image_size=gallery['image_size'],
            num_items=gallery['num_items'],
            normalization=gallery['normalization'],
            model_name=gallery['model_name']
        )
        print(f"Gallery index saved cleanly to: {save_path}")
        
    return gallery

def load_gallery(gallery_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a pre-built gallery index from an .npz file.
    """
    gallery_path = Path(gallery_path)
    if not gallery_path.exists():
        raise FileNotFoundError(f"Gallery index file not found at: {gallery_path}")
        
    data = np.load(gallery_path, allow_pickle=True)
    gallery = {
        'embeddings': data['embeddings'],
        'image_paths': data['image_paths'].tolist(),
        'source_ids': data['source_ids'].tolist() if 'source_ids' in data else [],
        'embedding_dim': int(data['embedding_dim']),
        'image_size': int(data['image_size']),
        'num_items': int(data['num_items']),
        'normalization': str(data['normalization']),
        'model_name': str(data['model_name'])
    }
    return gallery

def retrieve(query_input: Union[str, Path, Image.Image, np.ndarray],
             gallery: Union[Dict[str, Any], str, Path],
             top_k: int = 5,
             model: torch.nn.Module = None,
             transform: Any = None,
             device: torch.device = None) -> List[Dict[str, Any]]:
    """
    Retrieves the Top-K most similar images from the reference gallery for exploratory visual retrieval.
    Uses exact cosine similarity dot products: S = query @ gallery.T.
    """
    if isinstance(gallery, (str, Path)):
        gallery = load_gallery(gallery)
        
    if isinstance(query_input, np.ndarray):
        query_emb = query_input
        if query_emb.ndim == 2:
            query_emb = query_emb.squeeze(0)
    else:
        if model is None:
            raise ValueError("Model must be provided when query_input is an image or file path.")
        query_emb = embed_image(query_input, model, transform, device)
        
    # Ensure query embedding is unit-normalized
    query_norm = np.linalg.norm(query_emb)
    if not np.isclose(query_norm, 1.0, atol=1e-4):
        query_emb = query_emb / (query_norm + 1e-12)
        
    # Exact Cosine Similarity search: Dot product with unit vectors
    similarities = np.dot(gallery['embeddings'], query_emb)
    
    # Sort descending
    k = min(top_k, len(similarities))
    top_indices = np.argsort(-similarities)[:k]
    
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            'rank': rank,
            'image_path': gallery['image_paths'][idx],
            'source_id': gallery['source_ids'][idx] if len(gallery['source_ids']) > idx else None,
            'similarity': float(similarities[idx])
        })
        
    return results

def verify_pair(image_a: Union[str, Path, Image.Image],
                image_b: Union[str, Path, Image.Image],
                threshold: float = FROZEN_VAL_EER_THRESHOLD,
                model: torch.nn.Module = None,
                transform: Any = None,
                device: torch.device = None) -> Dict[str, Any]:
    """
    Performs pairwise verification between two saree images:
    Given two images, determine whether they carry the same design.
    Uses frozen validation-calibrated operating threshold.
    """
    if model is None:
        model, _, device = load_model()
    if transform is None:
        transform = get_inference_transform()
    if device is None:
        device = next(model.parameters()).device
        
    emb_a = embed_image(image_a, model, transform, device)
    emb_b = embed_image(image_b, model, transform, device)
    
    similarity = float(np.dot(emb_a, emb_b))
    is_same = bool(similarity >= threshold)
    
    return {
        'similarity': round(similarity, 4),
        'threshold': round(float(threshold), 4),
        'prediction': 'SAME' if is_same else 'DIFFERENT',
        'is_same_design': is_same,
        'threshold_calibration': 'Validation split only (Phase 5 frozen)',
        'image_a': str(image_a) if isinstance(image_a, (str, Path)) else 'PIL.Image',
        'image_b': str(image_b) if isinstance(image_b, (str, Path)) else 'PIL.Image'
    }
