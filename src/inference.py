import os
import sys
from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model

DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/checkpoints/baseline_best.pth"

def get_inference_transform(image_size: int = 224) -> T.Compose:
    """
    Exact preprocessing transform required by the trained MotifWeave model.
    Bicubic resize to (224, 224) and standard ImageNet normalization.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

def load_model(checkpoint_path: Union[str, Path] = None, device: torch.device = None) -> Tuple[torch.nn.Module, dict, torch.device]:
    """
    Loads the trained MotifWeave baseline embedding model for inference.
    Ensures model is in eval() mode and all parameters are strictly non-trainable.
    """
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get('config', {})
    
    model = build_model(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Freeze all parameters explicitly for safety
    for param in model.parameters():
        param.requires_grad = False
        
    return model, config, device

def embed_image(image_input: Union[str, Path, Image.Image, torch.Tensor],
                model: torch.nn.Module,
                transform: T.Compose = None,
                device: torch.device = None) -> np.ndarray:
    """
    Extracts a 256-D unit-normalized embedding vector for a single image.
    Returns:
        np.ndarray of shape (256,) with L2 norm == 1.0.
    """
    if device is None:
        device = next(model.parameters()).device
        
    if transform is None:
        transform = get_inference_transform()
        
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")
        with Image.open(image_path) as img:
            img_rgb = img.convert('RGB')
            tensor = transform(img_rgb).unsqueeze(0).to(device)
    elif isinstance(image_input, Image.Image):
        tensor = transform(image_input.convert('RGB')).unsqueeze(0).to(device)
    elif isinstance(image_input, torch.Tensor):
        if image_input.dim() == 3:
            tensor = image_input.unsqueeze(0).to(device)
        else:
            tensor = image_input.to(device)
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")
        
    with torch.inference_mode():
        emb = model(tensor)
        emb_norm = torch.norm(emb, p=2, dim=1)
        assert torch.allclose(emb_norm, torch.ones_like(emb_norm), atol=1e-5), "Embedding not L2 normalized!"
        emb_np = emb.squeeze(0).cpu().numpy().astype(np.float32)
        
    return emb_np

def embed_images(image_inputs: List[Union[str, Path, Image.Image]],
                 model: torch.nn.Module,
                 transform: T.Compose = None,
                 device: torch.device = None,
                 batch_size: int = 32) -> np.ndarray:
    """
    Extracts 256-D unit-normalized embedding vectors for a batch of images.
    Returns:
        np.ndarray of shape (N, 256) where each row has L2 norm == 1.0.
    """
    if device is None:
        device = next(model.parameters()).device
        
    if transform is None:
        transform = get_inference_transform()
        
    all_embeddings = []
    
    with torch.inference_mode():
        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i:i + batch_size]
            batch_tensors = []
            for item in batch:
                if isinstance(item, (str, Path)):
                    with Image.open(item) as img:
                        batch_tensors.append(transform(img.convert('RGB')))
                elif isinstance(item, Image.Image):
                    batch_tensors.append(transform(item.convert('RGB')))
                else:
                    raise TypeError(f"Unsupported batch item type: {type(item)}")
                    
            batch_tensor = torch.stack(batch_tensors).to(device)
            embs = model(batch_tensor)
            all_embeddings.append(embs.cpu().numpy().astype(np.float32))
            
    embeddings_matrix = np.concatenate(all_embeddings, axis=0)
    norms = np.linalg.norm(embeddings_matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Batch embeddings not L2 normalized!"
    
    return embeddings_matrix
