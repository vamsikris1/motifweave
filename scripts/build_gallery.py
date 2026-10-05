import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model, get_inference_transform
from src.retrieval import build_gallery

def parse_args():
    parser = argparse.ArgumentParser(description="Build MotifWeave Reference Gallery Index")
    parser.add_argument("--manifest", type=str, default="data/processed/splits/test.csv",
                        help="Path to manifest or split CSV file")
    parser.add_argument("--split", type=str, default="test",
                        help="Split filter ('train', 'val', 'test', or 'all')")
    parser.add_argument("--variant-type", type=str, default="original",
                        help="Variant type filter ('original', 'all', or specific variant)")
    parser.add_argument("--output", type=str, default="outputs/final/galleries/test_gallery.npz",
                        help="Output path for gallery index (.npz)")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/baseline_best.pth",
                        help="Trained checkpoint path")
    return parser.parse_args()

def main():
    args = parse_args()
    manifest_path = PROJECT_ROOT / args.manifest
    output_path = PROJECT_ROOT / args.output
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    
    print("=" * 70)
    print("MOTIFWEAVE: BUILDING REFERENCE GALLERY INDEX")
    print("=" * 70)
    print(f"Manifest Source:   {manifest_path}")
    print(f"Model Checkpoint:  {checkpoint_path}")
    print(f"Split Filter:      {args.split}")
    print(f"Variant Filter:    {args.variant_type}")
    
    df = pd.read_csv(manifest_path)
    
    if args.split != 'all' and 'split' in df.columns:
        df = df[df['split'] == args.split]
    if args.variant_type != 'all' and 'variant_type' in df.columns:
        df = df[df['variant_type'] == args.variant_type]
        
    image_paths = df['image_path'].tolist()
    source_ids = df['source_id'].tolist() if 'source_id' in df.columns else None
    
    print(f"Filtered Gallery Items: {len(image_paths)} images")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config, device = load_model(checkpoint_path, device=device)
    transform = get_inference_transform(config.get('image_size', 224))
    
    gallery = build_gallery(
        image_paths=image_paths,
        model=model,
        transform=transform,
        device=device,
        source_ids=source_ids,
        save_path=output_path
    )
    
    print("-" * 70)
    print(f"Successfully constructed reference gallery with {gallery['num_items']} items.")
    print(f"Index persisted to: {output_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
