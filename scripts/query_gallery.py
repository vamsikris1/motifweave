import os
import sys
import argparse
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model, get_inference_transform
from src.retrieval import load_gallery, retrieve

def parse_args():
    parser = argparse.ArgumentParser(description="Query MotifWeave Reference Gallery for Top-K Matches")
    parser.add_argument("--query", type=str, required=True,
                        help="Path to query saree image")
    parser.add_argument("--gallery", type=str, default="outputs/final/galleries/test_gallery.npz",
                        help="Path to gallery index (.npz)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top candidates to retrieve")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/baseline_best.pth",
                        help="Trained checkpoint path")
    return parser.parse_args()

def main():
    args = parse_args()
    query_path = Path(args.query)
    gallery_path = PROJECT_ROOT / args.gallery if not Path(args.gallery).is_absolute() else Path(args.gallery)
    checkpoint_path = PROJECT_ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    
    if not query_path.exists():
        print(f"Error: Query image not found at: {query_path}")
        sys.exit(1)
    if not gallery_path.exists():
        print(f"Error: Gallery index not found at: {gallery_path}")
        print("Please build a gallery index first using `scripts/build_gallery.py`.")
        sys.exit(1)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config, device = load_model(checkpoint_path, device=device)
    transform = get_inference_transform(config.get('image_size', 224))
    
    gallery = load_gallery(gallery_path)
    results = retrieve(query_path, gallery, top_k=args.top_k, model=model, transform=transform, device=device)
    
    print("\n" + "=" * 78)
    print("MOTIFWEAVE: TOP-K TEXTILE DESIGN RETRIEVAL RESULTS")
    print("=" * 78)
    print(f"Query Image:       {query_path.name}")
    print(f"Reference Gallery: {gallery_path.name} ({gallery['num_items']} reference images)")
    print(f"Top-K Requested:   {args.top_k}")
    print("-" * 78)
    print(f"{'Rank':<5} | {'Similarity':<10} | {'Source ID':<15} | {'Gallery Image'}")
    print("-" * 78)
    for res in results:
        img_name = Path(res['image_path']).name
        src_id = res['source_id'] if res['source_id'] else "unknown"
        print(f"{res['rank']:<5} | {res['similarity']:<10.4f} | {src_id:<15} | {img_name}")
    print("=" * 78)
    print("Caveat: These are exploratory visual retrieval matches on the reference gallery.")
    print("=" * 78 + "\n")

if __name__ == '__main__':
    main()
