import os
import sys
import argparse
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model, get_inference_transform
from src.retrieval import verify_pair, FROZEN_VAL_EER_THRESHOLD, FROZEN_VAL_MAX_F1_THRESHOLD

def parse_args():
    parser = argparse.ArgumentParser(description="MotifWeave Pairwise Saree Design Verification")
    parser.add_argument("--image-a", type=str, required=True,
                        help="Path to first saree image")
    parser.add_argument("--image-b", type=str, required=True,
                        help="Path to second saree image")
    parser.add_argument("--threshold", type=float, default=FROZEN_VAL_EER_THRESHOLD,
                        help=f"Cosine similarity threshold (default: {FROZEN_VAL_EER_THRESHOLD} from frozen validation EER)")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/baseline_best.pth",
                        help="Trained checkpoint path")
    return parser.parse_args()

def main():
    args = parse_args()
    path_a = Path(args.image_a)
    path_b = Path(args.image_b)
    checkpoint_path = PROJECT_ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    
    if not path_a.exists():
        print(f"Error: Image A not found at: {path_a}")
        sys.exit(1)
    if not path_b.exists():
        print(f"Error: Image B not found at: {path_b}")
        sys.exit(1)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config, device = load_model(checkpoint_path, device=device)
    transform = get_inference_transform(config.get('image_size', 224))
    
    result = verify_pair(path_a, path_b, threshold=args.threshold, model=model, transform=transform, device=device)
    
    print("\n" + "=" * 65)
    print("MOTIFWEAVE: PAIRWISE TEXTILE DESIGN VERIFICATION")
    print("=" * 65)
    print(f"Image A:            {path_a.name}")
    print(f"Image B:            {path_b.name}")
    print("-" * 65)
    print(f"Cosine Similarity:  {result['similarity']:.4f}")
    print(f"Decision Threshold: {result['threshold']:.4f}")
    print(f"Threshold Origin:   {result['threshold_calibration']}")
    print("-" * 65)
    print(f"VERIFICATION RESULT: {result['prediction']}")
    print("=" * 65)
    if result['is_same_design']:
        print("Note: Similarity meets threshold for same-design classification.")
    else:
        print("Note: Similarity below threshold; classified as distinct designs.")
    print("=" * 65 + "\n")

if __name__ == '__main__':
    main()
