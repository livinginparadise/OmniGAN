import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from denoisegan.models import DenoiseGenerator, UNetDiscriminatorSN, DualDiscriminator

def check_model_for_nan(model, model_name="Model"):
    """Check if model contains any NaN values."""
    has_nan = False
    nan_params = []

    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            has_nan = True
            nan_count = torch.isnan(param).sum().item()
            nan_params.append((name, nan_count, param.numel()))

    if has_nan:
        print(f"❌ {model_name} contains NaN values!")
        print(f"\nAffected parameters:")
        for param_name, nan_count, total in nan_params:
            pct = 100 * nan_count / total
            print(f"  {param_name}: {nan_count}/{total} ({pct:.2f}%)")
        return False
    else:
        print(f"✓ {model_name} is clean (no NaN values)")
        return True

def load_checkpoint(checkpoint_path):
    """Load a checkpoint and check all models."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    results = {}

    if 'G' in ckpt:
        print("\n📊 Checking Generator...")
        G = DenoiseGenerator()
        G.load_state_dict(ckpt['G'], strict=False)
        results['G'] = check_model_for_nan(G, "Generator")

    if 'D' in ckpt:
        print("\n📊 Checking Discriminator...")
        D = UNetDiscriminatorSN()
        D.load_state_dict(ckpt['D'], strict=False)
        results['D'] = check_model_for_nan(D, "Discriminator")

    print("\n" + "="*50)
    if all(results.values()):
        print("✓ All models are clean!")
    else:
        print("⚠ Some models contain NaN values")

    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model_nan.py <checkpoint_path>")
        print("Example: python check_model_nan.py checkpoints/model_latest.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    load_checkpoint(checkpoint_path)
