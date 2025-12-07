#!/usr/bin/env python3
"""
Quick script to check BC training progress.
Run this periodically to see how training is going.
"""

import json
from pathlib import Path
from datetime import datetime

def check_training_progress():
    print("=" * 60)
    print("BEHAVIOR CLONING TRAINING STATUS")
    print("=" * 60)
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check for checkpoint
    checkpoint_dir = Path("checkpoints/bc")
    
    latest_model = checkpoint_dir / "latest_model.pt"
    best_model = checkpoint_dir / "best_model.pt"
    results_file = checkpoint_dir / "training_results.json"
    
    if latest_model.exists():
        import torch
        checkpoint = torch.load(latest_model, map_location='cpu')
        print(f"✓ Latest checkpoint found!")
        print(f"  - Epoch: {checkpoint['epoch']}")
        print(f"  - Validation Loss: {checkpoint['val_loss']:.4f}")
        print(f"  - Last modified: {datetime.fromtimestamp(latest_model.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("⏳ No checkpoint yet (training just started)")
    
    print()
    
    if best_model.exists():
        checkpoint = torch.load(best_model, map_location='cpu')
        print(f"✓ Best model found!")
        print(f"  - Best Validation Loss: {checkpoint['val_loss']:.4f}")
        print(f"  - Achieved at Epoch: {checkpoint['epoch']}")
    else:
        print("⏳ No best model saved yet")
    
    print()
    
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        print("✅ TRAINING COMPLETE!")
        print(f"  - Best Validation Loss: {results['best_val_loss']:.4f}")
        print(f"  - Total Epochs: {results['total_epochs']}")
        print(f"  - Training Samples: {results['config']['num_train_samples']:,}")
        print(f"  - Validation Samples: {results['config']['num_val_samples']:,}")
    else:
        print("🔄 Training in progress...")
        print("\nTo view real-time progress:")
        print("  tensorboard --logdir logs/bc")
        print("  (Then open http://localhost:6006 in browser)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_training_progress()
