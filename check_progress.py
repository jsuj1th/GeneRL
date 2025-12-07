#!/usr/bin/env python3
"""
Check preprocessing progress.
"""

import numpy as np
from pathlib import Path

print("="*60)
print("📊 PREPROCESSING PROGRESS CHECK")
print("="*60)

# Check raw data
raw_dir = Path("data/raw")
print("\n📁 RAW DATA (Downloaded):")
for split in ['train', 'val', 'test']:
    split_dir = raw_dir / split
    if split_dir.exists():
        replay_files = list(split_dir.glob("replay_*.json"))
        print(f"  {split:5s}: {len(replay_files):6,} replays")
    else:
        print(f"  {split:5s}: Directory not found")

# Check processed data
processed_dir = Path("data/processed")
print("\n📁 PROCESSED DATA (Training Ready):")
for split in ['train', 'val', 'test']:
    split_file = processed_dir / split / "data.npz"
    if split_file.exists():
        data = np.load(split_file)
        print(f"  {split:5s}: {len(data['states']):6,} examples")
        print(f"         Shape: states={data['states'].shape}, actions={data['actions'].shape}")
    else:
        print(f"  {split:5s}: Not yet processed")

print("\n" + "="*60)

# Check if preprocessing is complete
train_file = processed_dir / "train" / "data.npz"
val_file = processed_dir / "val" / "data.npz"
test_file = processed_dir / "test" / "data.npz"

if train_file.exists() and val_file.exists() and test_file.exists():
    print("✅ PREPROCESSING COMPLETE!")
    print("▶️  Next step: python src/training/train_bc.py")
else:
    print("⏳ PREPROCESSING IN PROGRESS...")
    print("▶️  Run: python src/preprocessing/preprocess_replays.py")

print("="*60)
