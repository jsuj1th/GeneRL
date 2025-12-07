#!/usr/bin/env python3
"""
Simple script to download 50k replays - designed to work reliably after restart.
"""

import json
from pathlib import Path
from datasets import load_dataset
import numpy as np

# Configuration
OUTPUT_DIR = "data/raw"
NUM_REPLAYS = 50000  # Change to smaller number for testing (e.g., 100)
SEED = 42

print("="*70)
print("🎮 GENERALS.IO REPLAY DOWNLOADER")
print("="*70)
print(f"Target: {NUM_REPLAYS:,} replays")
print(f"Output: {OUTPUT_DIR}")
print("="*70)

# Create directories
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
for split in ['train', 'val', 'test']:
    (output_path / split).mkdir(parents=True, exist_ok=True)
print("✓ Directories created")

# Set seed
np.random.seed(SEED)
print(f"✓ Random seed set: {SEED}")

# Connect to HuggingFace
print("\n📥 Connecting to HuggingFace dataset...")
dataset = load_dataset(
    "strakammm/generals_io_replays",
    split="train",
    streaming=True
)
print("✓ Connected!")

# Download replays
print(f"\n⏬ Downloading {NUM_REPLAYS:,} replays (this will take time)...")
replays = []
for i, replay in enumerate(dataset):
    replays.append(replay)
    
    # Progress updates
    if (i + 1) % 500 == 0:
        print(f"   {i+1:,} downloaded...")
    elif (i + 1) % 100 == 0 and i < 1000:
        print(f"   {i+1:,} downloaded...")
    
    if i + 1 >= NUM_REPLAYS:
        break

print(f"\n✓ Downloaded {len(replays):,} replays")

# Create splits (80/10/10)
print("\n📊 Creating train/val/test splits...")
indices = np.random.permutation(len(replays))
train_size = int(len(replays) * 0.8)
val_size = int(len(replays) * 0.1)

splits = {
    'train': indices[:train_size].tolist(),
    'val': indices[train_size:train_size + val_size].tolist(),
    'test': indices[train_size + val_size:].tolist()
}

print(f"   Train: {len(splits['train']):,} replays (80%)")
print(f"   Val:   {len(splits['val']):,} replays (10%)")
print(f"   Test:  {len(splits['test']):,} replays (10%)")

# Save split indices
with open(output_path / "splits.json", 'w') as f:
    json.dump(splits, f, indent=2)
print("✓ Split indices saved")

# Create split mapping
split_map = {}
for split_name, idx_list in splits.items():
    for idx in idx_list:
        split_map[idx] = split_name

# Save replays to disk
print(f"\n💾 Saving {len(replays):,} replays to disk (this will take time)...")
for idx in range(len(replays)):
    replay = replays[idx]
    split_name = split_map[idx]
    filepath = output_path / split_name / f"replay_{idx:06d}.json"
    
    with open(filepath, 'w') as f:
        json.dump(replay, f)
    
    # Progress updates
    if (idx + 1) % 1000 == 0:
        print(f"   {idx+1:,}/{len(replays):,} saved...")
    elif (idx + 1) % 100 == 0 and idx < 1000:
        print(f"   {idx+1:,}/{len(replays):,} saved...")

print("\n" + "="*70)
print("✅ DOWNLOAD COMPLETE!")
print("="*70)
print(f"📁 Location: {OUTPUT_DIR}/")
print(f"   train/ - {len(splits['train']):,} files")
print(f"   val/   - {len(splits['val']):,} files")
print(f"   test/  - {len(splits['test']):,} files")
print("="*70)
