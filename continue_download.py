#!/usr/bin/env python3
"""
Continue downloading replays from 18.8k to 50k.
"""

import json
from pathlib import Path
from datasets import load_dataset
import numpy as np

# Configuration
OUTPUT_DIR = Path("data/raw")
TARGET_TOTAL = 50_000  # Total target
CURRENT_COUNT = 18_803  # Already have this many
TO_DOWNLOAD = TARGET_TOTAL - CURRENT_COUNT  # Need to download this many more
SEED = 42

print("="*60)
print("🎮 Continue Downloading Generals.io Replays")
print("="*60)
print(f"✓ Already downloaded: {CURRENT_COUNT:,} replays")
print(f"🎯 Target total: {TARGET_TOTAL:,} replays")
print(f"⏬ Need to download: {TO_DOWNLOAD:,} more replays\n")

# Load existing split info
splits_file = OUTPUT_DIR / "splits.json"
with open(splits_file, 'r') as f:
    old_splits = json.load(f)

print(f"Current splits:")
print(f"  Train: {len(old_splits['train']):,}")
print(f"  Val:   {len(old_splits['val']):,}")
print(f"  Test:  {len(old_splits['test']):,}\n")

# Connect to dataset
print("📥 Connecting to HuggingFace dataset...")
dataset_stream = load_dataset(
    "strakammm/generals_io_replays",
    split="train",
    streaming=True
)
print("✓ Connected!\n")

# Skip already downloaded replays
print(f"⏭️  Skipping first {CURRENT_COUNT:,} replays...")
dataset_iter = iter(dataset_stream)
for i in range(CURRENT_COUNT):
    next(dataset_iter)
    if (i + 1) % 1000 == 0:
        print(f"  Skipped {i+1:,}...")
print("✓ Skip complete\n")

# Download remaining replays
print(f"⏬ Downloading {TO_DOWNLOAD:,} more replays...")
new_replays = []
for i, replay in enumerate(dataset_iter):
    new_replays.append(replay)
    if (i + 1) % 500 == 0:
        print(f"  Downloaded {i+1:,}/{TO_DOWNLOAD:,}...")
    if i + 1 >= TO_DOWNLOAD:
        break

print(f"\n✓ Downloaded {len(new_replays):,} new replays\n")

# Create new indices for these replays
np.random.seed(SEED)
new_indices = np.arange(CURRENT_COUNT, CURRENT_COUNT + len(new_replays))
np.random.shuffle(new_indices)

# Split new replays (80/10/10)
train_size = int(len(new_indices) * 0.8)
val_size = int(len(new_indices) * 0.1)

new_splits = {
    'train': new_indices[:train_size].tolist(),
    'val': new_indices[train_size:train_size + val_size].tolist(),
    'test': new_indices[train_size + val_size:].tolist()
}

# Merge with old splits
merged_splits = {
    'train': old_splits['train'] + new_splits['train'],
    'val': old_splits['val'] + new_splits['val'],
    'test': old_splits['test'] + new_splits['test']
}

print("📊 Updated splits:")
print(f"  Train: {len(merged_splits['train']):,} (+{len(new_splits['train']):,})")
print(f"  Val:   {len(merged_splits['val']):,} (+{len(new_splits['val']):,})")
print(f"  Test:  {len(merged_splits['test']):,} (+{len(new_splits['test']):,})\n")

# Save updated splits
with open(splits_file, 'w') as f:
    json.dump(merged_splits, f, indent=2)
print(f"✓ Updated {splits_file}\n")

# Create split mapping for new replays
split_map = {}
for split_name, idx_list in new_splits.items():
    split_map.update({idx: split_name for idx in idx_list})

# Save new replays
print("💾 Saving new replays to disk...")
for i, replay in enumerate(new_replays):
    idx = CURRENT_COUNT + i
    split_name = split_map[idx]
    replay_path = OUTPUT_DIR / split_name / f"replay_{idx:06d}.json"
    with open(replay_path, 'w') as f:
        json.dump(replay, f)
    if (i + 1) % 500 == 0 or i == len(new_replays) - 1:
        print(f"  Saved {i+1:,}/{len(new_replays):,}...")

print("\n" + "="*60)
print("✅ Download complete!")
print("="*60)
print(f"📁 Total replays: {CURRENT_COUNT + len(new_replays):,}")
print(f"   Train: {len(merged_splits['train']):,}")
print(f"   Val:   {len(merged_splits['val']):,}")
print(f"   Test:  {len(merged_splits['test']):,}")
print("="*60)
