#!/usr/bin/env python3
"""
Simplified download script for debugging.
"""

print("SCRIPT STARTED")

from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np

print("Imports successful")

# Configuration
output_dir = "data/test_download"
num_replays = 5
seed = 42

print(f"Configuration: {num_replays} replays to {output_dir}")

# Create directories
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)
print(f"Created directory: {output_path}")

for split in ['train', 'val', 'test']:
    (output_path / split).mkdir(parents=True, exist_ok=True)
print("Created split directories")

# Set seed
np.random.seed(seed)
print(f"Set random seed: {seed}")

# Load dataset
print("Connecting to HuggingFace...")
dataset_stream = load_dataset("strakammm/generals_io_replays", split="train", streaming=True)
print("Connected!")

# Download replays
print(f"Downloading {num_replays} replays...")
replays = []
for i, replay in enumerate(dataset_stream):
    replays.append(replay)
    print(f"  Downloaded replay {i+1}/{num_replays}")
    if i + 1 >= num_replays:
        break

print(f"\nTotal replays downloaded: {len(replays)}")

# Create splits
indices = np.random.permutation(len(replays))
train_size = int(len(replays) * 0.8)
val_size = int(len(replays) * 0.1)

splits = {
    'train': indices[:train_size].tolist(),
    'val': indices[train_size:train_size + val_size].tolist(),
    'test': indices[train_size + val_size:].tolist()
}

print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

# Create mapping
split_map = {}
for split_name, idx_list in splits.items():
    split_map.update({idx: split_name for idx in idx_list})

# Save replays
print("Saving replays to disk...")
for idx in range(len(replays)):
    replay = replays[idx]
    split_name = split_map[idx]
    replay_path = output_path / split_name / f"replay_{idx:06d}.json"
    with open(replay_path, 'w') as f:
        json.dump(replay, f)
    print(f"  Saved to {split_name}/replay_{idx:06d}.json")

print("\n✅ DOWNLOAD COMPLETE!")
print(f"Files saved to: {output_dir}")
