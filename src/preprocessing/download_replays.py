#!/usr/bin/env python3
"""
Download Generals.io replays from Hugging Face.

Uses streaming mode to avoid loading the entire 347k dataset into memory.
"""

import sys
import json
import argparse
from pathlib import Path
from datasets import load_dataset
import numpy as np


def download_replays(output_dir, num_replays, seed=42):
    """Download and split replays."""
    
    print("="*60, flush=True)
    print("🎮 Downloading Generals.io Replays from Hugging Face", flush=True)
    print("="*60, flush=True)
    print(f"Dataset: strakammm/generals_io_replays", flush=True)
    print(f"Target: {num_replays:,} replays", flush=True)
    print(f"Output: {output_dir}\n", flush=True)
    
    # Create directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directories", flush=True)
    
    # Set seed
    np.random.seed(seed)
    print(f"✓ Set random seed: {seed}\n", flush=True)
    
    # Connect to HuggingFace
    print("📥 Connecting to HuggingFace dataset...", flush=True)
    dataset_stream = load_dataset(
        "strakammm/generals_io_replays",
        split="train",
        streaming=True
    )
    print("✓ Connected!\n", flush=True)
    
    # Download replays
    print(f"⏬ Downloading {num_replays:,} replays...", flush=True)
    replays = []
    for i, replay in enumerate(dataset_stream):
        replays.append(replay)
        if (i + 1) % 100 == 0:
            print(f"  Downloaded {i+1:,} replays...", flush=True)
        if i + 1 >= num_replays:
            break
    
    print(f"\n✓ Downloaded {len(replays):,} replays\n", flush=True)
    
    # Create splits
    indices = np.random.permutation(len(replays))
    train_size = int(len(replays) * 0.8)
    val_size = int(len(replays) * 0.1)
    
    splits = {
        'train': indices[:train_size].tolist(),
        'val': indices[train_size:train_size + val_size].tolist(),
        'test': indices[train_size + val_size:].tolist()
    }
    
    print("📊 Dataset splits:", flush=True)
    print(f"   Train: {len(splits['train']):,} replays (80%)", flush=True)
    print(f"   Val:   {len(splits['val']):,} replays (10%)", flush=True)
    print(f"   Test:  {len(splits['test']):,} replays (10%)\n", flush=True)
    
    # Save split indices
    splits_file = output_path / "splits.json"
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"✓ Saved split indices to {splits_file}\n", flush=True)
    
    # Create mapping
    split_map = {}
    for split_name, idx_list in splits.items():
        split_map.update({idx: split_name for idx in idx_list})
    
    # Save replays
    print("💾 Saving replays to disk...", flush=True)
    for idx in range(len(replays)):
        replay = replays[idx]
        split_name = split_map[idx]
        replay_path = output_path / split_name / f"replay_{idx:06d}.json"
        with open(replay_path, 'w') as f:
            json.dump(replay, f)
        if (idx + 1) % 100 == 0 or idx == len(replays) - 1:
            print(f"  Saved {idx+1:,}/{len(replays):,} replays...", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("✅ Successfully downloaded and split replays!", flush=True)
    print("="*60, flush=True)
    print(f"📁 Data saved to: {output_dir}", flush=True)
    print(f"   Train: {len(splits['train']):,} files in train/", flush=True)
    print(f"   Val:   {len(splits['val']):,} files in val/", flush=True)
    print(f"   Test:  {len(splits['test']):,} files in test/", flush=True)
    print("="*60, flush=True)
    
    return len(replays), splits


def main():
    parser = argparse.ArgumentParser(
        description="Download Generals.io replays from Hugging Face"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to save replays (default: data/raw)"
    )
    parser.add_argument(
        "--num_replays",
        type=int,
        default=50000,
        help="Number of replays to download (default: 50,000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    try:
        download_replays(
            output_dir=args.output_dir,
            num_replays=args.num_replays,
            seed=args.seed
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
