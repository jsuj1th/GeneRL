#!/usr/bin/env python3
"""Minimal download test - inline version."""

import sys
import json
from pathlib import Path
from datasets import load_dataset

print("Starting minimal download test...", flush=True)

# Setup
output_dir = Path("data/test_minimal")
output_dir.mkdir(parents=True, exist_ok=True)
num_replays = 10

print(f"Downloading {num_replays} replays to {output_dir}...", flush=True)

# Connect and download
dataset_stream = load_dataset(
    "strakammm/generals_io_replays",
    split="train",
    streaming=True
)
print("Connected to dataset!", flush=True)

replays = []
for i, replay in enumerate(dataset_stream):
    replays.append(replay)
    print(f"  Downloaded {i+1}/{num_replays}...", flush=True)
    if i + 1 >= num_replays:
        break

print(f"\nSaving {len(replays)} replays...", flush=True)
for i, replay in enumerate(replays):
    filepath = output_dir / f"replay_{i:03d}.json"
    with open(filepath, 'w') as f:
        json.dump(replay, f)
    print(f"  Saved {i+1}/{len(replays)}", flush=True)

print(f"\n✅ SUCCESS! Downloaded {len(replays)} replays to {output_dir}", flush=True)
