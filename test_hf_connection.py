#!/usr/bin/env python3
"""Test HuggingFace dataset connection with timeout."""

import sys
from datasets import load_dataset

print("Testing HuggingFace connection...", flush=True)
print("This should take 5-10 seconds...", flush=True)

try:
    print("Calling load_dataset with streaming=True...", flush=True)
    dataset_stream = load_dataset(
        "strakammm/generals_io_replays",
        split="train",
        streaming=True
    )
    print("✓ Connected!", flush=True)
    
    print("Fetching first replay...", flush=True)
    first_replay = next(iter(dataset_stream))
    print(f"✓ Got first replay: {list(first_replay.keys())}", flush=True)
    print(f"  ID: {first_replay.get('id', 'N/A')}", flush=True)
    print("\n✅ Connection test successful!", flush=True)
    
except Exception as e:
    print(f"\n❌ Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
