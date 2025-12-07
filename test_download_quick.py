#!/usr/bin/env python3
"""Quick test of download functionality."""

import sys
sys.path.insert(0, 'src')

from preprocessing.download_replays import download_replays

print("Starting quick download test...", flush=True)
try:
    num_downloaded, splits = download_replays(
        output_dir="data/test_100",
        num_replays=100,
        seed=42
    )
    print(f"\n✅ SUCCESS! Downloaded {num_downloaded} replays", flush=True)
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}", flush=True)
except Exception as e:
    print(f"\n❌ ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
