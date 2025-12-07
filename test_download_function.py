#!/usr/bin/env python3
"""Quick test of download functionality."""

import sys
sys.path.insert(0, 'src/preprocessing')

print("Starting test...")

try:
    from download_replays import download_replays
    print("✓ Import successful")
    
    print("\nCalling download_replays...")
    result = download_replays(
        output_dir="data/test_raw",
        num_replays=5,
        seed=42
    )
    print(f"\n✓ Download complete: {result}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
