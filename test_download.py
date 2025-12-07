#!/usr/bin/env python3
"""
Test script to debug HuggingFace dataset download.
"""

import sys
print("="*60)
print("Testing HuggingFace Dataset Download")
print("="*60)

print("\n1. Testing imports...")
try:
    from datasets import load_dataset
    print("✓ datasets imported")
except Exception as e:
    print(f"✗ Failed to import datasets: {e}")
    sys.exit(1)

print("\n2. Attempting to load dataset...")
try:
    dataset = load_dataset("strakammm/generals_io_replays", split="train", streaming=True)
    print("✓ Dataset loaded in streaming mode")
    
    print("\n3. Fetching first replay...")
    first_replay = next(iter(dataset))
    print("✓ First replay retrieved")
    
    print("\n4. Replay structure:")
    print(f"Type: {type(first_replay)}")
    print(f"Keys: {first_replay.keys() if hasattr(first_replay, 'keys') else 'No keys'}")
    
    if hasattr(first_replay, 'keys'):
        print("\n5. Replay content sample:")
        for key, value in first_replay.items():
            if isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if len(value) > 0:
                    print(f"    First item type: {type(value[0])}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())[:5]}")
            else:
                print(f"  {key}: {type(value).__name__} = {str(value)[:100]}")
    
    print("\n✅ Dataset is accessible!")
    
except Exception as e:
    print(f"\n❌ Error loading dataset: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
