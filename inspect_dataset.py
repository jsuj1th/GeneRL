"""
Inspect the Hugging Face Generals.io dataset structure.

This script downloads a small sample and prints the format,
so we can update our preprocessing code accordingly.
"""

from datasets import load_dataset
import json


def inspect_dataset():
    """Download and inspect a few samples from the dataset."""
    
    print("=" * 80)
    print("Inspecting Generals.io Replays Dataset")
    print("Dataset: strakammm/generals_io_replays")
    print("=" * 80)
    
    try:
        # Download just 5 samples to inspect
        print("\nDownloading 5 sample replays...")
        dataset = load_dataset(
            "strakammm/generals_io_replays",
            split="train[:5]",
            streaming=False
        )
        
        print(f"✓ Downloaded {len(dataset)} samples\n")
        
        # Print dataset info
        print("Dataset features:")
        print("-" * 80)
        for feature_name, feature_type in dataset.features.items():
            print(f"  {feature_name}: {feature_type}")
        
        # Inspect first replay in detail
        print("\n" + "=" * 80)
        print("First Replay Structure:")
        print("=" * 80)
        
        replay = dataset[0]
        
        print("\nTop-level keys:")
        for key in replay.keys():
            print(f"  - {key}: {type(replay[key])}")
            if isinstance(replay[key], (list, dict)):
                if isinstance(replay[key], list):
                    print(f"    Length: {len(replay[key])}")
                    if len(replay[key]) > 0:
                        print(f"    First element type: {type(replay[key][0])}")
                        if isinstance(replay[key][0], dict):
                            print(f"    First element keys: {list(replay[key][0].keys())}")
                elif isinstance(replay[key], dict):
                    print(f"    Keys: {list(replay[key].keys())}")
        
        # Print full structure of first replay (formatted JSON)
        print("\n" + "=" * 80)
        print("First Replay (Full JSON):")
        print("=" * 80)
        print(json.dumps(replay, indent=2, default=str))
        
        # Print statistics about replay lengths
        print("\n" + "=" * 80)
        print("Replay Statistics (from 5 samples):")
        print("=" * 80)
        
        for i, replay in enumerate(dataset):
            num_states = len(replay.get('states', []))
            num_actions = len(replay.get('actions', []))
            print(f"Replay {i}: {num_states} states, {num_actions} actions")
        
        print("\n✓ Inspection complete!")
        print("\nNext steps:")
        print("  1. Update download_replays.py based on this structure")
        print("  2. Run: python src/preprocessing/download_replays.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have installed the required packages:")
        print("  pip install datasets huggingface-hub")


if __name__ == "__main__":
    inspect_dataset()
