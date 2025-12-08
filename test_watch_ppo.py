"""Quick test to verify PPO checkpoint loading works correctly."""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from models.networks import GeneralsAgent

def test_ppo_checkpoint_loading():
    """Test that we can load PPO checkpoint correctly."""
    
    checkpoint_path = "checkpoints/ppo_from_bc/latest_model.pt"
    device = torch.device("cpu")
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check checkpoint type
    is_ppo = 'policy_state_dict' in checkpoint
    print(f"✅ Checkpoint type: {'PPO' if is_ppo else 'BC'}")
    
    if is_ppo:
        print(f"   Episode: {checkpoint.get('episode', 'N/A')}")
        print(f"   Total steps: {checkpoint.get('total_steps', 'N/A'):,}")
        print(f"   Best win rate: {checkpoint.get('best_win_rate', 0.0):.1%}")
    
    # Check keys in policy_state_dict
    policy_keys = list(checkpoint['policy_state_dict'].keys())
    print(f"\n📋 First 5 keys in policy_state_dict:")
    for key in policy_keys[:5]:
        print(f"   - {key}")
    
    # Create network
    config = Config()
    network = GeneralsAgent(
        num_channels=config.NUM_CHANNELS,
        num_actions=config.NUM_ACTIONS,
        cnn_channels=config.CNN_CHANNELS
    ).to(device)
    
    print(f"\n🔧 Network architecture keys (first 5):")
    network_keys = list(network.state_dict().keys())
    for key in network_keys[:5]:
        print(f"   - {key}")
    
    # Extract weights by removing "backbone." prefix
    print(f"\n🔄 Extracting weights from PPO checkpoint...")
    full_state_dict = checkpoint['policy_state_dict']
    unwrapped_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith('backbone.'):
            new_key = key.replace('backbone.', '', 1)
            unwrapped_state_dict[new_key] = value
    
    print(f"   Original keys: {len(full_state_dict)}")
    print(f"   Unwrapped keys: {len(unwrapped_state_dict)}")
    print(f"   Network expects: {len(network_keys)}")
    
    # Try to load
    print(f"\n✨ Loading weights into network...")
    try:
        network.load_state_dict(unwrapped_state_dict)
        print("✅ SUCCESS! PPO checkpoint loaded correctly!")
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_ppo_checkpoint_loading()
    sys.exit(0 if success else 1)
