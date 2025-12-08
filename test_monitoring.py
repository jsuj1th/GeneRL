#!/usr/bin/env python3
"""
Quick test of the monitoring script to verify the Action attribute bug is fixed.
This will play just 1 game to test if the script runs without errors.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import and run monitoring
from evaluation.monitor_exploration import monitor_training

if __name__ == "__main__":
    print("🧪 Testing monitoring script with 1 game...")
    print("This will verify the Action attribute bug is fixed.\n")
    
    # Test with existing checkpoint
    checkpoint_dir = "checkpoints/ppo_from_bc"
    
    # We'll modify monitor_training to play just 1 game for testing
    import os
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    from src.config import Config
    from src.models.networks import DuelingDQN
    from generals import GymnasiumGenerals, GridFactory, Agent, Action
    from evaluation.monitor_exploration import PolicyNetwork, PPOAgent, play_game_with_stats
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    config = Config()
    checkpoint_path = Path(checkpoint_dir) / "latest_model.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ No checkpoint found at {checkpoint_path}")
        sys.exit(1)
    
    # Load agent
    print("Loading PPO agent...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent_policy = PolicyNetwork(
        num_channels=config.NUM_CHANNELS,
        num_actions=config.NUM_ACTIONS,
        cnn_channels=config.CNN_CHANNELS
    ).to(device)
    # PPO checkpoints have nested backbone, so load the full policy
    agent_policy.load_state_dict(checkpoint['policy_state_dict'])
    agent = PPOAgent(agent_policy, device, deterministic=False, name="PPO_Agent")
    
    # Load BC opponent
    print("Loading BC opponent...")
    bc_checkpoint = torch.load("checkpoints/bc/best_model.pt", map_location=device)
    opponent_policy = PolicyNetwork(
        num_channels=config.NUM_CHANNELS,
        num_actions=config.NUM_ACTIONS,
        cnn_channels=config.CNN_CHANNELS
    ).to(device)
    opponent_policy.backbone.load_state_dict(bc_checkpoint['model_state_dict'])
    opponent = PPOAgent(opponent_policy, device, deterministic=True, name="BC_Opponent")
    
    # Play 1 test game
    print("\n" + "="*70)
    print("🎮 PLAYING TEST GAME (NO VISUAL)")
    print("="*70 + "\n")
    
    try:
        stats = play_game_with_stats(agent, opponent, device, render=False)
        
        print("\n✅ SUCCESS! No Action attribute errors!")
        print("\n" + "="*70)
        print("📊 GAME STATISTICS")
        print("="*70)
        print(f"Result: {stats['result']}")
        print(f"Steps: {stats['steps']}")
        print(f"Max tiles explored: {stats['max_tiles']}")
        print(f"Max army: {stats['max_army']}")
        print(f"Cities captured: {stats['cities']}")
        print(f"Unique actions: {stats['unique_actions']}")
        print(f"Action diversity: {stats['action_diversity']*100:.1f}%")
        print(f"Avg tiles: {stats['avg_tiles']:.1f}")
        print(f"Tiles growth: {stats['tiles_growth']}")
        print("="*70 + "\n")
        
        print("✅ Monitoring script is working correctly!")
        print("You can now run the full monitoring with:")
        print(f"  python src/evaluation/monitor_exploration.py --checkpoint_dir {checkpoint_dir}")
        
    except AttributeError as e:
        if "Action" in str(e):
            print(f"\n❌ FAILED: Action attribute error still exists!")
            print(f"Error: {e}")
            sys.exit(1)
        else:
            raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
