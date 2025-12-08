"""
Watch PPO Agent Play Generals.io with Visual Rendering

This script loads a trained PPO agent and lets you watch it play games
with visual rendering enabled.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from models.networks import DuelingDQN

# Generals environment imports
from generals import GymnasiumGenerals, GridFactory, Agent, Action
import gymnasium as gym


class PolicyNetwork(nn.Module):
    """PPO Policy Network."""
    
    def __init__(self, num_channels, num_actions, cnn_channels):
        super().__init__()
        self.num_actions = num_actions
        self.backbone = DuelingDQN(num_channels, num_actions, cnn_channels)
        
    def forward(self, state, mask=None):
        logits = self.backbone(state)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        action_probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        return action_probs, log_probs
    
    def sample_action(self, state, mask=None):
        action_probs, log_probs = self.forward(state, mask)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, log_probs.gather(1, action.unsqueeze(1)).squeeze(1)


class PPOAgent(Agent):
    """PPO agent for watching games."""
    
    def __init__(self, policy_net, device, deterministic=False, name="PPO_Agent"):
        super().__init__(id=name)
        self.policy = policy_net
        self.device = device
        self.deterministic = deterministic
        self.name = name
        self.policy.eval()
    
    def reset(self):
        pass
    
    def act(self, observation, action_mask):
        h, w = action_mask.shape[:2]
        mask_flat = action_mask.transpose(2, 0, 1).reshape(-1)
        
        with torch.no_grad():
            state = self.observation_to_state(observation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_tensor = torch.FloatTensor(mask_flat).unsqueeze(0).to(self.device)
            
            if self.deterministic:
                action_probs, _ = self.policy.forward(state_tensor, mask_tensor)
                action_idx = action_probs.argmax(dim=1).item()
            else:
                action_idx, _ = self.policy.sample_action(state_tensor, mask_tensor)
                action_idx = action_idx.item()
        
        return self.index_to_action(action_idx, observation.shape[1:])
    
    def observation_to_state(self, observation):
        c, h, w = observation.shape
        state = np.zeros((7, h, w), dtype=np.float32)
        
        terrain = np.zeros((h, w), dtype=np.float32)
        terrain += observation[3] * 1
        terrain += observation[2] * 2
        terrain += observation[1] * 3
        state[0] = np.clip(terrain, 0, 3)
        
        state[1] = observation[5]
        state[2] = observation[6]
        state[3] = observation[0] * observation[5]
        state[4] = observation[0] * observation[6]
        state[5] = observation[0] * observation[4]
        state[6] = observation[7]
        
        return state
    
    def index_to_action(self, action_idx, grid_shape):
        h, w = grid_shape
        num_cells = h * w
        
        direction_idx = action_idx // num_cells
        cell_idx = action_idx % num_cells
        
        source_row = cell_idx // w
        source_col = cell_idx % w
        
        direction_idx = int(np.clip(direction_idx, 0, 3))
        source_row = int(np.clip(source_row, 0, h-1))
        source_col = int(np.clip(source_col, 0, w-1))
        
        return Action(
            to_pass=False,
            row=source_row,
            col=source_col,
            direction=direction_idx,
            to_split=False
        )


def watch_game(checkpoint_path: str, num_games: int = 1, deterministic: bool = True):
    """Watch PPO agent play with visual rendering."""
    
    print("\n" + "="*70)
    print("🎮 WATCHING PPO AGENT PLAY GENERALS.IO 🎮")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Games to play: {num_games}")
    print(f"Mode: {'Deterministic (greedy)' if deterministic else 'Stochastic (sampling)'}")
    print("="*70 + "\n")
    
    # Load checkpoint
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = Config()
    
    # Initialize policy
    policy = PolicyNetwork(
        num_channels=config.NUM_CHANNELS,
        num_actions=config.NUM_ACTIONS,
        cnn_channels=config.CNN_CHANNELS
    ).to(device)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    print(f"✅ Loaded checkpoint from episode {checkpoint['episode']}")
    print(f"   Best win rate: {checkpoint['best_win_rate']:.1%}\n")
    
    # Create environment with rendering
    grid_factory = GridFactory(
        min_grid_dims=(15, 15),  # Smaller for easier viewing
        max_grid_dims=(20, 20),
    )
    
    agent_names = ["PPO_Agent", "Opponent"]
    
    env = GymnasiumGenerals(
        agents=agent_names,
        grid_factory=grid_factory,
        pad_observations_to=30,
        truncation=10000,
        render_mode="human"  # Enable visual rendering!
    )
    
    # Create agents
    ppo_agent = PPOAgent(policy, device, deterministic=deterministic, name="PPO_Agent")
    
    # Create opponent (BC baseline)
    opponent_policy = PolicyNetwork(
        num_channels=config.NUM_CHANNELS,
        num_actions=config.NUM_ACTIONS,
        cnn_channels=config.CNN_CHANNELS
    ).to(device)
    
    # Load BC checkpoint for opponent
    bc_checkpoint = torch.load("checkpoints/bc/best_model.pt", map_location=device)
    opponent_policy.backbone.load_state_dict(bc_checkpoint['model_state_dict'])
    opponent = PPOAgent(opponent_policy, device, deterministic=True, name="BC_Opponent")
    
    # Play games
    wins = 0
    losses = 0
    draws = 0
    
    for game_num in range(1, num_games + 1):
        print(f"\n{'='*70}")
        print(f"🎮 GAME {game_num}/{num_games}")
        print(f"{'='*70}")
        
        observations, infos = env.reset()
        
        done = False
        truncated = False
        step_count = 0
        episode_reward = 0
        
        while not (done or truncated):
            # Get actions
            agent_obs = observations[0]
            opponent_obs = observations[1]
            
            agent_mask = infos[agent_names[0]]["masks"]
            opponent_mask = infos[agent_names[1]]["masks"]
            
            agent_action = ppo_agent.act(agent_obs, agent_mask)
            opponent_action = opponent.act(opponent_obs, opponent_mask)
            
            # Step environment
            actions = [agent_action, opponent_action]
            observations, _, done, truncated, infos = env.step(actions)
            
            # Render (this will show the game visually)
            env.render()
            
            # Track reward
            episode_reward += infos[agent_names[0]]["reward"]
            step_count += 1
            
            if step_count > 10000:
                truncated = True
                break
        
        # Determine winner
        winner = infos[agent_names[0]].get("winner", None)
        
        if truncated and winner is None:
            result = "DRAW ⚖️"
            draws += 1
        elif winner == agent_names[0]:
            result = "WON ✓"
            wins += 1
        else:
            result = "LOST ✗"
            losses += 1
        
        print(f"\n🏁 Game {game_num} Result: {result}")
        print(f"   Steps: {step_count}")
        print(f"   Total Reward: {episode_reward:.1f}")
        
    # Summary
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Wins:   {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"Losses: {losses}/{num_games} ({losses/num_games*100:.1f}%)")
    print(f"Draws:  {draws}/{num_games} ({draws/num_games*100:.1f}%)")
    print(f"{'='*70}\n")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Watch PPO agent play Generals.io")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/ppo_fast/latest_model.pt",
                       help="Path to PPO checkpoint")
    parser.add_argument("--num_games", type=int, default=3,
                       help="Number of games to watch")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic policy (default: deterministic)")
    
    args = parser.parse_args()
    
    watch_game(
        checkpoint_path=args.checkpoint,
        num_games=args.num_games,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()
