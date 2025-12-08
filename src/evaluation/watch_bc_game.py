"""
Watch BC Agent Play Generals.io with Visual Rendering

This script loads the trained BC agent and lets you watch it play games
with visual rendering enabled. You can watch BC vs BC, or BC vs Random.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from models.networks import GeneralsAgent

# Generals environment imports
from generals import GymnasiumGenerals, GridFactory, Agent, Action
import gymnasium as gym


class BCAgent(Agent):
    """BC agent for watching games."""
    
    def __init__(self, network, device, epsilon=0.0, name="BC_Agent"):
        super().__init__(id=name)
        self.network = network
        self.device = device
        self.epsilon = epsilon
        self.name = name
        self.network.eval()
    
    def reset(self):
        pass
    
    def act(self, observation, action_mask):
        """Select action using BC network."""
        h, w = action_mask.shape[:2]
        mask_flat = action_mask.transpose(2, 0, 1).reshape(-1)
        
        # Epsilon-greedy for some randomness
        if np.random.rand() < self.epsilon:
            # Random valid action
            valid_actions = np.where(mask_flat > 0)[0]
            if len(valid_actions) > 0:
                action_idx = np.random.choice(valid_actions)
            else:
                action_idx = 0
        else:
            # BC network prediction
            with torch.no_grad():
                state = self.observation_to_state(observation)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                mask_tensor = torch.FloatTensor(mask_flat).unsqueeze(0).to(self.device)
                
                q_values = self.network(state_tensor)
                
                # Mask invalid actions
                q_values = q_values.masked_fill(mask_tensor == 0, -1e9)
                
                # Greedy action
                action_idx = q_values.argmax(dim=1).item()
        
        return self.index_to_action(action_idx, observation.shape[1:])
    
    def observation_to_state(self, observation):
        """Convert observation to state format."""
        c, h, w = observation.shape
        state = np.zeros((7, h, w), dtype=np.float32)
        
        terrain = np.zeros((h, w), dtype=np.float32)
        terrain += observation[3] * 1  # Mountains
        terrain += observation[2] * 2  # Cities
        terrain += observation[1] * 3  # Generals
        state[0] = np.clip(terrain, 0, 3)
        
        state[1] = observation[5]  # my tiles
        state[2] = observation[6]  # enemy tiles
        state[3] = observation[0] * observation[5]  # my armies
        state[4] = observation[0] * observation[6]  # enemy armies
        state[5] = observation[0] * observation[4]  # neutral armies
        state[6] = observation[7]  # fog
        
        return state
    
    def index_to_action(self, action_idx, grid_shape):
        """Convert flat index to Action object."""
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


class RandomAgent(Agent):
    """Random agent for comparison."""
    
    def __init__(self, name="Random"):
        super().__init__(id=name)
        self.name = name
    
    def reset(self):
        pass
    
    def act(self, observation, action_mask):
        """Select random valid action."""
        h, w = action_mask.shape[:2]
        mask_flat = action_mask.transpose(2, 0, 1).reshape(-1)
        
        valid_actions = np.where(mask_flat > 0)[0]
        if len(valid_actions) > 0:
            action_idx = np.random.choice(valid_actions)
        else:
            action_idx = 0
        
        return self.index_to_action(action_idx, observation.shape[1:])
    
    def index_to_action(self, action_idx, grid_shape):
        """Convert flat index to Action object."""
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


def watch_game(checkpoint_path: str, num_games: int = 1, 
               opponent: str = "bc", epsilon: float = 0.0,
               render_delay: float = 0.1):
    """Watch agent play with visual rendering (supports both BC and PPO checkpoints)."""
    
    # Load checkpoint
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = Config()
    
    # Detect checkpoint type (BC or PPO)
    is_ppo = 'policy_state_dict' in checkpoint
    checkpoint_type = "PPO" if is_ppo else "BC"
    
    print("\n" + "="*70)
    print(f"🎮 WATCHING {checkpoint_type} AGENT PLAY GENERALS.IO 🎮")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Type: {checkpoint_type}")
    print(f"Games to play: {num_games}")
    print(f"Opponent: {opponent.upper()}")
    print(f"Epsilon (exploration): {epsilon}")
    print(f"Render delay: {render_delay}s per step")
    print("="*70 + "\n")
    
    # Initialize network based on checkpoint type
    agent_network = GeneralsAgent(
        num_channels=config.NUM_CHANNELS,
        num_actions=config.NUM_ACTIONS,
        cnn_channels=config.CNN_CHANNELS
    ).to(device)
    
    if is_ppo:
        # PPO checkpoint - load from policy_state_dict
        # PPO wraps GeneralsAgent in a PolicyNetwork with .backbone attribute
        # So we need to strip the "backbone." prefix from all keys
        full_state_dict = checkpoint['policy_state_dict']
        unwrapped_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '', 1)  # Remove first occurrence only
                unwrapped_state_dict[new_key] = value
        
        agent_network.load_state_dict(unwrapped_state_dict)
        
        print(f"✅ Loaded PPO checkpoint")
        if 'episode' in checkpoint:
            print(f"   Episode: {checkpoint['episode']}")
        if 'total_steps' in checkpoint:
            print(f"   Total steps: {checkpoint['total_steps']:,}")
        if 'best_win_rate' in checkpoint:
            print(f"   Best win rate: {checkpoint['best_win_rate']:.1%}")
        print()
    else:
        # BC checkpoint - load from model_state_dict
        agent_network.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Loaded BC checkpoint")
        if 'test_accuracy' in checkpoint:
            print(f"   Test accuracy: {checkpoint['test_accuracy']:.2%}")
        print()
    
    agent_network.eval()
    
    # Create environment with rendering
    grid_factory = GridFactory(
        min_grid_dims=(8, 8),  # Smaller for easier viewing
        max_grid_dims=(9, 9),
    )
    
    agent_names = ["BC_Agent", "Opponent"]
    
    env = GymnasiumGenerals(
        agents=agent_names,
        grid_factory=grid_factory,
        pad_observations_to=30,
        truncation=5000,  # Shorter games for faster viewing
        render_mode="human"  # Enable visual rendering!
    )
    
    # Create agents
    agent_name = f"{checkpoint_type}_Agent"
    bc_agent = BCAgent(agent_network, device, epsilon=epsilon, name=agent_name)
    
    if opponent == "bc":
        # Agent vs same model
        opponent_network = GeneralsAgent(
            num_channels=config.NUM_CHANNELS,
            num_actions=config.NUM_ACTIONS,
            cnn_channels=config.CNN_CHANNELS
        ).to(device)
        
        # Load opponent from same checkpoint
        if is_ppo:
            full_state_dict = checkpoint['policy_state_dict']
            unwrapped_state_dict = {}
            for key, value in full_state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', '', 1)
                    unwrapped_state_dict[new_key] = value
            opponent_network.load_state_dict(unwrapped_state_dict)
        else:
            opponent_network.load_state_dict(checkpoint['model_state_dict'])
        
        opponent_agent = BCAgent(opponent_network, device, epsilon=epsilon, name=f"{checkpoint_type}_Opponent")
        print(f"🤖 Opponent: {checkpoint_type} Agent (same model)")
    else:
        # BC vs Random
        opponent_agent = RandomAgent(name="Random_Opponent")
        print("🎲 Opponent: Random Agent")
    
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
        
        print("\n🎬 Starting game... (Close window or Ctrl+C to stop)")
        print("   Watching at reduced speed for visibility...\n")
        
        try:
            while not (done or truncated):
                # Get actions
                agent_obs = observations[0]
                opponent_obs = observations[1]
                
                agent_mask = infos[agent_names[0]]["masks"]
                opponent_mask = infos[agent_names[1]]["masks"]
                
                agent_action = bc_agent.act(agent_obs, agent_mask)
                opponent_action = opponent_agent.act(opponent_obs, opponent_mask)
                
                # Step environment
                actions = [agent_action, opponent_action]
                observations, _, done, truncated, infos = env.step(actions)
                
                # Render (this will show the game visually)
                env.render()
                
                # Add delay so you can see what's happening
                time.sleep(render_delay)
                
                # Track reward
                episode_reward += infos[agent_names[0]]["reward"]
                step_count += 1
                
                # Print progress every 100 steps
                if step_count % 100 == 0:
                    print(f"   Step {step_count}: Reward = {episode_reward:.1f}")
                
                if step_count > 5000:
                    truncated = True
                    break
                    
        except KeyboardInterrupt:
            print("\n\n⚠️  Game interrupted by user")
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
        
        # Get final stats
        agent_info = infos[agent_names[0]]
        print(f"   Final Territory: {agent_info.get('land', 0)} tiles")
        print(f"   Final Army: {agent_info.get('army', 0)} units")
    
    # Summary
    if num_games > 0:
        print(f"\n{'='*70}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"Wins:   {wins}/{num_games} ({wins/num_games*100:.1f}%)")
        print(f"Losses: {losses}/{num_games} ({losses/num_games*100:.1f}%)")
        print(f"Draws:  {draws}/{num_games} ({draws/num_games*100:.1f}%)")
        print(f"{'='*70}\n")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Watch agent play Generals.io (BC or PPO)")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/ppo_from_bc/latest_model.pt",
                       help="Path to checkpoint (BC or PPO)")
    parser.add_argument("--num_games", type=int, default=3,
                       help="Number of games to watch")
    parser.add_argument("--opponent", type=str, choices=["bc", "random"], default="bc",
                       help="Opponent type: 'bc' (BC vs BC) or 'random' (BC vs Random)")
    parser.add_argument("--epsilon", type=float, default=0.1,
                       help="Epsilon for exploration (0.0 = greedy, 0.1 = 10%% random, 0.2 = 20%% random)")
    parser.add_argument("--delay", type=float, default=0.1,
                       help="Delay between steps in seconds (for visualization)")
    
    args = parser.parse_args()
    
    watch_game(
        checkpoint_path=args.checkpoint,
        num_games=args.num_games,
        opponent=args.opponent,
        epsilon=args.epsilon,
        render_delay=args.delay
    )


if __name__ == "__main__":
    main()
