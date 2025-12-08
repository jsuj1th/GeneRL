"""
Monitor PPO Training Progress - Watch Agent Exploration During Training

This script periodically loads the latest checkpoint during training and
plays games to visualize exploration behavior.

Usage:
    # Monitor BC warm start training
    python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_bc
    
    # Monitor from-scratch training  
    python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_scratch
"""

import os
import sys
import argparse
import time
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
    """PPO agent for monitoring."""
    
    def __init__(self, policy_net, device, deterministic=False, name="PPO_Agent"):
        super().__init__(id=name)
        self.policy = policy_net
        self.device = device
        self.deterministic = deterministic
        self.name = name
        self.policy.eval()
        self.last_action_idx = None  # Track action index for statistics
    
    def reset(self):
        self.last_action_idx = None
    
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
        
        self.last_action_idx = action_idx  # Store for tracking
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


def play_game_with_stats(agent, opponent, device, render=True):
    """Play one game and collect exploration statistics."""
    
    grid_factory = GridFactory(
        min_grid_dims=(15, 15),
        max_grid_dims=(20, 20),
    )
    
    agent_names = ["PPO_Agent", "Opponent"]
    
    render_mode = "human" if render else None
    
    env = GymnasiumGenerals(
        agents=agent_names,
        grid_factory=grid_factory,
        pad_observations_to=30,
        truncation=5000,
        render_mode=render_mode
    )
    
    observations, infos = env.reset()
    
    done = False
    truncated = False
    step_count = 0
    
    # Exploration stats
    max_tiles = 1
    max_army = 1
    cities_captured = 0
    unique_actions = set()
    tiles_over_time = []
    
    while not (done or truncated):
        agent_obs = observations[0]
        opponent_obs = observations[1]
        
        agent_mask = infos[agent_names[0]]["masks"]
        opponent_mask = infos[agent_names[1]]["masks"]
        
        agent_action = agent.act(agent_obs, agent_mask)
        opponent_action = opponent.act(opponent_obs, opponent_mask)
        
        # Track unique actions (use action index as string)
        if hasattr(agent, 'last_action_idx') and agent.last_action_idx is not None:
            unique_actions.add(str(agent.last_action_idx))
        
        actions = [agent_action, opponent_action]
        observations, _, done, truncated, infos = env.step(actions)
        
        # Render
        if render:
            env.render()
            time.sleep(0.05)  # Slow down for viewing
        
        # Track stats
        curr_land = infos[agent_names[0]].get('land', 0)
        curr_army = infos[agent_names[0]].get('army', 0)
        curr_cities = infos[agent_names[0]].get('cities', 0)
        
        if curr_land > max_tiles:
            max_tiles = curr_land
        if curr_army > max_army:
            max_army = curr_army
        if curr_cities > cities_captured:
            cities_captured = curr_cities
        
        tiles_over_time.append(curr_land)
        
        step_count += 1
        
        if step_count > 5000:
            truncated = True
            break
    
    # Determine winner
    winner = infos[agent_names[0]].get("winner", None)
    
    if truncated and winner is None:
        result = "DRAW"
    elif winner == agent_names[0]:
        result = "WIN"
    else:
        result = "LOSS"
    
    env.close()
    
    stats = {
        'result': result,
        'steps': step_count,
        'max_tiles': max_tiles,
        'max_army': max_army,
        'cities': cities_captured,
        'unique_actions': len(unique_actions),
        'action_diversity': len(unique_actions) / max(step_count, 1),
        'avg_tiles': np.mean(tiles_over_time),
        'tiles_growth': max_tiles - 1
    }
    
    return stats


def monitor_training(checkpoint_dir: str, check_interval: int = 300):
    """
    Monitor training by periodically loading checkpoints and playing games.
    
    Args:
        checkpoint_dir: Directory where checkpoints are saved
        check_interval: Seconds between checks (default: 5 minutes)
    """
    
    print("\n" + "="*70)
    print("👀 MONITORING PPO TRAINING PROGRESS")
    print("="*70)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Check interval: {check_interval} seconds ({check_interval/60:.1f} minutes)")
    print("="*70 + "\n")
    
    checkpoint_path = Path(checkpoint_dir) / "latest_model.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config = Config()
    
    # Load BC opponent
    bc_checkpoint = torch.load("checkpoints/bc/best_model.pt", map_location=device)
    opponent_policy = PolicyNetwork(
        num_channels=config.NUM_CHANNELS,
        num_actions=config.NUM_ACTIONS,
        cnn_channels=config.CNN_CHANNELS
    ).to(device)
    opponent_policy.backbone.load_state_dict(bc_checkpoint['model_state_dict'])
    opponent = PPOAgent(opponent_policy, device, deterministic=True, name="BC_Opponent")
    
    last_check_time = 0
    last_episode = -1
    
    print("⏳ Waiting for training to start...")
    
    while True:
        try:
            # Check if checkpoint exists
            if not checkpoint_path.exists():
                time.sleep(10)
                continue
            
            # Check if checkpoint has been updated
            current_check_time = checkpoint_path.stat().st_mtime
            
            if current_check_time > last_check_time:
                # New checkpoint available!
                last_check_time = current_check_time
                
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=device)
                episode = checkpoint['episode']
                
                # Skip if same episode
                if episode == last_episode:
                    time.sleep(10)
                    continue
                
                last_episode = episode
                total_steps = checkpoint['total_steps']
                best_win_rate = checkpoint.get('best_win_rate', 0.0)
                max_tiles_ever = checkpoint.get('max_tiles_ever', 1)
                
                print(f"\n{'='*70}")
                print(f"📊 CHECKPOINT UPDATE - Episode {episode}")
                print(f"{'='*70}")
                print(f"  Total steps: {total_steps:,}")
                print(f"  Best win rate: {best_win_rate:.1%}")
                print(f"  Max tiles ever: {max_tiles_ever}")
                print()
                
                # Load agent
                policy = PolicyNetwork(
                    num_channels=config.NUM_CHANNELS,
                    num_actions=config.NUM_ACTIONS,
                    cnn_channels=config.CNN_CHANNELS
                ).to(device)
                policy.load_state_dict(checkpoint['policy_state_dict'])
                agent = PPOAgent(policy, device, deterministic=False, name="PPO_Agent")
                
                # Play 3 games with visualization
                print("🎮 Playing 3 evaluation games with visualization...")
                print()
                
                all_stats = []
                
                for game_num in range(1, 4):
                    print(f"  Game {game_num}/3: ", end='', flush=True)
                    stats = play_game_with_stats(agent, opponent, device, render=True)
                    all_stats.append(stats)
                    
                    print(f"{stats['result']} | Steps: {stats['steps']} | "
                          f"Max tiles: {stats['max_tiles']} | Cities: {stats['cities']} | "
                          f"Action diversity: {stats['action_diversity']:.1%}")
                
                # Summary
                avg_tiles = np.mean([s['max_tiles'] for s in all_stats])
                avg_cities = np.mean([s['cities'] for s in all_stats])
                avg_diversity = np.mean([s['action_diversity'] for s in all_stats])
                wins = sum(1 for s in all_stats if s['result'] == 'WIN')
                
                print()
                print(f"  📈 Summary:")
                print(f"     Win rate: {wins}/3 ({wins/3*100:.0f}%)")
                print(f"     Avg max tiles: {avg_tiles:.1f}")
                print(f"     Avg cities: {avg_cities:.1f}")
                print(f"     Avg action diversity: {avg_diversity:.1%}")
                print()
                
                # Comparison with BC
                print(f"  🔍 Exploration Analysis:")
                if avg_tiles > 20:
                    print(f"     ✅ GOOD: Agent exploring {avg_tiles:.0f} tiles on average!")
                elif avg_tiles > 10:
                    print(f"     ⚠️  OK: Agent exploring {avg_tiles:.0f} tiles (room for improvement)")
                else:
                    print(f"     ❌ BAD: Agent only exploring {avg_tiles:.0f} tiles (similar to BC)")
                
                if avg_diversity > 0.3:
                    print(f"     ✅ GOOD: High action diversity ({avg_diversity:.1%})")
                elif avg_diversity > 0.2:
                    print(f"     ⚠️  OK: Moderate action diversity ({avg_diversity:.1%})")
                else:
                    print(f"     ❌ BAD: Low action diversity ({avg_diversity:.1%}) - like BC!")
                
                print()
                print(f"  ⏰ Next check in {check_interval/60:.0f} minutes...")
                print(f"{'='*70}\n")
                
                # Wait before next check
                time.sleep(check_interval)
            else:
                # No new checkpoint, wait a bit
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Monitor PPO training progress")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing training checkpoints")
    parser.add_argument("--interval", type=int, default=300,
                       help="Check interval in seconds (default: 300 = 5 minutes)")
    
    args = parser.parse_args()
    
    monitor_training(args.checkpoint_dir, args.interval)


if __name__ == "__main__":
    main()
