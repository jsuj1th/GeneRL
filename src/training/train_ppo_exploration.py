"""
PPO Training with AGGRESSIVE EXPLORATION REWARDS

This version adds dense rewards specifically for:
1. Territory expansion (exploring new tiles)
2. Army growth (consolidating power)
3. City capture (economic advantage)
4. Fog reduction (scouting/searching)
5. Distance to opponent (aggressive play)

These rewards encourage the agent to explore the map and find the opponent.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import copy
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

# Suppress NumPy overflow warnings from generals-bots library
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow encountered.*')

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
    
    def evaluate_actions(self, state, action, mask=None):
        action_probs, log_probs = self.forward(state, mask)
        dist = Categorical(action_probs)
        action_log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        entropy = dist.entropy()
        return action_log_probs, entropy


class ValueNetwork(nn.Module):
    """Value network for PPO."""
    
    def __init__(self, num_channels, cnn_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cnn_channels[2], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        features = self.conv(state)
        value = self.value_head(features)
        return value.squeeze(-1)


class PPOAgent(Agent):
    """PPO agent."""
    
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


class RolloutBuffer:
    """Buffer for storing rollout data for PPO."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.masks = []
    
    def push(self, state, action, reward, value, log_prob, done, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.masks.append(mask)
    
    def get(self):
        if len(self.states) == 0:
            raise ValueError("RolloutBuffer is empty!")
        
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(np.array(self.rewards)),
            torch.FloatTensor(np.array(self.values)),
            torch.FloatTensor(np.array(self.log_probs)),
            torch.FloatTensor(np.array(self.dones)),
            torch.FloatTensor(np.array(self.masks))
        )
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.masks.clear()
    
    def __len__(self):
        return len(self.states)


class PPOTrainer:
    """PPO trainer with exploration rewards."""
    
    def __init__(self, config: Config, bc_checkpoint: str, output_dir: str, from_scratch: bool = False):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.from_scratch = from_scratch
        
        # Device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize policy network
        self.policy = PolicyNetwork(
            num_channels=config.NUM_CHANNELS,
            num_actions=config.NUM_ACTIONS,
            cnn_channels=config.CNN_CHANNELS
        ).to(self.device)
        
        # Initialize value network
        self.value_net = ValueNetwork(
            num_channels=config.NUM_CHANNELS,
            cnn_channels=config.CNN_CHANNELS
        ).to(self.device)
        
        if not from_scratch:
            # Load BC checkpoint for warm start
            print(f"\nLoading BC checkpoint: {bc_checkpoint}")
            checkpoint = torch.load(bc_checkpoint, map_location=self.device)
            self.policy.backbone.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded BC weights for warm start")
        else:
            print("\n🔥 Training from SCRATCH (no BC initialization)")
        
        print(f"Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"Value parameters: {sum(p.numel() for p in self.value_net.parameters()):,}")
        
        # Optimizers (higher learning rate for from-scratch)
        lr = 1e-3 if from_scratch else 3e-4
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr * 3)
        
        # PPO hyperparameters (AGGRESSIVE EXPLORATION)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.ppo_epochs = 2
        self.batch_size = 32
        self.value_coef = 0.5
        self.entropy_coef = 0.1  # 10x higher than before for MORE exploration!
        
        # Exploration reward weights (VERY AGGRESSIVE)
        self.reward_weights = {
            'win': 100.0,
            'loss': -100.0,
            'territory': 1.0,      # +1 per tile gained (was 0.1)
            'army': 0.05,          # +0.05 per army unit (was 0.01)
            'city': 10.0,          # +10 per city (was 5.0)
            'fog_reduction': 0.5,  # +0.5 per fog tile revealed (NEW!)
            'max_tiles_bonus': 5.0 # +5 when reaching new max tiles (NEW!)
        }
        
        print(f"\n🎯 EXPLORATION REWARD WEIGHTS:")
        for key, val in self.reward_weights.items():
            print(f"   {key:20s}: {val:6.2f}")
        
        # Tracking
        self.episode = 0
        self.total_steps = 0
        self.max_tiles_ever = 1  # Track maximum tiles controlled
        
        # Opponent pool
        if not from_scratch:
            bc_checkpoint_data = torch.load(bc_checkpoint, map_location=self.device)
            self.opponent_pool = [
                copy.deepcopy(bc_checkpoint_data['model_state_dict']),
                "RANDOM"
            ]
        else:
            self.opponent_pool = ["RANDOM"]  # Only random opponents when from scratch
        
        print(f"Opponent pool size: {len(self.opponent_pool)}")
        
        # TensorBoard
        log_dir = Path("logs/ppo_exploration") / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}\n")
        
        # Metrics
        self.best_win_rate = 0.0
        self.training_start_time = None
    
    def create_environment(self):
        """Create environment with smaller grids for faster episodes."""
        grid_factory = GridFactory(
            min_grid_dims=(12, 12),  # Even smaller for faster exploration
            max_grid_dims=(18, 18),
        )
        
        agent_names = ["PPO_Agent", "Opponent"]
        
        env = GymnasiumGenerals(
            agents=agent_names,
            grid_factory=grid_factory,
            pad_observations_to=30,
            truncation=5000  # Shorter games to encourage faster exploration
        )
        
        return env, agent_names
    
    def compute_exploration_reward(self, info, prev_info):
        """Compute dense exploration rewards."""
        reward = 0.0
        
        # Territory expansion (MOST IMPORTANT!)
        curr_land = info.get('land', 0)
        prev_land = prev_info.get('land', 1)
        tiles_gained = curr_land - prev_land
        if tiles_gained > 0:
            reward += tiles_gained * self.reward_weights['territory']
            # Bonus for reaching new maximum!
            if curr_land > self.max_tiles_ever:
                reward += self.reward_weights['max_tiles_bonus']
                self.max_tiles_ever = curr_land
        
        # Army growth
        curr_army = info.get('army', 0)
        prev_army = prev_info.get('army', 1)
        army_gained = curr_army - prev_army
        if army_gained > 0:
            reward += army_gained * self.reward_weights['army']
        
        # City capture
        curr_cities = info.get('cities', 0)
        prev_cities = prev_info.get('cities', 0)
        if curr_cities > prev_cities:
            reward += (curr_cities - prev_cities) * self.reward_weights['city']
        
        # Fog reduction (reveals map - encourages scouting!)
        # Count visible tiles (non-fog)
        # This is approximate - we'd need actual fog data from observation
        # For now, correlate with territory expansion
        if tiles_gained > 0:
            reward += tiles_gained * self.reward_weights['fog_reduction']
        
        return reward
    
    def collect_rollout(self, verbose=False):
        """Collect one episode with exploration rewards."""
        if verbose:
            print(f"\n🎮 Starting episode {self.episode + 1}...")
        
        env, agent_names = self.create_environment()
        
        # Create agents
        agent = PPOAgent(self.policy, self.device, deterministic=False, name="PPO_Agent")
        
        # Simple curriculum: 100% random early, then mix
        if self.from_scratch or self.episode < 100:
            opponent_type = "RANDOM"
        elif self.episode < 500:
            opponent_type = "RANDOM" if np.random.rand() < 0.5 else self.opponent_pool[0]
        else:
            opponent_type = "RANDOM" if np.random.rand() < 0.3 else self.opponent_pool[0]
        
        if opponent_type == "RANDOM" or self.from_scratch:
            # Random opponent
            if self.from_scratch:
                opponent_policy = PolicyNetwork(
                    num_channels=self.config.NUM_CHANNELS,
                    num_actions=self.config.NUM_ACTIONS,
                    cnn_channels=self.config.CNN_CHANNELS
                ).to(self.device)
            else:
                opponent_policy = PolicyNetwork(
                    num_channels=self.config.NUM_CHANNELS,
                    num_actions=self.config.NUM_ACTIONS,
                    cnn_channels=self.config.CNN_CHANNELS
                ).to(self.device)
                opponent_policy.backbone.load_state_dict(self.opponent_pool[0])
            opponent = PPOAgent(opponent_policy, self.device, deterministic=False, name="Random_Opponent")
            if verbose:
                print(f"  🎲 Opponent: Random")
        else:
            # BC opponent
            opponent_policy = PolicyNetwork(
                num_channels=self.config.NUM_CHANNELS,
                num_actions=self.config.NUM_ACTIONS,
                cnn_channels=self.config.CNN_CHANNELS
            ).to(self.device)
            opponent_policy.backbone.load_state_dict(opponent_type)
            opponent = PPOAgent(opponent_policy, self.device, deterministic=False, name="BC_Opponent")
            if verbose:
                print(f"  🤖 Opponent: BC Baseline")
        
        # Reset environment
        observations, infos = env.reset()
        
        rollout = RolloutBuffer()
        episode_reward = 0.0
        exploration_reward_total = 0.0
        done = False
        truncated = False
        step_count = 0
        
        # Track stats for verbose output
        max_tiles = 1
        cities_captured = 0
        
        prev_info = infos[agent_names[0]]
        
        while not (done or truncated):
            agent_obs = observations[0]
            opponent_obs = observations[1]
            
            agent_mask = infos[agent_names[0]]["masks"]
            opponent_mask = infos[agent_names[1]]["masks"]
            
            # Agent action
            with torch.no_grad():
                agent_state = agent.observation_to_state(agent_obs)
                state_tensor = torch.FloatTensor(agent_state).unsqueeze(0).to(self.device)
                mask_flat = agent_mask.transpose(2, 0, 1).reshape(-1)
                mask_tensor = torch.FloatTensor(mask_flat).unsqueeze(0).to(self.device)
                
                action_idx, log_prob = self.policy.sample_action(state_tensor, mask_tensor)
                value = self.value_net(state_tensor)
                
                action_idx = action_idx.item()
                log_prob = log_prob.item()
                value = value.item()
            
            agent_action = agent.index_to_action(action_idx, agent_obs.shape[1:])
            opponent_action = opponent.act(opponent_obs, opponent_mask)
            
            # Step environment
            actions = [agent_action, opponent_action]
            next_observations, _, terminated, truncated, next_infos = env.step(actions)
            
            # Base reward from environment
            base_reward = next_infos[agent_names[0]]["reward"]
            
            # Exploration reward
            exploration_reward = self.compute_exploration_reward(
                next_infos[agent_names[0]], 
                prev_info
            )
            
            # Total reward
            total_reward = base_reward + exploration_reward
            
            # Store transition
            rollout.push(
                agent_state,
                action_idx,
                total_reward,
                value,
                log_prob,
                float(terminated or truncated),
                mask_flat
            )
            
            episode_reward += total_reward
            exploration_reward_total += exploration_reward
            
            # Track stats
            curr_land = next_infos[agent_names[0]].get('land', 0)
            if curr_land > max_tiles:
                max_tiles = curr_land
            curr_cities = next_infos[agent_names[0]].get('cities', 0)
            if curr_cities > cities_captured:
                cities_captured = curr_cities
            
            observations = next_observations
            infos = next_infos
            prev_info = next_infos[agent_names[0]]
            done = terminated
            step_count += 1
            
            if step_count > 5000:
                truncated = True
                break
        
        # Determine winner
        winner = next_infos[agent_names[0]].get("winner", None)
        
        # Final reward
        if winner == agent_names[0]:
            agent_won = True
            final_reward = self.reward_weights['win']
        elif winner is None:
            agent_won = None
            final_reward = 0.0
        else:
            agent_won = False
            final_reward = self.reward_weights['loss']
        
        # Add final reward to last transition
        if len(rollout) > 0:
            rollout.rewards[-1] += final_reward
            episode_reward += final_reward
        
        if verbose:
            result = "WON ✓" if agent_won == True else "LOST ✗" if agent_won == False else "DRAW ⚖️"
            print(f"  🏁 Result: {result} | Steps: {step_count}")
            print(f"  📊 Max Tiles: {max_tiles} | Cities: {cities_captured}")
            print(f"  💰 Total Reward: {episode_reward:.1f} (Exploration: {exploration_reward_total:.1f})")
        
        if len(rollout) == 0:
            print(f"⚠️  WARNING: Episode ended with 0 steps!")
        
        return rollout, agent_won, episode_reward, step_count, max_tiles, cities_captured
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)
    
    def ppo_update(self, rollout):
        if len(rollout) == 0:
            print("⚠️  WARNING: Cannot update PPO with empty rollout!")
            return 0.0, 0.0
        
        states, actions, rewards, values, old_log_probs, dones, masks = rollout.get()
        
        with torch.no_grad():
            next_value = 0
        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = advantages + values
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        masks = masks.to(self.device)
        
        policy_losses = []
        value_losses = []
        
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_masks = masks[batch_indices]
                
                new_log_probs, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions, batch_masks
                )
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                values_pred = self.value_net(batch_states)
                value_loss = nn.MSELoss()(values_pred, batch_returns)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
        
        return np.mean(policy_losses), np.mean(value_losses)
    
    def evaluate(self, num_games=20):
        wins = 0
        draws = 0
        
        print(f"  Playing {num_games} evaluation games...", end='', flush=True)
        
        for i in range(num_games):
            if i > 0 and i % 5 == 0:
                print(f" {i}/{num_games}", end='', flush=True)
            
            rollout, won, _, _, _, _ = self.collect_rollout(verbose=False)
            
            if won == True:
                wins += 1
            elif won is None:
                draws += 1
        
        print(f" {num_games}/{num_games} ✓")
        
        return (wins + 0.5 * draws) / num_games
    
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'best_win_rate': self.best_win_rate,
            'max_tiles_ever': self.max_tiles_ever,
            'config': {
                'num_channels': self.config.NUM_CHANNELS,
                'num_actions': self.config.NUM_ACTIONS,
                'cnn_channels': self.config.CNN_CHANNELS
            }
        }
        
        latest_path = self.output_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (win rate: {self.best_win_rate:.1%})")
    
    def train(self, training_hours: float):
        print("\n" + "🎮"*40)
        print("🚀 PPO WITH EXPLORATION REWARDS 🚀")
        print("🎮"*40)
        print(f"\n⏱️  Training Duration: {training_hours} hours")
        print(f"🖥️  Device: {self.device}")
        print(f"🔥 Training Mode: {'FROM SCRATCH' if self.from_scratch else 'Warm Start (BC)'}")
        print(f"📦 Batch Size: {self.batch_size}")
        print(f"🎯 Entropy Coef: {self.entropy_coef} (HIGH exploration!)")
        print("="*80 + "\n")
        
        self.training_start_time = time.time()
        end_time = self.training_start_time + (training_hours * 3600)
        
        episode_rewards = []
        episode_lengths = []
        max_tiles_list = []
        cities_list = []
        wins = []
        
        try:
            while time.time() < end_time:
                self.episode += 1
                verbose = (self.episode == 1 or self.episode % 10 == 0)
                
                rollout, won, reward, length, max_tiles, cities = self.collect_rollout(verbose=verbose)
                
                policy_loss, value_loss = self.ppo_update(rollout)
                
                episode_rewards.append(reward)
                episode_lengths.append(length)
                max_tiles_list.append(max_tiles)
                cities_list.append(cities)
                wins.append(1 if won == True else 0)
                self.total_steps += length
                
                if verbose:
                    print(f"  📊 Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}")
                
                if self.episode % 10 == 0:
                    elapsed = time.time() - self.training_start_time
                    
                    avg_reward = np.mean(episode_rewards[-10:])
                    win_rate = np.mean(wins[-10:])
                    avg_length = np.mean(episode_lengths[-10:])
                    avg_tiles = np.mean(max_tiles_list[-10:])
                    avg_cities = np.mean(cities_list[-10:])
                    
                    print(f"\n{'='*80}")
                    print(f"📊 Episode {self.episode:4d} Summary")
                    print(f"{'='*80}")
                    print(f"  Reward:      {avg_reward:+.1f}")
                    print(f"  Win Rate:    {win_rate:.0%} (last 10)")
                    print(f"  Avg Tiles:   {avg_tiles:.1f} (max ever: {self.max_tiles_ever})")
                    print(f"  Avg Cities:  {avg_cities:.1f}")
                    print(f"  Avg Length:  {avg_length:.0f} steps")
                    print(f"  Total Steps: {self.total_steps:,}")
                    print(f"  Time:        {elapsed/3600:.2f}h / {training_hours:.2f}h")
                    print(f"{'='*80}\n")
                    
                    self.writer.add_scalar('Train/Reward', avg_reward, self.episode)
                    self.writer.add_scalar('Train/WinRate', win_rate, self.episode)
                    self.writer.add_scalar('Train/AvgTiles', avg_tiles, self.episode)
                    self.writer.add_scalar('Train/AvgCities', avg_cities, self.episode)
                    self.writer.add_scalar('Train/EpisodeLength', avg_length, self.episode)
                
                if self.episode % 50 == 0:
                    print("\n" + "🔍" + "="*78 + "🔍")
                    print("📊 EVALUATION...")
                    print("="*80)
                    win_rate = self.evaluate(num_games=20)
                    print(f"\n✅ Evaluation: Win rate = {win_rate:.1%}")
                    
                    self.writer.add_scalar('Eval/WinRate', win_rate, self.episode)
                    
                    is_best = win_rate > self.best_win_rate
                    if is_best:
                        self.best_win_rate = win_rate
                        print(f"  ⭐ New best win rate!")
                    
                    self.save_checkpoint(is_best=is_best)
                    print()
                
                if self.episode % 100 == 0:
                    self.save_checkpoint(is_best=False)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted")
        
        print("\n" + "="*70)
        print("✅ Training Complete!")
        print("="*70)
        
        final_win_rate = self.evaluate(num_games=50)
        print(f"Final win rate: {final_win_rate:.1%}")
        print(f"Best win rate: {self.best_win_rate:.1%}")
        print(f"Max tiles ever: {self.max_tiles_ever}")
        print(f"Total episodes: {self.episode}")
        print(f"Total steps: {self.total_steps:,}")
        
        self.save_checkpoint(is_best=False)
        
        summary = {
            'training_hours': training_hours,
            'total_episodes': self.episode,
            'total_steps': self.total_steps,
            'best_win_rate': self.best_win_rate,
            'final_win_rate': final_win_rate,
            'max_tiles_ever': self.max_tiles_ever,
            'from_scratch': self.from_scratch,
            'device': str(self.device)
        }
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Saved to {self.output_dir}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="PPO with exploration rewards")
    parser.add_argument("--bc_checkpoint", type=str, default="checkpoints/bc/best_model.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/ppo_explore")
    parser.add_argument("--training_hours", type=float, default=1.0)
    parser.add_argument("--from_scratch", action="store_true",
                       help="Train from scratch (no BC initialization)")
    
    args = parser.parse_args()
    
    config = Config()
    
    trainer = PPOTrainer(
        config=config,
        bc_checkpoint=args.bc_checkpoint,
        output_dir=args.output_dir,
        from_scratch=args.from_scratch
    )
    
    trainer.train(training_hours=args.training_hours)


if __name__ == "__main__":
    main()
