"""
Real Environment PPO Training Script for Generals.io

PPO (Proximal Policy Optimization) alternative to DQN.
Advantages:
- Policy-based (better for complex action spaces)
- More stable training
- No replay buffer needed
- Simpler implementation

Based on: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
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
    """
    PPO Policy Network - outputs action probabilities.
    Reuses same architecture as DQN but outputs probabilities instead of Q-values.
    """
    
    def __init__(self, num_channels, num_actions, cnn_channels):
        super().__init__()
        self.num_actions = num_actions
        
        # Reuse DQN architecture for feature extraction
        self.backbone = DuelingDQN(num_channels, num_actions, cnn_channels)
        
    def forward(self, state, mask=None):
        """
        Forward pass.
        
        Args:
            state: (batch, channels, h, w)
            mask: (batch, actions) - valid action mask
            
        Returns:
            action_probs: (batch, actions) - probability distribution
            log_probs: (batch, actions) - log probabilities
        """
        logits = self.backbone(state)  # (batch, actions)
        
        if mask is not None:
            # Mask invalid actions
            logits = logits.masked_fill(mask == 0, -1e9)
        
        # Softmax to get probabilities
        action_probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        return action_probs, log_probs
    
    def sample_action(self, state, mask=None):
        """Sample action from policy."""
        action_probs, log_probs = self.forward(state, mask)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
    
    def evaluate_actions(self, state, action, mask=None):
        """Evaluate log probability and entropy of given actions."""
        action_probs, log_probs = self.forward(state, mask)
        dist = Categorical(action_probs)
        
        action_log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        entropy = dist.entropy()
        
        return action_log_probs, entropy


class ValueNetwork(nn.Module):
    """Value network for PPO - estimates state value V(s)."""
    
    def __init__(self, num_channels, cnn_channels):
        super().__init__()
        
        # CNN feature extractor (same as policy)
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cnn_channels[2], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        """Predict state value."""
        features = self.conv(state)
        value = self.value_head(features)
        return value.squeeze(-1)


class PPOAgent(Agent):
    """PPO agent for Generals.io environment."""
    
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
        """Select action from policy."""
        h, w = action_mask.shape[:2]
        mask_flat = action_mask.transpose(2, 0, 1).reshape(-1)
        
        with torch.no_grad():
            state = self.observation_to_state(observation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_tensor = torch.FloatTensor(mask_flat).unsqueeze(0).to(self.device)
            
            if self.deterministic:
                # Greedy action
                action_probs, _ = self.policy.forward(state_tensor, mask_tensor)
                action_idx = action_probs.argmax(dim=1).item()
            else:
                # Sample from policy
                action_idx, _ = self.policy.sample_action(state_tensor, mask_tensor)
                action_idx = action_idx.item()
        
        return self.index_to_action(action_idx, observation.shape[1:])
    
    def observation_to_state(self, observation):
        """Convert observation to state format (same as DQN)."""
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
        """Get all data as tensors."""
        if len(self.states) == 0:
            raise ValueError("RolloutBuffer is empty! No transitions were collected.")
        
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
    """PPO trainer for Generals.io."""
    
    def __init__(self, config: Config, bc_checkpoint: str, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load BC checkpoint
        print(f"\nLoading BC checkpoint: {bc_checkpoint}")
        checkpoint = torch.load(bc_checkpoint, map_location=self.device)
        
        # Initialize policy network from BC
        self.policy = PolicyNetwork(
            num_channels=config.NUM_CHANNELS,
            num_actions=config.NUM_ACTIONS,
            cnn_channels=config.CNN_CHANNELS
        ).to(self.device)
        
        # Load BC weights into policy backbone
        self.policy.backbone.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize value network
        self.value_net = ValueNetwork(
            num_channels=config.NUM_CHANNELS,
            cnn_channels=config.CNN_CHANNELS
        ).to(self.device)
        
        print(f"Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"Value parameters: {sum(p.numel() for p in self.value_net.parameters()):,}")
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)
        
        # PPO hyperparameters (optimized for speed AND exploration)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.ppo_epochs = 2  # Reduced from 4 for faster updates
        self.batch_size = 32  # Reduced from 64 for faster processing
        self.value_coef = 0.5
        self.entropy_coef = 0.1  # INCREASED from 0.01 to encourage exploration!
        
        # Tracking
        self.episode = 0
        self.total_steps = 0
        
        # Opponent pool
        self.opponent_pool = [
            copy.deepcopy(self.policy.backbone.state_dict()),  # BC baseline
            "RANDOM"
        ]
        print(f"Opponent pool size: {len(self.opponent_pool)} (BC + Random)")
        
        # TensorBoard
        log_dir = Path("logs/ppo_real_env") / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")
        
        # Metrics
        self.best_win_rate = 0.0
        self.training_start_time = None
    
    def create_environment(self):
        """Create Generals.io environment."""
        grid_factory = GridFactory(
            min_grid_dims=(15, 15),  # Smaller grids = faster games
            max_grid_dims=(20, 20),  # Reduced from 28x28 for speed
        )
        
        agent_names = ["PPO_Agent", "Opponent"]
        
        env = GymnasiumGenerals(
            agents=agent_names,
            grid_factory=grid_factory,
            pad_observations_to=30,
            truncation=5000  # Reduced from 100k for faster episodes (still allows natural endings)
        )
        
        return env, agent_names
    
    def collect_rollout(self, verbose=False):
        """Collect one episode of experience."""
        if verbose:
            print(f"\n🎮 Starting episode {self.episode + 1}...")
        
        env, agent_names = self.create_environment()
        
        # Create agents
        agent = PPOAgent(self.policy, self.device, deterministic=False, name="PPO_Agent")
        
        # Curriculum learning
        if self.episode < 500:
            opponent_type = "RANDOM" if np.random.rand() < 0.8 else self.opponent_pool[0]
        elif self.episode < 1500:
            opponent_type = np.random.choice(self.opponent_pool)
        else:
            opponent_type = "RANDOM" if np.random.rand() < 0.2 else self.opponent_pool[0]
        
        if opponent_type == "RANDOM":
            opponent_policy = PolicyNetwork(
                num_channels=self.config.NUM_CHANNELS,
                num_actions=self.config.NUM_ACTIONS,
                cnn_channels=self.config.CNN_CHANNELS
            ).to(self.device)
            opponent_policy.backbone.load_state_dict(self.opponent_pool[0])
            opponent = PPOAgent(opponent_policy, self.device, deterministic=False, name="Random_Opponent")
            if verbose:
                print(f"  🎲 Opponent: Random (stochastic policy)")
        else:
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
        episode_reward = 0
        done = False
        truncated = False
        step_count = 0
        
        while not (done or truncated):
            agent_obs = observations[0]
            opponent_obs = observations[1]
            
            agent_mask = infos[agent_names[0]]["masks"]
            opponent_mask = infos[agent_names[1]]["masks"]
            
            # Agent action (with value estimate and log prob)
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
            
            # Reward
            agent_reward = next_infos[agent_names[0]]["reward"]
            
            # Store transition
            rollout.push(
                agent_state,
                action_idx,
                agent_reward,
                value,
                log_prob,
                float(terminated or truncated),
                mask_flat
            )
            
            episode_reward += agent_reward
            observations = next_observations
            infos = next_infos
            done = terminated
            step_count += 1
            
            # Safety limit (should rarely hit with 5000 truncation in env)
            if step_count > 5000:
                truncated = True
                if verbose:
                    print(f"  ⚠️  Hit 5k step safety limit")
                break
        
        # Determine winner
        winner = next_infos[agent_names[0]].get("winner", None)
        
        if truncated and winner is None:
            agent_won = None
        elif winner == agent_names[0]:
            agent_won = True
        else:
            agent_won = False
        
        if verbose:
            result = "WON ✓" if agent_won == True else "LOST ✗" if agent_won == False else "DRAW ⚖️"
            print(f"  🏁 Episode finished: {result} | Steps: {step_count} | Reward: {episode_reward:.1f}")
        
        # Debug: Check if rollout is empty
        if len(rollout) == 0:
            print(f"⚠️  WARNING: Episode ended with 0 steps! This shouldn't happen.")
            print(f"   done={done}, truncated={truncated}, winner={winner}")
        
        return rollout, agent_won, episode_reward, step_count
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
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
        """Perform PPO update."""
        # Safety check for empty rollouts
        if len(rollout) == 0:
            print("⚠️  WARNING: Cannot update PPO with empty rollout! Skipping...")
            return 0.0, 0.0
        
        states, actions, rewards, values, old_log_probs, dones, masks = rollout.get()
        
        # Compute advantages
        with torch.no_grad():
            next_value = 0  # Terminal state
        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        masks = masks.to(self.device)
        
        # PPO epochs
        policy_losses = []
        value_losses = []
        
        for _ in range(self.ppo_epochs):
            # Shuffle data
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
                
                # Evaluate actions
                new_log_probs, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions, batch_masks
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values_pred = self.value_net(batch_states)
                value_loss = nn.MSELoss()(values_pred, batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # Update
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
        """Evaluate policy."""
        wins = 0
        draws = 0
        
        print(f"  Playing {num_games} evaluation games...", end='', flush=True)
        
        for i in range(num_games):
            if i > 0 and i % 5 == 0:
                print(f" {i}/{num_games}", end='', flush=True)
            
            rollout, won, _, _ = self.collect_rollout(verbose=False)
            
            if won == True:
                wins += 1
            elif won is None:
                draws += 1
        
        print(f" {num_games}/{num_games} ✓")
        
        return (wins + 0.5 * draws) / num_games
    
    def save_checkpoint(self, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'best_win_rate': self.best_win_rate,
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
        """Main training loop."""
        print("\n" + "🎮"*40)
        print("🚀 PPO REAL ENVIRONMENT TRAINING 🚀")
        print("🎮"*40)
        print(f"\n⏱️  Training Duration: {training_hours} hours")
        print(f"🖥️  Device: {self.device}")
        print(f"📦 Batch Size: {self.batch_size}")
        print(f"🎯 PPO Epochs: {self.ppo_epochs}")
        print(f"✂️  Clip Epsilon: {self.clip_epsilon}")
        print("="*80 + "\n")
        
        self.training_start_time = time.time()
        end_time = self.training_start_time + (training_hours * 3600)
        
        episode_rewards = []
        episode_lengths = []
        wins = []
        
        try:
            while time.time() < end_time:
                self.episode += 1
                verbose = (self.episode == 1 or self.episode % 10 == 0)
                
                # Collect rollout
                rollout, won, reward, length = self.collect_rollout(verbose=verbose)
                
                # PPO update
                policy_loss, value_loss = self.ppo_update(rollout)
                
                # Track metrics
                episode_rewards.append(reward)
                episode_lengths.append(length)
                wins.append(1 if won == True else 0)
                self.total_steps += length
                
                if verbose:
                    print(f"  📊 Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}")
                
                # Logging
                if self.episode % 10 == 0:
                    elapsed = time.time() - self.training_start_time
                    remaining = end_time - time.time()
                    
                    avg_reward = np.mean(episode_rewards[-10:])
                    win_rate = np.mean(wins[-10:])
                    avg_length = np.mean(episode_lengths[-10:])
                    
                    print(f"\n{'='*80}")
                    print(f"📊 Episode {self.episode:4d} Summary")
                    print(f"{'='*80}")
                    print(f"  Reward:   {avg_reward:+.1f}")
                    print(f"  Win Rate: {win_rate:.0%} (last 10 games)")
                    print(f"  Length:   {avg_length:.0f} steps")
                    print(f"  Steps:    {self.total_steps:,}")
                    print(f"  Time:     {elapsed/3600:.2f}h / {training_hours:.2f}h")
                    print(f"{'='*80}\n")
                    
                    self.writer.add_scalar('Train/Reward', avg_reward, self.episode)
                    self.writer.add_scalar('Train/WinRate', win_rate, self.episode)
                    self.writer.add_scalar('Train/EpisodeLength', avg_length, self.episode)
                    self.writer.add_scalar('Train/PolicyLoss', policy_loss, self.episode)
                    self.writer.add_scalar('Train/ValueLoss', value_loss, self.episode)
                
                # Evaluation
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
                
                # Save periodically
                if self.episode % 100 == 0:
                    self.save_checkpoint(is_best=False)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted")
        
        # Final evaluation
        print("\n" + "="*70)
        print("✅ Training Complete!")
        print("="*70)
        
        final_win_rate = self.evaluate(num_games=50)
        print(f"Final win rate: {final_win_rate:.1%}")
        print(f"Best win rate: {self.best_win_rate:.1%}")
        print(f"Total episodes: {self.episode}")
        print(f"Total steps: {self.total_steps:,}")
        
        self.save_checkpoint(is_best=False)
        
        summary = {
            'training_hours': training_hours,
            'total_episodes': self.episode,
            'total_steps': self.total_steps,
            'best_win_rate': self.best_win_rate,
            'final_win_rate': final_win_rate,
            'device': str(self.device)
        }
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Saved to {self.output_dir}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="PPO training")
    parser.add_argument("--bc_checkpoint", type=str, default="checkpoints/bc/best_model.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/ppo_real")
    parser.add_argument("--training_hours", type=float, default=6.0)
    
    args = parser.parse_args()
    
    config = Config()
    
    trainer = PPOTrainer(
        config=config,
        bc_checkpoint=args.bc_checkpoint,
        output_dir=args.output_dir
    )
    
    trainer.train(training_hours=args.training_hours)


if __name__ == "__main__":
    main()
