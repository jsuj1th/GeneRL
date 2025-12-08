"""
Real Environment DQN Training Script for Generals.io

This script trains a DQN agent using the real Generals.io environment.
Based on the paper: "Artificial Generals Intelligence: Mastering Generals.io with RL"

Key features:
- Real game environment (not surrogate)
- Self-play against opponent pool
- Potential-based reward shaping
- Double DQN with experience replay
- Proper action masking

Requirements:
- Python 3.13+
- generals-bots package
- torch, numpy, gymnasium

Training time: 6-12 hours for significant improvement
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import deque
import numpy as np
import copy
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Suppress NumPy overflow warnings from generals-bots library
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow encountered.*')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from models.networks import DuelingDQN

# Generals environment imports
from generals import GymnasiumGenerals, GridFactory, Agent, Action
import gymnasium as gym


class NeuralAgent(Agent):
    """
    Neural network agent wrapper for Generals.io environment.
    Converts our DQN model to work with the environment's Agent interface.
    """
    
    def __init__(self, model, device, epsilon=0.0, name="DQN_Agent"):
        super().__init__(id=name)
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.name = name
        self.model.eval()
    
    def reset(self):
        """Reset agent state (not needed for our stateless agent)."""
        pass
    
    def act(self, observation, action_mask):
        """
        Select action using the neural network.
        
        Args:
            observation: Game state observation (C, H, W) with 15 channels
            action_mask: Valid actions mask (H, W, 4)
            
        Returns:
            Action object
        """
        # Flatten action mask from (H, W, 4) to (H*W*4)
        # Reshape to (4, H, W) then flatten to match our action space
        h, w = action_mask.shape[:2]
        mask_flat = action_mask.transpose(2, 0, 1).reshape(-1)  # (4, H, W) -> (4*H*W)
        
        # Epsilon-greedy exploration
        if np.random.rand() < self.epsilon:
            # Random valid action
            valid_actions = np.where(mask_flat > 0)[0]
            if len(valid_actions) > 0:
                action_idx = np.random.choice(valid_actions)
            else:
                action_idx = 0
        else:
            # Greedy action from Q-network
            with torch.no_grad():
                # Convert observation to our format (C, H, W)
                state = self.observation_to_state(observation)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                mask_tensor = torch.FloatTensor(mask_flat).unsqueeze(0).to(self.device)
                
                q_values = self.model(state_tensor)
                q_values_masked = q_values.masked_fill(mask_tensor == 0, -1e9)
                action_idx = q_values_masked.argmax(dim=1).item()
        
        # Convert flat action index to Action object
        # observation is (C, H, W), so grid shape is (H, W)
        return self.index_to_action(action_idx, observation.shape[1:])
    
    def observation_to_state(self, observation):
        """
        Convert Generals environment observation to our state format.
        
        Generals obs shape: (C, H, W) with 15 channels:
        0: armies, 1: generals, 2: cities, 3: mountains, 4: neutral_cells,
        5: owned_cells, 6: opponent_cells, 7: fog_cells, 8: structures_in_fog,
        9: owned_land_count, 10: owned_army_count, 11: opponent_land_count,
        12: opponent_army_count, 13: timestep, 14: priority
        
        Our format: (7, H, W) with channels:
        0: terrain (0=empty, 1=mountain, 2=city, 3=general)
        1: my_tiles
        2: enemy_tiles
        3: my_armies
        4: enemy_armies
        5: neutral_armies
        6: fog
        """
        # Observation is already (C, H, W) format from as_tensor()
        c, h, w = observation.shape
        state = np.zeros((7, h, w), dtype=np.float32)
        
        # Channel 0: terrain
        # mountains=1, cities=2, generals=3
        terrain = np.zeros((h, w), dtype=np.float32)
        terrain += observation[3] * 1  # Mountains
        terrain += observation[2] * 2  # Cities
        terrain += observation[1] * 3  # Generals
        state[0] = np.clip(terrain, 0, 3)
        
        # Channel 1: my tiles (owned cells)
        state[1] = observation[5]
        
        # Channel 2: enemy tiles (opponent cells)
        state[2] = observation[6]
        
        # Channel 3: my armies (owned cells * armies)
        state[3] = observation[0] * observation[5]
        
        # Channel 4: enemy armies (opponent cells * armies)
        state[4] = observation[0] * observation[6]
        
        # Channel 5: neutral armies (neutral cells * armies)
        state[5] = observation[0] * observation[4]
        
        # Channel 6: fog
        state[6] = observation[7]
        
        return state
    
    def index_to_action(self, action_idx, grid_shape):
        """
        Convert flat action index to Action object.
        
        Action space: (H * W * 4) where 4 = [up, down, left, right]
        
        Action constructor: Action(to_pass, row, col, direction, to_split)
        Directions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        h, w = grid_shape
        num_cells = h * w
        
        # Determine source cell and direction
        direction_idx = action_idx // num_cells
        cell_idx = action_idx % num_cells
        
        source_row = cell_idx // w
        source_col = cell_idx % w
        
        # Clamp values to valid ranges
        direction_idx = int(np.clip(direction_idx, 0, 3))
        source_row = int(np.clip(source_row, 0, h-1))
        source_col = int(np.clip(source_col, 0, w-1))
        
        # Create Action object (to_pass=False, row, col, direction, to_split=False)
        return Action(
            to_pass=False,
            row=source_row,
            col=source_col,
            direction=direction_idx,  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
            to_split=False  # Full army move
        )


def extract_state_features(observation):
    """Extract game features for reward shaping."""
    # observation shape: (H, W, C)
    obs_t = observation.transpose(2, 0, 1)
    
    agent_land = (obs_t[7] > 0).sum()
    enemy_land = (obs_t[8] > 0).sum()
    agent_army = obs_t[0].sum()
    enemy_army = obs_t[1].sum()
    
    # Count cities and generals
    agent_castles = (obs_t[4] > 0).sum() + (obs_t[2] > 0).sum()
    enemy_castles = (obs_t[5] > 0).sum() + (obs_t[3] > 0).sum()
    
    return {
        'agent_land': max(agent_land, 1),
        'enemy_land': max(enemy_land, 1),
        'agent_army': max(agent_army, 1),
        'enemy_army': max(enemy_army, 1),
        'agent_castles': max(agent_castles, 0.1),
        'enemy_castles': max(enemy_castles, 0.1),
    }


def potential_function(state_features, gamma=0.99):
    """Potential-based reward shaping from paper."""
    max_ratio = 10.0
    
    phi_land = np.log(state_features['agent_land'] / state_features['enemy_land']) / np.log(max_ratio)
    phi_army = np.log(state_features['agent_army'] / state_features['enemy_army']) / np.log(max_ratio)
    phi_castle = np.log(state_features['agent_castles'] / state_features['enemy_castles']) / np.log(max_ratio)
    
    phi_land = np.clip(phi_land, -1.0, 1.0)
    phi_army = np.clip(phi_army, -1.0, 1.0)
    phi_castle = np.clip(phi_castle, -1.0, 1.0)
    
    return 0.3 * phi_land + 0.3 * phi_army + 0.4 * phi_castle


def shaped_reward(state_obs, next_state_obs, original_reward, gamma=0.99):
    """Calculate shaped reward: r_shaped = r + γ*ϕ(s') - ϕ(s)"""
    features = extract_state_features(state_obs)
    next_features = extract_state_features(next_state_obs)
    
    phi_s = potential_function(features, gamma)
    phi_s_next = potential_function(next_features, gamma)
    
    return original_reward + gamma * phi_s_next - phi_s


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            torch.FloatTensor(np.array(masks)),
            torch.FloatTensor(np.array(next_masks))
        )
    
    def __len__(self):
        return len(self.buffer)


class RealEnvDQNTrainer:
    """DQN trainer with real Generals.io environment."""
    
    def __init__(self, config: Config, bc_checkpoint: str, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load BC model
        print(f"\nLoading BC checkpoint: {bc_checkpoint}")
        checkpoint = torch.load(bc_checkpoint, map_location=self.device)
        
        # Initialize Q-network
        self.q_network = DuelingDQN(
            num_channels=config.NUM_CHANNELS,
            num_actions=config.NUM_ACTIONS,
            cnn_channels=config.CNN_CHANNELS
        ).to(self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize target network
        self.target_network = DuelingDQN(
            num_channels=config.NUM_CHANNELS,
            num_actions=config.NUM_ACTIONS,
            cnn_channels=config.CNN_CHANNELS
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        print(f"Model parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.DQN_LEARNING_RATE
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.DQN_BUFFER_SIZE)
        
        # Tracking
        self.steps = 0
        self.epsilon = config.EPSILON_START
        self.episode = 0
        
        # Opponent pool (start with BC model and random opponent for curriculum learning)
        self.opponent_pool = [
            copy.deepcopy(self.q_network.state_dict()),  # BC baseline
            "RANDOM"  # Random opponent for easier initial wins
        ]
        print(f"Opponent pool size: {len(self.opponent_pool)} (BC + Random)")
        
        # TensorBoard
        log_dir = Path("logs/dqn_real_env") / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")
        
        # Metrics
        self.best_win_rate = 0.0
        self.training_start_time = None
    
    def create_environment(self):
        """Create Generals.io environment."""
        grid_factory = GridFactory(
            min_grid_dims=(20, 20),
            max_grid_dims=(28, 28),  # Keep slightly under 30 for padding
        )
        
        agent_names = ["DQN_Agent", "Opponent"]
        
        env = GymnasiumGenerals(
            agents=agent_names,
            grid_factory=grid_factory,
            pad_observations_to=30,  # Pad to 30x30 to match our model
            truncation=100000  # No truncation - games must finish naturally with a winner
        )
        
        return env, agent_names
    
    def play_episode(self, opponent_epsilon=0.1, verbose=False):
        """
        Play one episode of self-play.
        
        Returns:
            Episode data: transitions, winner, length
        """
        if verbose:
            print(f"\n🎮 Starting episode {self.episode + 1}...")
        
        env, agent_names = self.create_environment()
        
        # Create agents
        agent = NeuralAgent(self.q_network, self.device, self.epsilon, "DQN_Agent")
        
        # Curriculum learning: select opponent based on training progress
        # Early training: mostly random opponents (easy wins)
        # Late training: mostly BC opponents (harder challenges)
        if self.episode < 500:
            # First 500 episodes: 80% random, 20% BC
            opponent_type = "RANDOM" if np.random.rand() < 0.8 else self.opponent_pool[0]
        elif self.episode < 1500:
            # Episodes 500-1500: 50% random, 50% BC
            opponent_type = np.random.choice(self.opponent_pool)
        else:
            # After 1500 episodes: 20% random, 80% BC/pool
            opponent_type = "RANDOM" if np.random.rand() < 0.2 else self.opponent_pool[np.random.randint(len([x for x in self.opponent_pool if x != "RANDOM"]))]
        
        if opponent_type == "RANDOM":
            # Create a random opponent with a separate network initialized from BC
            # This prevents both agents from sharing the same network during training
            random_opponent_network = DuelingDQN(
                num_channels=self.config.NUM_CHANNELS,
                num_actions=self.config.NUM_ACTIONS,
                cnn_channels=self.config.CNN_CHANNELS
            ).to(self.device)
            random_opponent_network.load_state_dict(self.opponent_pool[0])  # Use BC weights
            opponent = NeuralAgent(random_opponent_network, self.device, 1.0, "Random_Opponent")
            if verbose:
                print(f"  🎲 Opponent: Random (ε=1.0, separate network)")
        else:
            opponent_network = DuelingDQN(
                num_channels=self.config.NUM_CHANNELS,
                num_actions=self.config.NUM_ACTIONS,
                cnn_channels=self.config.CNN_CHANNELS
            ).to(self.device)
            opponent_network.load_state_dict(opponent_type)
            opponent = NeuralAgent(opponent_network, self.device, opponent_epsilon, "BC_Opponent")
            if verbose:
                print(f"  🤖 Opponent: BC Baseline (ε={opponent_epsilon})")
        
        # Reset environment
        if verbose:
            print("  📋 Resetting environment...")
        observations, infos = env.reset()
        
        if verbose:
            print(f"  ✓ Environment reset. Observation shape: {observations[0].shape} (C, H, W)")
        
        # observations shape: (n_agents, C, H, W) with 15 channels
        agent_obs = observations[0]  # Our agent (C, H, W)
        opponent_obs = observations[1]  # Opponent (C, H, W)
        
        episode_transitions = []
        episode_reward = 0
        done = False
        truncated = False
        step_count = 0
        
        if verbose:
            print("  🏃 Starting game loop...")
        
        while not (done or truncated):
            # Get action masks
            agent_mask = infos[agent_names[0]]["masks"]
            opponent_mask = infos[agent_names[1]]["masks"]
            
            if verbose and step_count % 50 == 0:
                print(f"    Step {step_count}...")
            
            # Select actions
            agent_action = agent.act(agent_obs, agent_mask)
            opponent_action = opponent.act(opponent_obs, opponent_mask)
            
            # Convert Action objects to indices for storage
            # (We'll need this for replay buffer)
            # agent_obs is (C, H, W), so grid shape is (H, W)
            agent_action_idx = self.action_to_index(agent_action, agent_obs.shape[1:])
            
            # Step environment
            actions = [agent_action, opponent_action]
            next_observations, _, terminated, truncated, next_infos = env.step(actions)
            
            # Extract rewards from infos
            agent_reward = next_infos[agent_names[0]]["reward"]
            
            # Calculate shaped reward with dense auxiliary rewards
            next_agent_obs = next_observations[0]
            
            # Add dense rewards for intermediate progress
            dense_reward = 0.0
            
            # Territory expansion reward
            curr_land = (agent_obs[5] > 0).sum()  # owned_cells
            next_land = (next_agent_obs[5] > 0).sum()
            dense_reward += 0.1 * (next_land - curr_land)
            
            # Army growth reward
            curr_army = agent_obs[0].sum()  # total armies
            next_army = next_agent_obs[0].sum()
            dense_reward += 0.01 * (next_army - curr_army)
            
            # City capture reward
            curr_cities = (agent_obs[2] > 0).sum()  # my cities
            next_cities = (next_agent_obs[2] > 0).sum()
            if next_cities > curr_cities:
                dense_reward += 5.0  # Big reward for capturing cities
            
            # Combined reward: original + potential shaping + dense rewards
            shaped_r = shaped_reward(agent_obs, next_agent_obs, agent_reward, self.config.DQN_GAMMA)
            total_reward = shaped_r + dense_reward
            
            # Store transition
            agent_state = agent.observation_to_state(agent_obs)
            next_agent_state = agent.observation_to_state(next_agent_obs)
            next_agent_mask = next_infos[agent_names[0]]["masks"]
            
            # Flatten masks from (H, W, 4) to (4*H*W)
            agent_mask_flat = agent_mask.transpose(2, 0, 1).reshape(-1)
            next_agent_mask_flat = next_agent_mask.transpose(2, 0, 1).reshape(-1)
            
            episode_transitions.append((
                agent_state,
                agent_action_idx,
                total_reward,  # Use combined reward
                next_agent_state,
                float(terminated or truncated),
                agent_mask_flat,
                next_agent_mask_flat
            ))
            
            episode_reward += agent_reward
            
            # Update observations
            agent_obs = next_agent_obs
            opponent_obs = next_observations[1]
            infos = next_infos
            done = terminated
            
            step_count += 1
            
            if step_count > 100000:  # Safety limit (essentially unlimited - games MUST finish naturally)
                truncated = True
                print(f"⚠️  WARNING: Game hit 100k step safety limit! This should never happen.")
                break
        
        # Determine winner and game outcome
        winner = next_infos[agent_names[0]].get("winner", None)
        
        # If truncated with no winner, it's a draw
        if truncated and winner is None:
            agent_won = None  # Draw
            game_result = "DRAW ⚖️"
        elif winner == agent_names[0]:
            agent_won = True
            game_result = "WON ✓"
        else:
            agent_won = False
            game_result = "LOST ✗"
        
        if verbose:
            print(f"  🏁 Episode finished: {game_result} | Steps: {step_count} | Reward: {episode_reward:.1f} | Transitions: {len(episode_transitions)}")
        
        return episode_transitions, agent_won, episode_reward, step_count
    
    def action_to_index(self, action: Action, grid_shape):
        """
        Convert Action object to flat index.
        
        Action format: [to_pass, row, col, direction, to_split]
        Direction: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        h, w = grid_shape
        
        # Extract from Action array
        source_row = int(action[1])
        source_col = int(action[2])
        direction_idx = int(action[3])
        
        # Convert to flat index
        cell_idx = source_row * w + source_col
        num_cells = h * w
        
        return direction_idx * num_cells + cell_idx
    
    def compute_loss(self, batch):
        """Compute DQN loss (Double DQN)."""
        states, actions, rewards, next_states, dones, masks, next_masks = batch
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_masks = next_masks.to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN
        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            next_q_online_masked = next_q_online.masked_fill(next_masks == 0, -1e9)
            next_actions = next_q_online_masked.argmax(dim=1)
            
            next_q_target = self.target_network(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q_values = rewards + (1 - dones) * self.config.DQN_GAMMA * next_q_values
        
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        return loss
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.config.DQN_MIN_BUFFER_SIZE:
            return None
        
        batch = self.replay_buffer.sample(self.config.DQN_BATCH_SIZE)
        loss = self.compute_loss(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        if self.steps % self.config.DQN_TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_network()
        
        self.steps += 1
        return loss.item()
    
    def update_target_network(self):
        """Soft update of target network."""
        tau = self.config.DQN_TAU
        for target_param, param in zip(self.target_network.parameters(), 
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def evaluate_vs_pool(self, num_games=20):
        """
        Evaluate current model against opponent pool.
        
        Returns:
            Win rate (0.0 to 1.0) - draws count as 0.5
        """
        wins = 0
        draws = 0
        
        print(f"  Playing {num_games} evaluation games...", end='', flush=True)
        
        for i in range(num_games):
            if i > 0 and i % 5 == 0:
                print(f" {i}/{num_games}", end='', flush=True)
            _, won, _, _ = self.play_episode(opponent_epsilon=0.0)  # Deterministic opponent
            if won == True:
                wins += 1
            elif won is None:
                draws += 1
        
        print(f" {num_games}/{num_games} ✓")
        
        # Count draws as 0.5 wins for fairer win rate calculation
        return (wins + 0.5 * draws) / num_games
    
    def update_opponent_pool(self, win_rate, threshold=0.45):
        """Update opponent pool if win rate exceeds threshold."""
        if win_rate > threshold:
            self.opponent_pool.append(copy.deepcopy(self.q_network.state_dict()))
            if len(self.opponent_pool) > 3:  # Keep pool size at 3
                self.opponent_pool.pop(0)
            return True
        return False
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'episode': self.episode,
            'steps': self.steps,
            'model_state_dict': self.q_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_win_rate': self.best_win_rate,
            'opponent_pool_size': len(self.opponent_pool),
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
        print("🚀 REAL ENVIRONMENT DQN TRAINING 🚀")
        print("🎮"*40)
        print(f"\n⏱️  Training Duration: {training_hours} hours ({training_hours*60:.0f} minutes)")
        print(f"🖥️  Device: {self.device}")
        print(f"💾 Replay Buffer: {self.config.DQN_BUFFER_SIZE:,} transitions")
        print(f"📦 Batch Size: {self.config.DQN_BATCH_SIZE}")
        print(f"📚 Learning Rate: {self.config.DQN_LEARNING_RATE}")
        print(f"🎲 Epsilon Start: {self.config.EPSILON_START}")
        print(f"🎯 Epsilon End: {self.config.EPSILON_END}")
        print(f"📉 Epsilon Decay: {self.config.EPSILON_DECAY}")
        print(f"🎮 Grid Size: 30x30")
        print(f"🤖 Opponent Pool Size: {len(self.opponent_pool)}")
        print("="*80 + "\n")
        
        self.training_start_time = time.time()
        end_time = self.training_start_time + (training_hours * 3600)
        
        self.q_network.train()
        episode_rewards = []
        episode_lengths = []
        wins = []
        losses = []
        
        try:
            while time.time() < end_time:
                self.episode += 1
                
                # Show verbose output for first episode and every 10th episode
                verbose = (self.episode == 1 or self.episode % 10 == 0)
                
                # Play episode
                transitions, won, reward, length = self.play_episode(verbose=verbose)
                
                # Store transitions in replay buffer
                for transition in transitions:
                    self.replay_buffer.push(*transition)
                
                if verbose:
                    print(f"  💾 Stored {len(transitions)} transitions. Buffer: {len(self.replay_buffer)}/{self.config.DQN_BUFFER_SIZE}")
                
                # Train for multiple steps
                episode_loss = []
                train_steps = 10 if len(self.replay_buffer) >= self.config.DQN_MIN_BUFFER_SIZE else 0
                
                if verbose and train_steps > 0:
                    print(f"  🎯 Training {train_steps} steps...")
                
                for _ in range(train_steps):
                    loss = self.train_step()
                    if loss is not None:
                        episode_loss.append(loss)
                        losses.append(loss)
                
                # Decay epsilon
                self.epsilon = max(
                    self.config.EPSILON_END,
                    self.epsilon * self.config.EPSILON_DECAY
                )
                
                # Track metrics
                episode_rewards.append(reward)
                episode_lengths.append(length)
                # Track wins: 1 for win, 0 for loss/draw (None or False)
                # This treats draws as non-wins for win rate calculation
                wins.append(1 if won == True else 0)
                
                # Logging
                if self.episode % 10 == 0:
                    elapsed = time.time() - self.training_start_time
                    remaining = end_time - time.time()
                    
                    avg_loss = np.mean(episode_loss) if episode_loss else 0
                    avg_reward = np.mean(episode_rewards[-10:])
                    win_rate = np.mean(wins[-10:])
                    avg_length = np.mean(episode_lengths[-10:])
                    
                    print(f"\n{'='*80}")
                    print(f"📊 Episode {self.episode:4d} Summary")
                    print(f"{'='*80}")
                    print(f"  Loss:     {avg_loss:.4f}")
                    print(f"  Reward:   {avg_reward:+.1f}")
                    print(f"  Win Rate: {win_rate:.0%} (last 10 games)")
                    print(f"  Length:   {avg_length:.0f} steps")
                    print(f"  Epsilon:  {self.epsilon:.3f}")
                    print(f"  Buffer:   {len(self.replay_buffer):,} / {self.config.DQN_BUFFER_SIZE:,}")
                    print(f"  Time:     {elapsed/3600:.2f}h / {training_hours:.2f}h ({remaining/3600:.2f}h remaining)")
                    print(f"{'='*80}\n")
                    
                    # TensorBoard
                    self.writer.add_scalar('Train/Loss', avg_loss, self.episode)
                    self.writer.add_scalar('Train/Reward', avg_reward, self.episode)
                    self.writer.add_scalar('Train/WinRate', win_rate, self.episode)
                    self.writer.add_scalar('Train/EpisodeLength', avg_length, self.episode)
                    self.writer.add_scalar('Train/Epsilon', self.epsilon, self.episode)
                    self.writer.add_scalar('Train/BufferSize', len(self.replay_buffer), self.episode)
                
                # Evaluation
                if self.episode % 50 == 0:
                    print("\n" + "🔍" + "="*78 + "🔍")
                    print("📊 EVALUATION: Testing against opponent pool...")
                    print("="*80)
                    win_rate = self.evaluate_vs_pool(num_games=20)
                    print(f"\n✅ Evaluation complete: Win rate = {win_rate:.1%}")
                    
                    self.writer.add_scalar('Eval/WinRate', win_rate, self.episode)
                    
                    is_best = win_rate > self.best_win_rate
                    if is_best:
                        self.best_win_rate = win_rate
                        print(f"  ⭐ New best win rate!")
                    
                    self.save_checkpoint(is_best=is_best)
                    
                    # Update opponent pool
                    if self.update_opponent_pool(win_rate):
                        print(f"  🤖 Updated opponent pool (size: {len(self.opponent_pool)})")
                    
                    print()
                
                # Save checkpoint periodically
                if self.episode % 100 == 0:
                    self.save_checkpoint(is_best=False)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
        
        # Final evaluation
        print("\n" + "="*70)
        print("✅ Training Complete!")
        print("="*70)
        
        final_win_rate = self.evaluate_vs_pool(num_games=50)
        print(f"Final win rate: {final_win_rate:.1%}")
        print(f"Best win rate: {self.best_win_rate:.1%}")
        print(f"Total episodes: {self.episode}")
        print(f"Total steps: {self.steps:,}")
        
        self.save_checkpoint(is_best=False)
        
        # Save training summary
        summary = {
            'training_hours': training_hours,
            'total_episodes': self.episode,
            'total_steps': self.steps,
            'best_win_rate': self.best_win_rate,
            'final_win_rate': final_win_rate,
            'opponent_pool_size': len(self.opponent_pool),
            'device': str(self.device)
        }
        
        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Saved training summary to {self.output_dir / 'training_summary.json'}")
        print(f"📊 TensorBoard logs: logs/dqn_real_env/")
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Real environment DQN training")
    parser.add_argument(
        "--bc_checkpoint",
        type=str,
        default="checkpoints/bc/best_model.pt",
        help="Path to BC checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/dqn_real",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--training_hours",
        type=float,
        default=6.0,
        help="Training duration in hours (can be decimal, e.g., 0.5 for 30 min)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Create trainer
    trainer = RealEnvDQNTrainer(
        config=config,
        bc_checkpoint=args.bc_checkpoint,
        output_dir=args.output_dir
    )
    
    # Train
    trainer.train(training_hours=args.training_hours)


if __name__ == "__main__":
    main()
