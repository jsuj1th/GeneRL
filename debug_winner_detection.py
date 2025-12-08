#!/usr/bin/env python3
"""
Debug Winner Detection - Check if we're correctly detecting wins

This script plays a few games and prints detailed winner information
to verify we're correctly identifying when the agent wins.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from models.networks import DuelingDQN
from generals import GymnasiumGenerals, GridFactory, Agent, Action
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow encountered.*')

# Copy the necessary classes
class PolicyNetwork(torch.nn.Module):
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
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action, log_probs.gather(1, action.unsqueeze(1)).squeeze(1)


class PPOAgent(Agent):
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


def play_debug_game(agent_policy, opponent_policy, device, agent_names):
    """Play one game and print detailed winner info."""
    
    # Create environment
    grid_factory = GridFactory(
        min_grid_dims=(12, 12),
        max_grid_dims=(18, 18),
    )
    
    env = GymnasiumGenerals(
        agents=agent_names,
        grid_factory=grid_factory,
        pad_observations_to=30,
        truncation=5000
    )
    
    agent = PPOAgent(agent_policy, device, deterministic=False, name=agent_names[0])
    opponent = PPOAgent(opponent_policy, device, deterministic=False, name=agent_names[1])
    
    observations, infos = env.reset()
    
    done = False
    truncated = False
    step_count = 0
    
    while not (done or truncated) and step_count < 5000:
        agent_obs = observations[0]
        opponent_obs = observations[1]
        
        agent_mask = infos[agent_names[0]]["masks"]
        opponent_mask = infos[agent_names[1]]["masks"]
        
        agent_action = agent.act(agent_obs, agent_mask)
        opponent_action = opponent.act(opponent_obs, opponent_mask)
        
        actions = [agent_action, opponent_action]
        next_observations, _, done, truncated, next_infos = env.step(actions)
        
        observations = next_observations
        infos = next_infos
        step_count += 1
    
    # Debug winner information
    print("\n" + "="*70)
    print("🔍 WINNER DETECTION DEBUG")
    print("="*70)
    print(f"Agent names: {agent_names}")
    print(f"Agent 0 name: '{agent_names[0]}'")
    print(f"Agent 1 name: '{agent_names[1]}'")
    print()
    
    print("Final info for Agent 0:")
    for key, value in next_infos[agent_names[0]].items():
        if key not in ['observation', 'masks']:
            print(f"  {key}: {value}")
    print()
    
    print("Final info for Agent 1:")
    for key, value in next_infos[agent_names[1]].items():
        if key not in ['observation', 'masks']:
            print(f"  {key}: {value}")
    print()
    
    winner = next_infos[agent_names[0]].get("winner", None)
    print(f"Winner field value: '{winner}'")
    print(f"Winner type: {type(winner)}")
    print(f"Winner == agent_names[0]: {winner == agent_names[0]}")
    print(f"Winner == agent_names[1]: {winner == agent_names[1]}")
    print()
    
    # Check our logic
    if winner == agent_names[0]:
        result = "AGENT WON ✓"
        agent_won = True
    elif winner is None:
        result = "DRAW ⚖️"
        agent_won = None
    else:
        result = "AGENT LOST ✗"
        agent_won = False
    
    print(f"Our interpretation: {result}")
    print(f"agent_won variable: {agent_won}")
    
    # Check actual game state
    agent_land = next_infos[agent_names[0]].get('land', 0)
    opponent_land = next_infos[agent_names[1]].get('land', 0)
    agent_has_general = next_infos[agent_names[0]].get('generals', 0) > 0
    opponent_has_general = next_infos[agent_names[1]].get('generals', 0) > 0
    
    print()
    print("Game State:")
    print(f"  Agent land: {agent_land}")
    print(f"  Opponent land: {opponent_land}")
    print(f"  Agent has general: {agent_has_general}")
    print(f"  Opponent has general: {opponent_has_general}")
    print(f"  Done: {done}, Truncated: {truncated}")
    print(f"  Steps: {step_count}")
    
    # Determine actual winner from game state
    if not agent_has_general and opponent_has_general:
        actual_result = "AGENT LOST (general captured)"
    elif agent_has_general and not opponent_has_general:
        actual_result = "AGENT WON (captured opponent general)"
    elif truncated:
        actual_result = "DRAW (truncated)"
    else:
        actual_result = "UNKNOWN"
    
    print()
    print(f"Actual result from game state: {actual_result}")
    print("="*70)
    
    return winner, agent_won, actual_result


def main():
    print("🔍 DEBUGGING WINNER DETECTION IN PPO TRAINING\n")
    
    config = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load PPO checkpoint
    checkpoint_path = "checkpoints/ppo_from_bc/latest_model.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    agent_policy = PolicyNetwork(
        num_channels=config.NUM_CHANNELS,
        num_actions=config.NUM_ACTIONS,
        cnn_channels=config.CNN_CHANNELS
    ).to(device)
    agent_policy.load_state_dict(checkpoint['policy_state_dict'])
    
    # Load BC opponent
    bc_checkpoint = torch.load("checkpoints/bc/best_model.pt", map_location=device)
    opponent_policy = PolicyNetwork(
        num_channels=config.NUM_CHANNELS,
        num_actions=config.NUM_ACTIONS,
        cnn_channels=config.CNN_CHANNELS
    ).to(device)
    opponent_policy.backbone.load_state_dict(bc_checkpoint['model_state_dict'])
    
    # Test with the EXACT same agent names as training
    agent_names = ["PPO_Agent", "Opponent"]
    
    print(f"\nTesting with agent names: {agent_names}")
    print("="*70 + "\n")
    
    # Play 3 test games
    mismatches = []
    for i in range(3):
        print(f"\n{'='*70}")
        print(f"GAME {i+1}/3")
        print(f"{'='*70}")
        winner, agent_won, actual_result = play_debug_game(
            agent_policy, opponent_policy, device, agent_names
        )
        
        # Check if there's a mismatch
        if winner == agent_names[0] and agent_won == False:
            msg = "⚠️  BUG DETECTED: Winner says agent won but we marked it as LOST!"
            print(f"\n{msg}")
            mismatches.append((i+1, msg))
        elif winner != agent_names[0] and winner is not None and agent_won == True:
            msg = "⚠️  BUG DETECTED: Winner says agent lost but we marked it as WON!"
            print(f"\n{msg}")
            mismatches.append((i+1, msg))
        elif (winner == agent_names[0]) != (agent_won == True) and winner is not None:
            msg = "⚠️  MISMATCH between winner field and our interpretation!"
            print(f"\n{msg}")
            mismatches.append((i+1, msg))
        else:
            print("\n✅ Winner detection appears correct for this game")
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    if mismatches:
        print(f"\n❌ Found {len(mismatches)} mismatch(es):")
        for game_num, msg in mismatches:
            print(f"  Game {game_num}: {msg}")
    else:
        print("\n✅ All games had correct winner detection!")
    print("="*70)


if __name__ == "__main__":
    main()
