"""
Diagnose Agent Behavior - Find out why agent is stuck in loops

This script analyzes:
1. Action diversity (are we repeating same actions?)
2. State progression (are we making progress?)
3. Policy entropy (how much exploration?)
4. Territory/army changes over time
"""

import sys
from pathlib import Path
import numpy as np
import torch
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from models.networks import DuelingDQN
from generals import GymnasiumGenerals, GridFactory, Agent, Action
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
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


class DiagnosticAgent(Agent):
    def __init__(self, policy_net, device, name="Agent"):
        super().__init__(id=name)
        self.policy = policy_net
        self.device = device
        self.name = name
        self.policy.eval()
        
        # Tracking
        self.action_history = []
        self.entropy_history = []
        self.territory_history = []
        self.army_history = []
    
    def reset(self):
        self.action_history = []
        self.entropy_history = []
        self.territory_history = []
        self.army_history = []
    
    def act(self, observation, action_mask):
        h, w = action_mask.shape[:2]
        mask_flat = action_mask.transpose(2, 0, 1).reshape(-1)
        
        with torch.no_grad():
            state = self.observation_to_state(observation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_tensor = torch.FloatTensor(mask_flat).unsqueeze(0).to(self.device)
            
            # Get action probabilities
            action_probs, log_probs = self.policy.forward(state_tensor, mask_tensor)
            
            # Calculate entropy
            dist = Categorical(action_probs)
            entropy = dist.entropy().item()
            self.entropy_history.append(entropy)
            
            # Sample action
            action_idx = dist.sample().item()
            
            # Track territory and army
            my_tiles = state[1].sum()
            my_army = state[3].sum()
            self.territory_history.append(my_tiles)
            self.army_history.append(my_army)
            
            # Convert to action
            action = self.index_to_action(action_idx, observation.shape[1:])
            
            # Track action (use string representation)
            action_str = str(action_idx)
            self.action_history.append(action_str)
            
        return action
    
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


def diagnose_agent(checkpoint_path: str, checkpoint_type: str = "ppo"):
    """Diagnose agent behavior."""
    
    print("\n" + "="*70)
    print("🔍 AGENT BEHAVIOR DIAGNOSTIC")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Type: {checkpoint_type.upper()}")
    print("="*70 + "\n")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config = Config()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize policy
    if checkpoint_type == "ppo":
        policy = PolicyNetwork(
            num_channels=config.NUM_CHANNELS,
            num_actions=config.NUM_ACTIONS,
            cnn_channels=config.CNN_CHANNELS
        ).to(device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
    else:  # BC
        policy = PolicyNetwork(
            num_channels=config.NUM_CHANNELS,
            num_actions=config.NUM_ACTIONS,
            cnn_channels=config.CNN_CHANNELS
        ).to(device)
        policy.backbone.load_state_dict(checkpoint['model_state_dict'])
    
    policy.eval()
    
    # Create environment
    grid_factory = GridFactory(
        min_grid_dims=(15, 15),
        max_grid_dims=(20, 20),
    )
    
    agent_names = ["Agent", "Opponent"]
    
    env = GymnasiumGenerals(
        agents=agent_names,
        grid_factory=grid_factory,
        pad_observations_to=30,
        truncation=1000  # Limit to 1000 steps for diagnosis
    )
    
    # Create diagnostic agent
    agent = DiagnosticAgent(policy, device, name="Agent")
    
    # Create opponent (simple BC)
    opponent_policy = PolicyNetwork(
        num_channels=config.NUM_CHANNELS,
        num_actions=config.NUM_ACTIONS,
        cnn_channels=config.CNN_CHANNELS
    ).to(device)
    
    bc_checkpoint = torch.load("checkpoints/bc/best_model.pt", map_location=device)
    opponent_policy.backbone.load_state_dict(bc_checkpoint['model_state_dict'])
    opponent = DiagnosticAgent(opponent_policy, device, name="Opponent")
    
    # Play one game
    print("Playing diagnostic game (max 1000 steps)...\n")
    
    observations, infos = env.reset()
    agent.reset()
    opponent.reset()
    
    done = False
    truncated = False
    step_count = 0
    
    while not (done or truncated):
        agent_obs = observations[0]
        opponent_obs = observations[1]
        
        agent_mask = infos[agent_names[0]]["masks"]
        opponent_mask = infos[agent_names[1]]["masks"]
        
        agent_action = agent.act(agent_obs, agent_mask)
        opponent_action = opponent.act(opponent_obs, opponent_mask)
        
        observations, _, done, truncated, infos = env.step([agent_action, opponent_action])
        
        step_count += 1
        
        if step_count >= 1000:
            truncated = True
    
    # Analysis
    print("="*70)
    print("📊 DIAGNOSTIC RESULTS")
    print("="*70)
    print(f"\n1. BASIC STATS")
    print(f"   Total steps: {step_count}")
    print(f"   Game ended naturally: {'Yes' if done else 'No (truncated)'}")
    
    winner = infos[agent_names[0]].get("winner", None)
    if winner == agent_names[0]:
        result = "WON ✓"
    elif winner is None:
        result = "DRAW ⚖️"
    else:
        result = "LOST ✗"
    print(f"   Result: {result}")
    
    # Action diversity
    print(f"\n2. ACTION DIVERSITY")
    action_counts = Counter(agent.action_history)
    unique_actions = len(action_counts)
    total_actions = len(agent.action_history)
    
    print(f"   Unique actions: {unique_actions}/{total_actions}")
    print(f"   Diversity rate: {unique_actions/total_actions*100:.1f}%")
    
    # Find repetitive patterns
    print(f"\n   Top 5 most repeated actions:")
    for action, count in action_counts.most_common(5):
        pct = count/total_actions*100
        print(f"      {action}: {count} times ({pct:.1f}%)")
    
    # Check for loops (consecutive repeated actions)
    loops = 0
    for i in range(1, len(agent.action_history)):
        if agent.action_history[i] == agent.action_history[i-1]:
            loops += 1
    
    print(f"\n   Consecutive repeats: {loops}/{total_actions} ({loops/total_actions*100:.1f}%)")
    
    # Entropy analysis
    print(f"\n3. POLICY ENTROPY (Exploration)")
    avg_entropy = np.mean(agent.entropy_history)
    min_entropy = np.min(agent.entropy_history)
    max_entropy = np.max(agent.entropy_history)
    
    print(f"   Average entropy: {avg_entropy:.4f}")
    print(f"   Min entropy: {min_entropy:.4f}")
    print(f"   Max entropy: {max_entropy:.4f}")
    print(f"   (Higher = more exploration, Lower = more deterministic)")
    
    if avg_entropy < 1.0:
        print(f"   ⚠️  WARNING: Very low entropy! Agent is too deterministic.")
    elif avg_entropy < 2.0:
        print(f"   ⚠️  CAUTION: Low entropy. Limited exploration.")
    else:
        print(f"   ✓  Good exploration level.")
    
    # Progress analysis
    print(f"\n4. GAME PROGRESS")
    
    territory_start = agent.territory_history[0] if agent.territory_history else 0
    territory_end = agent.territory_history[-1] if agent.territory_history else 0
    territory_change = territory_end - territory_start
    
    army_start = agent.army_history[0] if agent.army_history else 0
    army_end = agent.army_history[-1] if agent.army_history else 0
    army_change = army_end - army_start
    
    print(f"   Territory: {territory_start:.0f} -> {territory_end:.0f} ({territory_change:+.0f})")
    print(f"   Army:      {army_start:.0f} -> {army_end:.0f} ({army_change:+.0f})")
    
    if territory_change <= 0 and army_change <= 0:
        print(f"   ⚠️  WARNING: No progress! Agent is stuck.")
    elif territory_change < 5:
        print(f"   ⚠️  CAUTION: Minimal territory gain.")
    else:
        print(f"   ✓  Making progress.")
    
    # Recommendations
    print(f"\n5. RECOMMENDATIONS")
    
    recommendations = []
    
    if avg_entropy < 1.5:
        recommendations.append("❌ INCREASE entropy bonus (try 0.05 or 0.1 instead of 0.01)")
    
    if loops/total_actions > 0.3:
        recommendations.append("❌ Agent repeating actions too much - add penalty for consecutive repeats")
    
    if territory_change <= 0:
        recommendations.append("❌ No territory gain - increase territory reward (try 1.0 instead of 0.1)")
    
    if unique_actions < total_actions * 0.5:
        recommendations.append("❌ Low action diversity - agent needs more exploration")
    
    if checkpoint_type == "ppo" and step_count < 100:
        recommendations.append("⚠️  Game too short - agent may be losing too quickly")
    
    if not recommendations:
        recommendations.append("✓ Agent behavior looks reasonable")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n" + "="*70)
    print("✅ Diagnosis complete!")
    print("="*70 + "\n")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose agent behavior")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/ppo_fast/latest_model.pt",
                       help="Path to checkpoint")
    parser.add_argument("--type", type=str, default="ppo",
                       choices=["ppo", "bc"],
                       help="Checkpoint type")
    
    args = parser.parse_args()
    
    diagnose_agent(args.checkpoint, args.type)
