"""
Neural network architectures for Generals.io agent.
Features:
- CNN backbone for spatial processing
- Dueling DQN architecture (value + advantage streams)
- Action masking support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNBackbone(nn.Module):
    """Convolutional backbone for processing game state."""
    
    def __init__(self, in_channels: int, cnn_channels: list, kernel_size: int = 3):
        super().__init__()
        
        layers = []
        prev_channels = in_channels
        
        for out_channels in cnn_channels:
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            prev_channels = out_channels
        
        self.layers = nn.Sequential(*layers)
        self.out_channels = cnn_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DuelingHead(nn.Module):
    """Dueling DQN head: separates value and advantage streams."""
    
    def __init__(self, feature_dim: int, num_actions: int, 
                 value_hidden: int = 256, advantage_hidden: int = 512):
        super().__init__()
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, advantage_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(advantage_hidden, num_actions)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, feature_dim]
        Returns:
            Q-values: [batch_size, num_actions]
        """
        value = self.value_stream(features)  # [batch, 1]
        advantage = self.advantage_stream(features)  # [batch, num_actions]
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class GeneralsAgent(nn.Module):
    """
    Main agent network for Generals.io.
    
    Architecture:
        1. CNN backbone for spatial feature extraction
        2. Global average pooling
        3. Dueling head for Q-value prediction
    
    Can be initialized with either a config object or explicit parameters.
    """
    
    def __init__(self, num_channels=None, num_actions=None, cnn_channels=None, config=None):
        super().__init__()
        
        # Support both config object and explicit parameters
        if config is not None:
            self.config = config
            num_channels = config.NUM_CHANNELS
            num_actions = config.NUM_ACTIONS
            cnn_channels = config.CNN_CHANNELS
            kernel_size = config.KERNEL_SIZE
            value_hidden = config.VALUE_HIDDEN
            advantage_hidden = config.ADVANTAGE_HIDDEN
        else:
            kernel_size = 3
            value_hidden = 256
            advantage_hidden = 512
        
        # CNN backbone
        self.backbone = CNNBackbone(
            in_channels=num_channels,
            cnn_channels=cnn_channels,
            kernel_size=kernel_size
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Calculate feature dimension after GAP
        feature_dim = self.backbone.out_channels
        
        # Dueling head
        self.dueling_head = DuelingHead(
            feature_dim=feature_dim,
            num_actions=num_actions,
            value_hidden=value_hidden,
            advantage_hidden=advantage_hidden
        )
    
    def forward(self, state: torch.Tensor, 
                action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional action masking.
        
        Args:
            state: [batch, channels, height, width] game state
            action_mask: [batch, num_actions] binary mask (1=valid, 0=invalid)
        
        Returns:
            q_values: [batch, num_actions] masked Q-values
        """
        # Extract features
        features = self.backbone(state)  # [batch, channels, height, width]
        
        # Global pooling
        pooled = self.gap(features).squeeze(-1).squeeze(-1)  # [batch, channels]
        
        # Get Q-values
        q_values = self.dueling_head(pooled)  # [batch, num_actions]
        
        # Apply action mask if provided
        if action_mask is not None:
            # Set invalid actions to large negative value
            q_values = q_values.masked_fill(action_mask == 0, float('-inf'))
        
        return q_values
    
    def get_action(self, state: torch.Tensor, action_mask: torch.Tensor, 
                   epsilon: float = 0.0) -> torch.Tensor:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: [batch, channels, height, width]
            action_mask: [batch, num_actions]
            epsilon: exploration rate
        
        Returns:
            actions: [batch] selected actions
        """
        if torch.rand(1).item() < epsilon:
            # Random valid action
            valid_actions = action_mask.float()
            probs = valid_actions / valid_actions.sum(dim=1, keepdim=True)
            actions = torch.multinomial(probs, 1).squeeze(1)
        else:
            # Greedy action
            q_values = self.forward(state, action_mask)
            actions = q_values.argmax(dim=1)
        
        return actions


class BCPolicyNetwork(nn.Module):
    """
    Behavior cloning policy network.
    Uses same backbone as DQN but with softmax policy head.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Shared CNN backbone
        self.backbone = CNNBackbone(
            in_channels=config.NUM_CHANNELS,
            cnn_channels=config.CNN_CHANNELS,
            kernel_size=config.KERNEL_SIZE
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Policy head
        feature_dim = self.backbone.out_channels
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, config.NUM_ACTIONS)
        )
    
    def forward(self, state: torch.Tensor, 
                action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass returning action logits.
        
        Args:
            state: [batch, channels, height, width]
            action_mask: [batch, num_actions]
        
        Returns:
            logits: [batch, num_actions] (masked if action_mask provided)
        """
        # Extract features
        features = self.backbone(state)
        pooled = self.gap(features).squeeze(-1).squeeze(-1)
        
        # Get logits
        logits = self.policy_head(pooled)
        
        # Apply action mask
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))
        
        return logits
    
    def get_action_probs(self, state: torch.Tensor, 
                        action_mask: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        logits = self.forward(state, action_mask)
        return F.softmax(logits, dim=1)
    
    def get_action(self, state: torch.Tensor, 
                   action_mask: torch.Tensor) -> torch.Tensor:
        """Sample action from policy."""
        probs = self.get_action_probs(state, action_mask)
        return torch.multinomial(probs, 1).squeeze(1)


def create_model(config, model_type: str = "dqn"):
    """
    Factory function to create models.
    
    Args:
        config: Configuration object
        model_type: "dqn" or "bc"
    
    Returns:
        model: Neural network model
    """
    if model_type == "dqn":
        return GeneralsAgent(config)
    elif model_type == "bc":
        return BCPolicyNetwork(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_bc_weights_into_dqn(bc_model: BCPolicyNetwork, 
                              dqn_model: GeneralsAgent) -> GeneralsAgent:
    """
    Transfer learned weights from BC model to DQN model.
    Only copies the CNN backbone weights.
    
    Args:
        bc_model: Trained BC model
        dqn_model: DQN model to initialize
    
    Returns:
        dqn_model: DQN model with BC backbone weights
    """
    # Copy backbone weights
    bc_state_dict = bc_model.backbone.state_dict()
    dqn_model.backbone.load_state_dict(bc_state_dict)
    
    print("✓ Transferred BC backbone weights to DQN model")
    return dqn_model


# Alias for compatibility
DuelingDQN = GeneralsAgent


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.append('..')
    from config import Config
    
    config = Config()
    
    # Test BC model
    bc_model = BCPolicyNetwork(config)
    print(f"BC Model parameters: {sum(p.numel() for p in bc_model.parameters()):,}")
    
    # Test DQN model
    dqn_model = GeneralsAgent(config)
    print(f"DQN Model parameters: {sum(p.numel() for p in dqn_model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    state = torch.randn(batch_size, config.NUM_CHANNELS, config.MAP_HEIGHT, config.MAP_WIDTH)
    action_mask = torch.ones(batch_size, config.NUM_ACTIONS)
    
    bc_logits = bc_model(state, action_mask)
    print(f"BC output shape: {bc_logits.shape}")
    
    q_values = dqn_model(state, action_mask)
    print(f"DQN output shape: {q_values.shape}")
    
    print("\n✓ All tests passed!")
