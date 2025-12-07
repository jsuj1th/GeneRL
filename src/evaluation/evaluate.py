"""
Evaluation Script

Evaluates a trained agent on the test set and against baseline agents.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from models.networks import DuelingDQN
from preprocessing.dataset import ReplayDataset


class Evaluator:
    """Evaluates trained models."""
    
    def __init__(self, config: Config, model_path: str):
        self.config = config
        
        # Setup device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = DuelingDQN(
            num_channels=config.NUM_CHANNELS,
            num_actions=config.NUM_ACTIONS,
            cnn_channels=config.CNN_CHANNELS
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("✓ Model loaded successfully")
    
    def evaluate_on_test_set(self, test_dir: str, batch_size: int = 512):
        """
        Evaluate model accuracy on held-out test replays.
        
        Args:
            test_dir: Directory with test data
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("Evaluating on Test Set")
        print("="*60)
        
        # Load test dataset
        test_dataset = ReplayDataset(test_dir)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print(f"Test samples: {len(test_dataset):,}")
        
        total_correct = 0
        total_samples = 0
        total_valid_actions = 0
        
        with torch.no_grad():
            for states, actions, masks in tqdm(test_loader, desc="Evaluating"):
                # Move to device
                states = states.to(self.device)
                actions = actions.to(self.device)
                masks = masks.to(self.device)
                
                # Get predictions
                q_values = self.model(states)
                q_values_masked = q_values.masked_fill(masks == 0, -1e9)
                predictions = q_values_masked.argmax(dim=1)
                
                # Calculate accuracy
                correct = (predictions == actions).sum().item()
                total_correct += correct
                total_samples += states.size(0)
                
                # Track if predicted actions are valid
                pred_valid = masks.gather(1, predictions.unsqueeze(1)).squeeze(1)
                total_valid_actions += pred_valid.sum().item()
        
        accuracy = 100 * total_correct / total_samples
        valid_action_rate = 100 * total_valid_actions / total_samples
        
        results = {
            'test_accuracy': accuracy,
            'valid_action_rate': valid_action_rate,
            'total_samples': total_samples
        }
        
        print("\n" + "-"*60)
        print("Test Set Results:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Valid Action Rate: {valid_action_rate:.2f}%")
        print(f"  Total Samples: {total_samples:,}")
        print("-"*60)
        
        return results
    
    def evaluate_interactive(self, num_games: int = 100):
        """
        Evaluate agent by playing games against baselines.
        
        This requires a Generals.io environment to be set up.
        For now, this is a placeholder.
        """
        print("\n" + "="*60)
        print("Interactive Evaluation (Placeholder)")
        print("="*60)
        print(f"Requested games: {num_games}")
        print("\n⚠️  Interactive evaluation requires:")
        print("  1. Generals.io environment wrapper")
        print("  2. Baseline agents (Random, Expander)")
        print("  3. Game simulation loop")
        print("\nSee: https://github.com/strakam/generals-bots")
        print("="*60)
        
        return {
            'note': 'Interactive evaluation not yet implemented',
            'next_steps': [
                'Install generals environment',
                'Implement baseline agents',
                'Create game loop',
                'Track win rates and territory'
            ]
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/processed/test",
        help="Directory with test data"
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="Number of games for interactive evaluation"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/evaluation.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = Config()
    
    # Create evaluator
    evaluator = Evaluator(config=config, model_path=args.model_path)
    
    # Evaluate on test set (if available)
    results = {}
    if Path(args.test_dir).exists():
        test_results = evaluator.evaluate_on_test_set(args.test_dir)
        results['test_set'] = test_results
    else:
        print(f"\n⚠️  Test directory not found: {args.test_dir}")
        print("Skipping test set evaluation")
    
    # Interactive evaluation (placeholder)
    interactive_results = evaluator.evaluate_interactive(args.num_games)
    results['interactive'] = interactive_results
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
