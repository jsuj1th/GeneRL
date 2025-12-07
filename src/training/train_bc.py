"""
Behavior Cloning Training Script

Trains an agent to imitate human replays using supervised learning.
This is the main training phase (16-20 hours) that leverages the 40k training replays.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from models.networks import DuelingDQN
from preprocessing.dataset import ReplayDataset


class BehaviorCloningTrainer:
    """Trains agent using behavior cloning from human replays."""
    
    def __init__(self, config: Config, train_dir: str, val_dir: str, output_dir: str):
        self.config = config
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = DuelingDQN(
            num_channels=config.NUM_CHANNELS,
            num_actions=config.NUM_ACTIONS,
            cnn_channels=config.CNN_CHANNELS
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.BC_LEARNING_RATE,
            weight_decay=config.BC_WEIGHT_DECAY
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function (will be masked cross-entropy)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # TensorBoard
        log_dir = Path("logs/bc") / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")
    
    def create_dataloaders(self, batch_size: int, num_workers: int = 4):
        """Create train and validation dataloaders."""
        print("\nCreating dataloaders...")
        
        train_dataset = ReplayDataset(self.train_dir)
        val_dataset = ReplayDataset(self.val_dir)
        
        print(f"Train samples: {len(train_dataset):,}")
        print(f"Val samples: {len(val_dataset):,}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (states, actions, masks) in enumerate(pbar):
            # Move to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            q_values = self.model(states)
            
            # Mask invalid actions (set to very negative value)
            q_values_masked = q_values.masked_fill(masks == 0, -1e9)
            
            # Compute loss (only on valid actions)
            loss_per_sample = self.criterion(q_values_masked, actions)
            loss = loss_per_sample.mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            with torch.no_grad():
                predictions = q_values_masked.argmax(dim=1)
                correct = (predictions == actions).sum().item()
                
                total_loss += loss.item() * states.size(0)
                total_correct += correct
                total_samples += states.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / states.size(0):.2f}%'
            })
            
            # Log to tensorboard (every 100 batches)
            if batch_idx % 100 == 0:
                global_step = epoch * len(dataloader) + batch_idx
                self.writer.add_scalar('Train/Loss_Batch', loss.item(), global_step)
                self.writer.add_scalar('Train/Accuracy_Batch', 
                                     100 * correct / states.size(0), global_step)
        
        avg_loss = total_loss / total_samples
        avg_acc = 100 * total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self, dataloader, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
            for states, actions, masks in pbar:
                # Move to device
                states = states.to(self.device)
                actions = actions.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                q_values = self.model(states)
                q_values_masked = q_values.masked_fill(masks == 0, -1e9)
                
                # Compute loss
                loss_per_sample = self.criterion(q_values_masked, actions)
                loss = loss_per_sample.mean()
                
                # Track metrics
                predictions = q_values_masked.argmax(dim=1)
                correct = (predictions == actions).sum().item()
                
                total_loss += loss.item() * states.size(0)
                total_correct += correct
                total_samples += states.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / states.size(0):.2f}%'
                })
        
        avg_loss = total_loss / total_samples
        avg_acc = 100 * total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': {
                'num_channels': self.config.NUM_CHANNELS,
                'num_actions': self.config.NUM_ACTIONS,
                'cnn_channels': self.config.CNN_CHANNELS
            }
        }
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss={val_loss:.4f})")
    
    def train(self, num_epochs: int, batch_size: int, patience: int = 10):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Behavior Cloning Training")
        print("="*60)
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(batch_size)
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, epoch)
            
            # Log to tensorboard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Accuracy', val_acc, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered (patience={patience})")
                print(f"Best val loss: {self.best_val_loss:.4f}")
                break
        
        # Save final results
        results = {
            'best_val_loss': float(self.best_val_loss),
            'total_epochs': epoch,
            'config': {
                'batch_size': batch_size,
                'learning_rate': self.config.BC_LEARNING_RATE,
                'num_train_samples': len(train_loader.dataset),
                'num_val_samples': len(val_loader.dataset)
            }
        }
        
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.writer.close()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train BC agent from replays")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with training data"
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Directory with validation data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/bc",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size (default: 1024)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Override config with args
    config.BC_BATCH_SIZE = args.batch_size
    
    # Create trainer
    trainer = BehaviorCloningTrainer(
        config=config,
        train_dir=args.data_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience
    )


if __name__ == "__main__":
    main()
