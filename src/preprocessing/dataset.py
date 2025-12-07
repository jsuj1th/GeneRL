"""Dataset classes for loading preprocessed replay data."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List
import random


class ReplayDataset(Dataset):
    """
    Dataset for loading preprocessed replay data.
    Supports both single data.npz files and multiple shard files.
    """
    
    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: Directory containing data.npz or shard files
        """
        self.data_dir = Path(data_dir)
        
        # Check if there's a single data.npz file
        single_file = self.data_dir / "data.npz"
        if single_file.exists():
            # Load single file into memory
            print(f"Loading data from {single_file}")
            data = np.load(single_file)
            self.states = data['states']
            self.actions = data['actions']
            self.masks = data['masks']
            self.total_length = len(self.states)
            self.use_shards = False
            print(f"Dataset: {self.total_length:,} transitions")
        else:
            # Use shard files
            self.shard_files = sorted(list(self.data_dir.glob("shard_*.npz")))
            if len(self.shard_files) == 0:
                raise ValueError(f"No data.npz or shard_*.npz files found in {data_dir}")
            
            self.use_shards = True
            # Load metadata from all shards
            self.shard_lengths = []
            self.cumulative_lengths = [0]
            
            for shard_file in self.shard_files:
                data = np.load(shard_file, allow_pickle=True)
                length = len(data['states'])
                self.shard_lengths.append(length)
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
            
            self.total_length = self.cumulative_lengths[-1]
            print(f"Dataset: {len(self.shard_files)} shards, {self.total_length:,} transitions")
    
    def __len__(self) -> int:
        return self.total_length
    
    def _get_shard_and_index(self, idx: int) -> Tuple[int, int]:
        """Convert global index to (shard_id, local_index)."""
        shard_id = 0
        while idx >= self.cumulative_lengths[shard_id + 1]:
            shard_id += 1
        local_idx = idx - self.cumulative_lengths[shard_id]
        return shard_id, local_idx
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single transition.
        
        Returns:
            state: [channels, height, width]
            action: scalar action ID
            mask: [num_actions] valid action mask
        """
        if self.use_shards:
            shard_id, local_idx = self._get_shard_and_index(idx)
            # Load shard (will be cached by OS)
            data = np.load(self.shard_files[shard_id])
            state = torch.from_numpy(data['states'][local_idx]).float()
            action = torch.tensor(data['actions'][local_idx]).long()
            mask = torch.from_numpy(data['masks'][local_idx]).float()
        else:
            # Single file mode - data already in memory
            state = torch.from_numpy(self.states[idx]).float()
            action = torch.tensor(self.actions[idx]).long()
            mask = torch.from_numpy(self.masks[idx]).float()
        
        return state, action, mask


def create_data_splits(data_dir: Path, config) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split preprocessed data into train/val/test sets.
    
    Args:
        data_dir: Directory containing processed shards
        config: Configuration object
    
    Returns:
        train_files, val_files, test_files: Lists of shard file paths
    """
    shard_files = sorted(list(data_dir.glob("shard_*.npz")))
    
    if len(shard_files) == 0:
        raise ValueError(f"No shard files found in {data_dir}")
    
    print(f"Found {len(shard_files)} shard files")
    
    # Shuffle shards
    random.shuffle(shard_files)
    
    # Calculate split sizes
    total_shards = len(shard_files)
    train_size = int(0.6 * total_shards)
    val_size = int(0.3 * total_shards)
    
    # Split
    train_files = shard_files[:train_size]
    val_files = shard_files[train_size:train_size + val_size]
    test_files = shard_files[train_size + val_size:]
    
    print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test shards")
    
    return train_files, val_files, test_files


def create_dataloader(shard_files: List[Path], data_dir: Path, 
                     batch_size: int, num_workers: int = 4,
                     shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for replay data.
    
    Args:
        shard_files: List of shard file paths
        data_dir: Directory containing shards
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = ReplayDataset(data_dir, shard_files)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # Not needed for MPS
        persistent_workers=num_workers > 0
    )
    
    return dataloader


if __name__ == "__main__":
    # Test data loading
    import sys
    sys.path.append('..')
    from config import Config
    
    config = Config()
    data_dir = Path("../../data/processed")
    
    if not data_dir.exists():
        print("Create processed data first!")
    else:
        train_files, val_files, test_files = create_data_splits(data_dir, config)
        
        # Test train loader
        train_loader = create_dataloader(
            train_files, data_dir,
            batch_size=32, num_workers=0, shuffle=True
        )
        
        print("\nTesting data loader...")
        for state, action, mask in train_loader:
            print(f"State shape: {state.shape}")
            print(f"Action shape: {action.shape}")
            print(f"Mask shape: {mask.shape}")
            break
        
        print("✓ Data loading test passed!")
