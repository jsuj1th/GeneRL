"""
Replay preprocessing for Generals.io.

This module converts raw replay files into preprocessed training data:
- Extracts state tensors at each decision step
- Computes valid action masks
- Encodes human actions
- Saves as compressed .npz shards for efficient loading

Expected replay format (adjust based on your actual format):
    - Replay should contain sequence of game states
    - Each state has: map, armies, player info, etc.
    - Actions are directional moves from source to target
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


class ReplayPreprocessor:
    """Preprocesses Generals.io replay files."""
    
    def __init__(self, config):
        self.config = config
        self.map_height = config.MAP_HEIGHT
        self.map_width = config.MAP_WIDTH
    
    def load_replay(self, replay_path: Path) -> Optional[Dict]:
        """
        Load a replay file.
        
        Args:
            replay_path: Path to replay file
        
        Returns:
            replay_data: Dictionary containing game states and actions
                         or None if replay is invalid
        """
        try:
            with open(replay_path, 'r') as f:
                replay = json.load(f)
            
            # Basic validation
            if 'states' not in replay or len(replay['states']) < self.config.MIN_GAME_LENGTH:
                return None
            
            if len(replay['states']) > self.config.MAX_GAME_LENGTH:
                return None
            
            return replay
        
        except Exception as e:
            print(f"Error loading {replay_path}: {e}")
            return None
    
    def state_to_tensor(self, state: Dict, player_id: int) -> np.ndarray:
        """
        Convert game state to tensor representation.
        
        Args:
            state: Game state dictionary
            player_id: Player perspective
        
        Returns:
            state_tensor: [channels, height, width] numpy array
        
        Channels:
            0: Owned tiles (1 if owned by player, 0 otherwise)
            1: Enemy tiles (1 if owned by opponent, 0 otherwise)
            2: Army count (normalized)
            3: General locations (1 if general present)
            4: City locations (1 if city present)
            5: Terrain (0=land, 1=mountain, 2=fog)
            6: Fog of war (1 if visible, 0 otherwise)
            7: Normalized turn number
        """
        # Initialize channels
        channels = np.zeros((8, self.map_height, self.map_width), dtype=np.float32)
        
        # Extract map data (adjust based on actual replay format)
        tiles = np.array(state.get('tiles', [])).reshape(self.map_height, self.map_width)
        armies = np.array(state.get('armies', [])).reshape(self.map_height, self.map_width)
        
        # Channel 0: Owned tiles
        channels[0] = (tiles == player_id).astype(np.float32)
        
        # Channel 1: Enemy tiles (any non-player, non-neutral)
        channels[1] = ((tiles != player_id) & (tiles != -1) & (tiles >= 0)).astype(np.float32)
        
        # Channel 2: Army count (log-normalized)
        channels[2] = np.log1p(armies) / 10.0  # Normalize to reasonable range
        
        # Channel 3: Generals
        generals = state.get('generals', [])
        for gen_idx in generals:
            if gen_idx >= 0:
                y, x = divmod(gen_idx, self.map_width)
                if 0 <= y < self.map_height and 0 <= x < self.map_width:
                    channels[3, y, x] = 1.0
        
        # Channel 4: Cities
        cities = state.get('cities', [])
        for city_idx in cities:
            if city_idx >= 0:
                y, x = divmod(city_idx, self.map_width)
                if 0 <= y < self.map_height and 0 <= x < self.map_width:
                    channels[4, y, x] = 1.0
        
        # Channel 5: Terrain (mountains)
        mountains = np.array(state.get('mountains', [])).reshape(self.map_height, self.map_width)
        channels[5] = mountains.astype(np.float32)
        
        # Channel 6: Fog of war (visible tiles)
        visible = np.array(state.get('visible', [])).reshape(self.map_height, self.map_width)
        channels[6] = visible.astype(np.float32)
        
        # Channel 7: Normalized turn number
        turn = state.get('turn', 0)
        channels[7] = np.full((self.map_height, self.map_width), turn / 500.0)
        
        return channels
    
    def get_action_mask(self, state: Dict, player_id: int) -> np.ndarray:
        """
        Compute valid action mask for current state.
        
        Action encoding: action_id = direction * (H*W) + source_idx
            direction: 0=up, 1=down, 2=left, 3=right
            source_idx: tile index (row * W + col)
        
        Args:
            state: Game state
            player_id: Player ID
        
        Returns:
            action_mask: [num_actions] binary array (1=valid, 0=invalid)
        """
        mask = np.zeros(self.config.NUM_ACTIONS, dtype=np.float32)
        
        tiles = np.array(state.get('tiles', [])).reshape(self.map_height, self.map_width)
        armies = np.array(state.get('armies', [])).reshape(self.map_height, self.map_width)
        mountains = np.array(state.get('mountains', [])).reshape(self.map_height, self.map_width)
        
        # For each tile owned by player with armies > 1
        for y in range(self.map_height):
            for x in range(self.map_width):
                if tiles[y, x] == player_id and armies[y, x] > 1:
                    source_idx = y * self.map_width + x
                    
                    # Check each direction
                    directions = [
                        (-1, 0, 0),  # up
                        (1, 0, 1),   # down
                        (0, -1, 2),  # left
                        (0, 1, 3)    # right
                    ]
                    
                    for dy, dx, dir_id in directions:
                        ny, nx = y + dy, x + dx
                        
                        # Check if target is valid (in bounds, not mountain)
                        if (0 <= ny < self.map_height and 
                            0 <= nx < self.map_width and
                            mountains[ny, nx] == 0):
                            
                            action_id = dir_id * (self.map_height * self.map_width) + source_idx
                            mask[action_id] = 1.0
        
        return mask
    
    def encode_action(self, action: Dict) -> int:
        """
        Encode human action to discrete action ID.
        
        Args:
            action: Action dictionary with 'from' and 'to' tile indices
        
        Returns:
            action_id: Discrete action index
        """
        from_idx = action.get('from', -1)
        to_idx = action.get('to', -1)
        
        if from_idx < 0 or to_idx < 0:
            return -1  # Invalid action
        
        # Convert to coordinates
        from_y, from_x = divmod(from_idx, self.map_width)
        to_y, to_x = divmod(to_idx, self.map_width)
        
        # Determine direction
        dy, dx = to_y - from_y, to_x - from_x
        
        if dy == -1 and dx == 0:
            direction = 0  # up
        elif dy == 1 and dx == 0:
            direction = 1  # down
        elif dy == 0 and dx == -1:
            direction = 2  # left
        elif dy == 0 and dx == 1:
            direction = 3  # right
        else:
            return -1  # Invalid move
        
        action_id = direction * (self.map_height * self.map_width) + from_idx
        return action_id
    
    def process_replay(self, replay_path: Path) -> Optional[Dict]:
        """
        Process a single replay file.
        
        Returns:
            data: Dictionary with states, actions, masks, and metadata
        """
        replay = self.load_replay(replay_path)
        if replay is None:
            return None
        
        states = []
        actions = []
        masks = []
        
        player_id = replay.get('player_id', 0)
        
        # Process each decision step
        for t, state in enumerate(replay['states']):
            if t >= len(replay.get('actions', [])):
                break
            
            # Convert state to tensor
            state_tensor = self.state_to_tensor(state, player_id)
            
            # Get valid action mask
            action_mask = self.get_action_mask(state, player_id)
            
            # Encode human action
            action_id = self.encode_action(replay['actions'][t])
            
            if action_id >= 0 and action_mask[action_id] > 0:
                states.append(state_tensor)
                actions.append(action_id)
                masks.append(action_mask)
        
        if len(states) == 0:
            return None
        
        return {
            'states': np.array(states),
            'actions': np.array(actions, dtype=np.int64),
            'masks': np.array(masks),
            'game_length': len(states),
            'replay_id': replay_path.stem
        }
    
    def process_batch(self, replay_paths: List[Path], 
                     output_path: Path, shard_id: int):
        """
        Process a batch of replays and save to disk.
        
        Args:
            replay_paths: List of replay file paths
            output_path: Directory to save processed data
            shard_id: Shard identifier
        """
        all_states = []
        all_actions = []
        all_masks = []
        metadata = []
        
        for replay_path in tqdm(replay_paths, desc=f"Shard {shard_id}", leave=False):
            data = self.process_replay(replay_path)
            if data is not None:
                all_states.append(data['states'])
                all_actions.append(data['actions'])
                all_masks.append(data['masks'])
                metadata.append({
                    'replay_id': data['replay_id'],
                    'game_length': data['game_length'],
                    'start_idx': len(all_actions) - 1
                })
        
        if len(all_states) > 0:
            # Concatenate all data
            states_concat = np.concatenate(all_states, axis=0)
            actions_concat = np.concatenate(all_actions, axis=0)
            masks_concat = np.concatenate(all_masks, axis=0)
            
            # Save shard
            output_file = output_path / f"shard_{shard_id:04d}.npz"
            np.savez_compressed(
                output_file,
                states=states_concat,
                actions=actions_concat,
                masks=masks_concat,
                metadata=metadata
            )
            
            print(f"✓ Saved shard {shard_id}: {len(states_concat):,} transitions")


def preprocess_replays(args):
    """Main preprocessing function."""
    from config import Config
    config = Config()
    
    # Setup paths
    replay_dir = Path(args.replay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all replay files
    replay_files = list(replay_dir.glob("*.json"))
    print(f"Found {len(replay_files):,} replay files")
    
    if len(replay_files) == 0:
        print("Error: No replay files found!")
        return
    
    # Split into shards
    num_shards = (len(replay_files) + args.batch_size - 1) // args.batch_size
    shards = [replay_files[i::num_shards] for i in range(num_shards)]
    
    print(f"Processing {num_shards} shards with {args.num_workers} workers...")
    
    # Process shards
    preprocessor = ReplayPreprocessor(config)
    
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for i, shard in enumerate(shards):
                future = executor.submit(
                    preprocessor.process_batch,
                    shard, output_dir, i
                )
                futures.append(future)
            
            for future in tqdm(futures, desc="Overall progress"):
                future.result()
    else:
        for i, shard in enumerate(tqdm(shards, desc="Processing shards")):
            preprocessor.process_batch(shard, output_dir, i)
    
    print(f"\n✓ Preprocessing complete! Data saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Generals.io replays")
    parser.add_argument("--replay_dir", type=str, required=True,
                       help="Directory containing raw replay files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save processed data")
    parser.add_argument("--batch_size", type=int, default=1000,
                       help="Replays per shard")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    preprocess_replays(args)
