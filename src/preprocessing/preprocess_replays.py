#!/usr/bin/env python3
"""
Preprocess Generals.io replays into training data.

Converts replay JSON files into state-action pairs for behavior cloning.
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


def replay_to_training_data(replay_path):
    """
    Convert a single replay into training examples.
    
    Returns:
        List of (state, action, valid_moves_mask) tuples
    """
    with open(replay_path, 'r') as f:
        replay = json.load(f)
    
    # Extract replay info
    width = replay['mapWidth']
    height = replay['mapHeight']
    generals = replay['generals']
    cities = replay['cities']
    mountains = replay['mountains']
    moves = replay['moves']
    
    # Initialize map state
    map_size = width * height
    
    # Terrain: 0=empty, 1=mountain, 2=city, 3=general
    terrain = np.zeros(map_size, dtype=np.float32)
    for m in mountains:
        terrain[m] = 1
    for c in cities:
        terrain[c] = 2
    for g in generals:
        terrain[g] = 3
    
    # Initialize state (ownership and army counts)
    ownership = np.full(map_size, -1, dtype=np.int32)  # -1=neutral, 0=player0, 1=player1
    armies = np.zeros(map_size, dtype=np.float32)
    
    # Set initial generals
    for player_id, gen_pos in enumerate(generals):
        ownership[gen_pos] = player_id
        armies[gen_pos] = 1
    
    # Process moves and create training examples
    training_examples = []
    turn = 0
    
    for move in moves:
        player_id, from_tile, to_tile, is_half, move_turn = move
        
        # Collect state before move (every 2 turns to reduce data)
        if turn % 2 == 0 and ownership[from_tile] == 0:  # Only learn from player 0
            # Create state representation
            state = create_state_features(
                width, height, terrain, ownership, armies,
                generals[0], player_id=0
            )
            
            # Convert move to action index
            action = move_to_action(from_tile, to_tile, width)
            
            # Create valid moves mask
            valid_mask = create_valid_moves_mask(
                width, height, ownership, armies, player_id=0
            )
            
            if action is not None and valid_mask[action]:
                training_examples.append({
                    'state': state,
                    'action': action,
                    'valid_mask': valid_mask
                })
        
        # Apply move to state
        if from_tile < map_size and to_tile < map_size:
            if armies[from_tile] > 1:
                # Determine army movement
                if is_half:
                    moving_army = armies[from_tile] // 2
                else:
                    moving_army = armies[from_tile] - 1
                
                # Move army
                if ownership[to_tile] == player_id:
                    # Moving to own tile
                    armies[to_tile] += moving_army
                else:
                    # Attacking enemy/neutral tile
                    if armies[to_tile] < moving_army:
                        # Successful attack
                        armies[to_tile] = moving_army - armies[to_tile]
                        ownership[to_tile] = player_id
                    else:
                        # Failed attack
                        armies[to_tile] -= moving_army
                
                armies[from_tile] -= moving_army
        
        # Army increment (every 25 turns)
        if move_turn != turn:
            turn = move_turn
            if turn % 25 == 0:
                for i in range(map_size):
                    if ownership[i] >= 0:  # Owned tile
                        if terrain[i] == 3:  # General
                            armies[i] += 1
                        elif terrain[i] == 2 and turn % 50 == 0:  # City (every 50)
                            armies[i] += 1
    
    return training_examples


def create_state_features(width, height, terrain, ownership, armies, my_general, player_id=0):
    """
    Create state feature tensor.
    
    Returns:
        state: np.array of shape (channels, height, width)
    """
    map_size = width * height
    
    # 7 channels: terrain, my_ownership, enemy_ownership, my_armies, enemy_armies, neutral_armies, fog
    state = np.zeros((7, height, width), dtype=np.float32)
    
    for i in range(map_size):
        y = i // width
        x = i % width
        
        # Terrain channel
        state[0, y, x] = terrain[i] / 3.0  # Normalize
        
        # Ownership channels
        if ownership[i] == player_id:
            state[1, y, x] = 1.0  # My tile
            state[3, y, x] = np.log1p(armies[i]) / 10.0  # My armies (log scale)
        elif ownership[i] >= 0:
            state[2, y, x] = 1.0  # Enemy tile
            state[4, y, x] = np.log1p(armies[i]) / 10.0  # Enemy armies
        else:
            state[5, y, x] = np.log1p(armies[i]) / 10.0  # Neutral armies
        
        # Fog of war (simplified: can see adjacent to owned tiles)
        # For simplicity, we'll assume full visibility in replays
        state[6, y, x] = 1.0
    
    return state


def move_to_action(from_tile, to_tile, width):
    """
    Convert from/to tiles into action index.
    
    Action space: 4 directions (up, down, left, right) × map_size tiles
    """
    from_y = from_tile // width
    from_x = from_tile % width
    to_y = to_tile // width
    to_x = to_tile % width
    
    # Determine direction
    if to_y == from_y - 1 and to_x == from_x:
        direction = 0  # Up
    elif to_y == from_y + 1 and to_x == from_x:
        direction = 1  # Down
    elif to_x == from_x - 1 and to_y == from_y:
        direction = 2  # Left
    elif to_x == from_x + 1 and to_y == from_y:
        direction = 3  # Right
    else:
        return None  # Invalid move (diagonal or non-adjacent)
    
    # Action index: tile_index * 4 + direction
    return from_tile * 4 + direction


def create_valid_moves_mask(width, height, ownership, armies, player_id=0):
    """
    Create boolean mask of valid moves.
    
    Returns:
        valid_mask: np.array of shape (map_size * 4,)
    """
    map_size = width * height
    valid_mask = np.zeros(map_size * 4, dtype=np.bool_)
    
    for tile in range(map_size):
        if ownership[tile] == player_id and armies[tile] > 1:
            y = tile // width
            x = tile % width
            
            # Check 4 directions
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
            for dir_idx, (dy, dx) in enumerate(directions):
                new_y = y + dy
                new_x = x + dx
                
                if 0 <= new_y < height and 0 <= new_x < width:
                    new_tile = new_y * width + new_x
                    # Can move to non-mountain tiles
                    if ownership[new_tile] != -2:  # Not a mountain (we use -1 for neutral)
                        action_idx = tile * 4 + dir_idx
                        valid_mask[action_idx] = True
    
    return valid_mask


def preprocess_split(split_name, input_dir, output_dir):
    """Preprocess all replays in a split."""
    input_path = Path(input_dir) / split_name
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    replay_files = sorted(input_path.glob("replay_*.json"))
    
    print(f"\n🔄 Processing {split_name} split ({len(replay_files)} replays)...")
    
    all_states = []
    all_actions = []
    all_masks = []
    
    skipped = 0
    processed = 0
    total_examples = 0
    
    # Fixed map size for padding (most maps are smaller than 30x30)
    MAX_HEIGHT = 30
    MAX_WIDTH = 30
    MAX_MAP_SIZE = MAX_HEIGHT * MAX_WIDTH
    
    for i, replay_file in enumerate(tqdm(replay_files, desc=f"  {split_name}")):
        try:
            examples = replay_to_training_data(replay_file)
            
            if len(examples) > 0:
                for ex in examples:
                    # Pad state to fixed size
                    state = ex['state']
                    channels, h, w = state.shape
                    
                    # Create padded state
                    padded_state = np.zeros((channels, MAX_HEIGHT, MAX_WIDTH), dtype=np.float32)
                    padded_state[:, :h, :w] = state
                    
                    # Pad action mask
                    mask = ex['valid_mask']
                    padded_mask = np.zeros(MAX_MAP_SIZE * 4, dtype=np.bool_)
                    padded_mask[:len(mask)] = mask
                    
                    all_states.append(padded_state)
                    all_actions.append(ex['action'])
                    all_masks.append(padded_mask)
                
                processed += 1
                total_examples += len(examples)
                
                # Print progress every 1000 replays
                if i % 1000 == 0 and i > 0:
                    print(f"\n  📊 Progress: {i}/{len(replay_files)} | {processed} valid | {total_examples:,} examples | {skipped} skipped", flush=True)
        except Exception as e:
            skipped += 1
            if skipped <= 10:  # Print first 10 errors
                print(f"\n  ⚠️  Skipped {replay_file.name}: {e}", flush=True)
    
    if len(all_states) == 0:
        print(f"  ❌ No valid examples found in {split_name}", flush=True)
        return
    
    print(f"\n  🔄 Converting to numpy arrays...", flush=True)
    # Convert to numpy arrays
    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int64)
    masks = np.array(all_masks, dtype=np.bool_)
    
    print(f"  💾 Saving to disk...", flush=True)
    # Save as compressed numpy files
    np.savez_compressed(
        output_path / "data.npz",
        states=states,
        actions=actions,
        masks=masks
    )
    
    print(f"\n  ✅ {split_name.upper()} COMPLETE!", flush=True)
    print(f"     Processed: {processed}/{len(replay_files)} replays", flush=True)
    print(f"     Skipped: {skipped} replays", flush=True)
    print(f"     Total examples: {len(states):,}", flush=True)
    print(f"     States shape: {states.shape}", flush=True)
    print(f"     Actions shape: {actions.shape}", flush=True)
    print(f"     Masks shape: {masks.shape}", flush=True)


def main():
    """Main preprocessing pipeline."""
    print("="*60)
    print("🔄 Preprocessing Generals.io Replays")
    print("="*60)
    
    input_dir = Path("data/raw")
    output_dir = Path("data/processed")
    
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    
    # Process each split
    for split in ['val', 'train', 'test']:
        preprocess_split(split, input_dir, output_dir)
    
    print("\n" + "="*60)
    print("✅ Preprocessing complete!")
    print("="*60)
    print(f"📁 Processed data saved to: {output_dir}")
    print("\n▶️  Next step: python src/training/train_bc.py")
    print("="*60)


if __name__ == "__main__":
    main()
