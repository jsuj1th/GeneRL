# 📊 HuggingFace Dataset Format - Generals.io Replays

## ✅ Dataset Successfully Accessed!

The HuggingFace dataset `strakammm/generals_io_replays` is working and accessible.

## 📋 Replay Structure

Each replay is a JSON object with the following keys:

```json
{
  "version": 13,                    // Game version
  "id": "--N2EmQuf",               // Unique replay ID
  "mapWidth": 18,                  // Map width
  "mapHeight": 21,                 // Map height
  "usernames": ["player1", "player2"],  // Player names
  "stars": [75, 61],               // Player rankings/stars
  "cities": [12, 45, 67, ...],    // City tile indices
  "cityArmies": [40, 40, 40, ...], // Initial city army counts
  "generals": [89, 164],           // General tile indices (one per player)
  "mountains": [5, 8, 12, ...],   // Mountain tile indices
  "moves": [[...], [...], ...]    // List of moves (main data)
}
```

## 🎮 Key Information

### Map Encoding
- Tiles are represented as **linear indices**: `index = y * mapWidth + x`
- Example: For a 18×21 map, tile at (x=5, y=3) → index = 3*18 + 5 = 59

### Generals
- Each player has one general (starting position)
- `generals[0]` = Player 1's general tile index
- `generals[1]` = Player 2's general tile index

### Cities
- Neutral cities at the start
- `cities` = list of tile indices
- `cityArmies` = initial army counts (usually 40)

### Mountains
- Impassable terrain
- List of tile indices where mountains are located

### Moves
The `moves` array contains the game history. Each move is typically:
```python
[start_tile_index, end_tile_index, is_half, turn_number]
```
- `start_tile_index`: Source tile
- `end_tile_index`: Destination tile (adjacent)
- `is_half`: Boolean (move 50% vs 100% of armies)
- `turn_number`: Which turn this move occurred

## 📝 What We Need to Extract

For training, we need to convert each replay into:

1. **States**: Game board at each turn
   - Territory ownership (player ID per tile)
   - Army counts per tile
   - Fog of war
   - Turn number

2. **Actions**: Human player moves
   - Source tile
   - Direction (up/down/left/right)
   - Action mask (which moves are valid)

3. **Metadata**:
   - Game length
   - Winner
   - Final territories

## ⚠️ Preprocessing Challenge

The current `preprocess_replays.py` script expects a different format. We need to update it to:

1. Parse this compact representation
2. Reconstruct the game state at each turn
3. Convert moves to our action space
4. Generate proper action masks

## 🔧 Next Steps

1. ✅ Download script works (using streaming mode)
2. ⚠️ Update `preprocess_replays.py` to match this format
3. ⚠️ Implement state reconstruction from moves
4. ⚠️ Create proper action encoding

## 📦 Files Downloaded

Successfully tested with 5 replays:
- `data/test_download/train/` - 4 replays
- `data/test_download/val/` - 0 replays (too small sample)
- `data/test_download/test/` - 1 replay

**Ready to scale to 100 or 50,000 replays once preprocessing is updated!**
