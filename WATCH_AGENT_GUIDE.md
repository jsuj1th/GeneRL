# 🎮 Watch Agent Play - Usage Guide

**Updated:** `watch_bc_game.py` now supports **both BC and PPO checkpoints!**

The script automatically detects the checkpoint type and loads it correctly.

---

## 🚀 Quick Start

### Option 1: Watch PPO Agent (Helper Script)

```bash
./watch_ppo.sh
```

This will:
- ✅ Check if PPO checkpoint exists
- ✅ Show checkpoint info (episode, win rate, etc.)
- ✅ Play 3 visual games
- ✅ Use 8×8 grids (matches training)

### Option 2: Direct Python Command

```bash
# Watch PPO agent
python src/evaluation/watch_bc_game.py \
    --checkpoint checkpoints/ppo_from_bc/latest_model.pt \
    --num_games 3

# Watch BC agent
python src/evaluation/watch_bc_game.py \
    --checkpoint checkpoints/bc/best_model.pt \
    --num_games 3
```

---

## 📋 Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | `checkpoints/ppo_from_bc/latest_model.pt` | Path to checkpoint (BC or PPO) |
| `--num_games` | int | 3 | Number of games to watch |
| `--opponent` | str | `bc` | Opponent type: `bc` (mirror) or `random` |
| `--epsilon` | float | 0.0 | Exploration rate (0.0 = greedy, 0.1 = 10% random) |
| `--delay` | float | 0.1 | Delay between steps (seconds) |

---

## 📊 Examples

### Watch PPO Agent (3 games)

```bash
python src/evaluation/watch_bc_game.py \
    --checkpoint checkpoints/ppo_from_bc/latest_model.pt \
    --num_games 3 \
    --opponent bc \
    --delay 0.1
```

**Output:**
```
╔══════════════════════════════════════════════════════════════════════════╗
║              🎮 WATCHING PPO AGENT PLAY GENERALS.IO                      ║
╚══════════════════════════════════════════════════════════════════════════╝

Checkpoint: checkpoints/ppo_from_bc/latest_model.pt
Type: PPO
Games to play: 3
Opponent: BC
Epsilon (exploration): 0.0
Render delay: 0.1s per step

✅ Loaded PPO checkpoint
   Episode: 50
   Total steps: 218,665
   Best win rate: 0.0%

🎮 GAME 1/3
...
```

### Watch BC Agent (baseline)

```bash
python src/evaluation/watch_bc_game.py \
    --checkpoint checkpoints/bc/best_model.pt \
    --num_games 3
```

### Compare PPO vs BC

**Terminal 1 (PPO):**
```bash
python src/evaluation/watch_bc_game.py \
    --checkpoint checkpoints/ppo_from_bc/latest_model.pt \
    --num_games 5
```

**Terminal 2 (BC):**
```bash
python src/evaluation/watch_bc_game.py \
    --checkpoint checkpoints/bc/best_model.pt \
    --num_games 5
```

Then compare:
- Exploration (how much of map agent visits)
- Strategy (attacking vs defending)
- Efficiency (steps to win)

### Slow Motion (for analysis)

```bash
python src/evaluation/watch_bc_game.py \
    --checkpoint checkpoints/ppo_from_bc/latest_model.pt \
    --delay 0.5 \
    --num_games 1
```

Slower playback makes it easier to see what the agent is doing.

### Watch with Exploration

```bash
python src/evaluation/watch_bc_game.py \
    --checkpoint checkpoints/ppo_from_bc/latest_model.pt \
    --epsilon 0.1 \
    --num_games 3
```

With `epsilon=0.1`, agent will take 10% random actions (more exploratory behavior).

### Watch vs Random Opponent

```bash
python src/evaluation/watch_bc_game.py \
    --checkpoint checkpoints/ppo_from_bc/latest_model.pt \
    --opponent random \
    --num_games 3
```

Good for testing if agent can beat weak opponents.

---

## 🔍 What to Watch For

### 1. Exploration Behavior

**Good Exploration:**
- ✅ Agent expands to many different areas
- ✅ Scouts the map to find enemy general
- ✅ Captures cities actively
- ✅ Uses diverse attack directions

**Poor Exploration (BC baseline):**
- ❌ Stays in small area
- ❌ Doesn't actively search for enemy
- ❌ Ignores cities
- ❌ Repetitive movements

### 2. Strategic Quality

**Good Strategy:**
- ✅ Defends own general
- ✅ Attacks enemy general when found
- ✅ Builds army size before attacking
- ✅ Uses terrain (cities, choke points)

**Poor Strategy:**
- ❌ Leaves general undefended
- ❌ Random attacks without purpose
- ❌ Doesn't accumulate armies
- ❌ Ignores strategic positions

### 3. Action Efficiency

**Efficient:**
- ✅ Decisive movements
- ✅ Consolidates armies
- ✅ Direct attacks when beneficial

**Inefficient:**
- ❌ Back-and-forth movements
- ❌ Splits armies unnecessarily
- ❌ Doesn't consolidate forces

---

## 🎯 Interpreting Results

### Game Results

```
🏁 Game 1 Result: WON ✓
   Steps: 342
   Total Reward: 5.3
   Final Territory: 48 tiles
   Final Army: 1,234 units
```

**What this means:**
- **WON** = Agent captured enemy general
- **Steps: 342** = Game took 342 turns (shorter is better)
- **Reward: 5.3** = Potential-based reward (higher is better)
- **Territory: 48/64** = Controlled 75% of map (good exploration!)
- **Army: 1,234** = Strong army size

### Summary Stats

```
📊 FINAL SUMMARY
Wins:   2/3 (66.7%)
Losses: 1/3 (33.3%)
Draws:  0/3 (0.0%)
```

**Interpretation:**
- **66.7% win rate** = Agent is performing well
- **0% draws** = Games are decisive (not timing out)
- **vs BC opponent** = Beating the baseline!

---

## 🐛 Troubleshooting

### Issue: "Checkpoint not found"

```bash
ls checkpoints/ppo_from_bc/
```

If no `latest_model.pt`, you need to train first:
```bash
python src/training/train_ppo_potential.py --auto_resume --training_hours 0.25
```

### Issue: Game window doesn't open

**macOS:**
```bash
# Install pygame properly
pip install --upgrade pygame
```

**Linux:**
```bash
# Install display dependencies
sudo apt-get install python3-pygame
```

### Issue: Game is too fast/slow

Adjust `--delay`:
```bash
# Faster (harder to see)
--delay 0.05

# Slower (easier to analyze)
--delay 0.3

# Very slow (step-by-step)
--delay 1.0
```

### Issue: Agent plays poorly

Check checkpoint episode:
```bash
python -c "import torch; print(torch.load('checkpoints/ppo_from_bc/latest_model.pt', map_location='cpu')['episode'])"
```

If episode < 100, agent is still learning. Train more:
```bash
python src/training/train_ppo_potential.py --auto_resume --training_hours 1.0
```

---

## 🔄 Checkpoint Types

### BC Checkpoint (Behavior Cloning)

**Structure:**
```python
{
    'model_state_dict': {...},  # Network weights
    'test_accuracy': 0.2775,    # 27.75% accuracy
    'epoch': 100,
    ...
}
```

**How it's loaded:**
```python
network.load_state_dict(checkpoint['model_state_dict'])
```

### PPO Checkpoint (Reinforcement Learning)

**Structure:**
```python
{
    'policy_state_dict': {      # Policy network (includes 'backbone.*' prefix)
        'backbone.layers.0.weight': ...,
        'backbone.layers.0.bias': ...,
        ...
    },
    'value_state_dict': {...},  # Value network (separate)
    'episode': 50,              # Training episode
    'total_steps': 218665,      # Total steps trained
    'best_win_rate': 0.0,       # Best win rate achieved
    ...
}
```

**How it's loaded:**
```python
# Extract backbone from policy_state_dict
full_state_dict = checkpoint['policy_state_dict']
backbone_state_dict = {}
for key, value in full_state_dict.items():
    if key.startswith('backbone.'):
        new_key = key.replace('backbone.', '')
        backbone_state_dict[new_key] = value

network.load_state_dict(backbone_state_dict)
```

The script **automatically detects** which type and loads correctly!

---

## 📈 Tracking Improvement

### Watch at Different Training Stages

**Early Training (Episode 0-50):**
```bash
# Copy checkpoint
cp checkpoints/ppo_from_bc/latest_model.pt checkpoints/ppo_ep50.pt

# Watch
python src/evaluation/watch_bc_game.py --checkpoint checkpoints/ppo_ep50.pt
```

**Mid Training (Episode 100-200):**
```bash
# After more training
python src/evaluation/watch_bc_game.py --checkpoint checkpoints/ppo_from_bc/latest_model.pt
```

**Compare:** Did exploration improve? Win rate increase?

### Create Video Recording

```bash
# Use screen recording tools
# macOS: Cmd+Shift+5 → Record Selected Portion
# Linux: OBS Studio or SimpleScreenRecorder

# Play game slowly for recording
python src/evaluation/watch_bc_game.py \
    --checkpoint checkpoints/ppo_from_bc/latest_model.pt \
    --delay 0.2 \
    --num_games 1
```

---

## 🎮 Controls During Playback

- **Close window**: Skip to next game
- **Ctrl+C**: Stop watching entirely
- **Let it run**: Watch full game

---

## 📊 Sample Output

```
╔══════════════════════════════════════════════════════════════════════════╗
║              🎮 WATCHING PPO AGENT PLAY GENERALS.IO                      ║
╚══════════════════════════════════════════════════════════════════════════╝

Checkpoint: checkpoints/ppo_from_bc/latest_model.pt
Type: PPO
Games to play: 3
Opponent: BC
Epsilon (exploration): 0.0
Render delay: 0.1s per step

Using device: mps

✅ Loaded PPO checkpoint
   Episode: 50
   Total steps: 218,665
   Best win rate: 0.0%

🤖 Opponent: PPO Agent (same model)

╔══════════════════════════════════════════════════════════════════════════╗
║                              🎮 GAME 1/3                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

🎬 Starting game... (Close window or Ctrl+C to stop)
   Watching at reduced speed for visibility...

   Step 100: Reward = 2.3
   Step 200: Reward = 4.1
   Step 300: Reward = 5.8

🏁 Game 1 Result: WON ✓
   Steps: 342
   Total Reward: 6.2
   Final Territory: 48 tiles
   Final Army: 1,234 units

[... Games 2 and 3 ...]

╔══════════════════════════════════════════════════════════════════════════╗
║                           📊 FINAL SUMMARY                                ║
╚══════════════════════════════════════════════════════════════════════════╝

Wins:   2/3 (66.7%)
Losses: 1/3 (33.3%)
Draws:  0/3 (0.0%)
```

---

## 🎯 Next Steps

1. **Watch your PPO agent:**
   ```bash
   ./watch_ppo.sh
   ```

2. **Compare with BC baseline:**
   ```bash
   python src/evaluation/watch_bc_game.py --checkpoint checkpoints/bc/best_model.pt
   ```

3. **If PPO exploration looks good:** Continue training!
   ```bash
   python src/training/train_ppo_potential.py --auto_resume --training_hours 2.0
   ```

4. **If PPO exploration looks poor:** Check diagnostics
   ```bash
   python src/evaluation/diagnose_agent_behavior.py checkpoints/ppo_from_bc/latest_model.pt
   ```

---

## ✨ Summary

**Script:** `watch_bc_game.py`  
**Supports:** BC and PPO checkpoints (auto-detect)  
**Default:** PPO from `checkpoints/ppo_from_bc/latest_model.pt`  
**Helper:** `./watch_ppo.sh` for quick visual games  

**Ready to watch your agent play!** 🎮
