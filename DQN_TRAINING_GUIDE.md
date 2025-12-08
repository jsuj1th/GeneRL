# 🎯 DQN Self-Play Training Guide

## Overview

This guide shows how to train a DQN agent using self-play to improve upon the Behavior Cloning baseline.

**Key Features:**
- ✅ Loads BC model as initialization
- ✅ Self-play against opponent pool
- ✅ Potential-based reward shaping (from paper)
- ✅ Uses expert replay data as "surrogate environment"
- ✅ Double DQN with target network
- ✅ Experience replay buffer

## Quick Start

### Option 1: Short Training Run (2 hours)
```bash
python3 src/training/train_dqn_self_play.py \
  --bc_checkpoint checkpoints/bc/best_model.pt \
  --train_dir data/processed/train \
  --val_dir data/processed/val \
  --output_dir checkpoints/dqn \
  --training_hours 2
```

### Option 2: Full Training Run (4-6 hours)
```bash
python3 src/training/train_dqn_self_play.py \
  --bc_checkpoint checkpoints/bc/best_model.pt \
  --train_dir data/processed/train \
  --val_dir data/processed/val \
  --output_dir checkpoints/dqn \
  --training_hours 6
```

### Option 3: Test Run (15 minutes)
```bash
python3 src/training/train_dqn_self_play.py \
  --training_hours 0.25
```

## How It Works

### 1. **Initialization**
- Loads your trained BC model (`checkpoints/bc/best_model.pt`)
- Creates a target network for stable learning
- Initializes opponent pool with BC model

### 2. **Self-Play Training Loop**
```
For each episode:
  1. Sample states from expert replays
  2. Agent selects actions (ε-greedy)
  3. Calculate rewards:
     - Base reward: +0.1 if matches expert, -0.05 otherwise
     - Shaped reward: Uses potential function from paper
  4. Store transitions in replay buffer
  5. Train on batches using Double DQN
  6. Update target network (soft update)
  7. Decay epsilon (exploration rate)
  
Every 50 episodes:
  - Evaluate on validation set
  - Save best model
  - Update opponent pool
```

### 3. **Reward Shaping**

Based on the paper's potential function:
```python
ϕ(s) = 0.3*ϕ_land + 0.3*ϕ_army + 0.4*ϕ_castle

where each component is:
ϕ_x = log(agent_x / enemy_x) / log(max_ratio)

Shaped reward:
r_shaped = r_base + γ*ϕ(s') - ϕ(s)
```

This provides dense feedback every step while preserving optimal policies.

## Expected Results

| Method | Accuracy | Improvement |
|--------|----------|-------------|
| BC Baseline | 27.75% | - |
| + DQN (2h) | ~29-30% | +2-3% |
| + DQN (4h) | ~30-32% | +3-5% |
| + DQN (6h) | ~32-35% | +5-7% |

**Note:** These are conservative estimates. Actual results may vary.

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/dqn_self_play
```

**Metrics to watch:**
- `Train/Loss` - Should decrease over time
- `Train/Reward` - Should increase over time
- `Val/Accuracy` - Should increase over time
- `Val/ValidRate` - Should stay at ~100%
- `Train/Epsilon` - Should decay from 1.0 → 0.1

### Console Output
```
Episode  100 | Loss: 0.0234 | Reward: +0.043 | ε: 0.950 | Buffer: 10000 | Time: 0.2h/4.0h
Episode  200 | Loss: 0.0198 | Reward: +0.067 | ε: 0.903 | Buffer: 20000 | Time: 0.4h/4.0h
...

📊 Validation - Accuracy: 28.45% | Valid rate: 100.00%
  ⭐ New best accuracy!
  💾 Saved best model (accuracy: 28.45%)
```

## Output Files

After training, you'll have:

```
checkpoints/dqn/
├── best_model.pt           # Best model (highest val accuracy)
├── latest_model.pt         # Most recent checkpoint
└── training_summary.json   # Training metrics

logs/dqn_self_play/
└── 20251206-HHMMSS/       # TensorBoard logs
    ├── events.out.tfevents.*
    └── ...
```

## Configuration

Edit `src/config.py` to adjust training parameters:

```python
# DQN Self-Play Parameters
DQN_LEARNING_RATE = 1e-4      # Learning rate
DQN_GAMMA = 0.99               # Discount factor
DQN_BATCH_SIZE = 128           # Batch size
DQN_BUFFER_SIZE = 100000       # Replay buffer size
DQN_MIN_BUFFER_SIZE = 1000     # Min buffer before training
DQN_TARGET_UPDATE_FREQUENCY = 100  # Target network update freq
DQN_TAU = 0.001                # Soft update parameter

# Epsilon decay
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.999
```

## Troubleshooting

### Issue: Training is slow
**Solution:** Reduce `--training_hours` or decrease batch size in config

### Issue: Loss is not decreasing
**Solution:** 
- Check that BC model loaded correctly
- Reduce learning rate
- Increase buffer size before training starts

### Issue: Accuracy not improving
**Solution:**
- Train for longer (6+ hours)
- Adjust reward shaping weights
- Ensure validation set is representative

### Issue: Out of memory
**Solution:**
- Reduce `DQN_BATCH_SIZE` in config
- Reduce `DQN_BUFFER_SIZE` in config
- Close other applications

## Advanced: Using Real Environment

If you have Python 3.11+ and want to use the real Generals.io environment:

```bash
# Install environment
pip install generals

# The environment provides true game dynamics:
from generals.envs import Generals
env = Generals()

# You'll need to:
# 1. Replace the surrogate environment with real env
# 2. Collect actual game trajectories
# 3. Play against different opponents
# 4. Measure true win rates
```

See `DQN_IMPLEMENTATION_GUIDE.md` for more details.

## Evaluation

After training, evaluate your model:

```bash
python3 src/evaluation/evaluate.py \
  --model_path checkpoints/dqn/best_model.pt \
  --test_dir data/processed/test \
  --output_file results/dqn_evaluation.json
```

## Next Steps

1. **Train the model:** Run the training command above
2. **Monitor progress:** Use TensorBoard or console output
3. **Evaluate results:** Check validation accuracy improvements
4. **Compare with BC:** Use evaluation script to compare
5. **Iterate:** Adjust hyperparameters and retrain if needed

## Paper Reference

This implementation is based on:
- **Paper:** "Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning"
- **Authors:** Matej Straka, Martin Schmid
- **arXiv:** 2507.06825v2

**Key differences from paper:**
- Paper uses PPO, we use DQN (simpler, similar results)
- Paper uses real environment, we use replay data surrogate
- Paper trains for 36h on H100, we train 4-6h on M4 Mac
- Paper achieves top 0.003% ranking, we aim for ~5-10% improvement

---

## Questions?

See the comprehensive guides:
- `DQN_IMPLEMENTATION_GUIDE.md` - Research paper findings
- `QUICK_START.md` - Using the BC model
- `EVALUATION_RESULTS.md` - BC model performance
- `PROJECT_FINAL_SUMMARY.md` - Complete project overview
