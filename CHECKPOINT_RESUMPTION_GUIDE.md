# 🔄 Checkpoint Resumption Guide

**Updated:** December 7, 2024  
**Feature:** Resume PPO training from previous checkpoints

---

## ✨ What's New

The `train_ppo_potential.py` script now supports **checkpoint resumption**! You can:

1. ✅ **Resume training** from where it left off
2. ✅ **Continue for more hours** with the same model
3. ✅ **Preserve all progress** (episodes, steps, win rate, optimizer state)
4. ✅ **Auto-resume** from latest checkpoint automatically

---

## 🚀 Quick Start

### Option 1: Auto-Resume (Easiest)

Automatically resumes from `latest_model.pt` in the output directory:

```bash
python src/training/train_ppo_potential.py \
    --auto_resume \
    --output_dir checkpoints/ppo_from_bc \
    --training_hours 2.0
```

### Option 2: Use Helper Script

```bash
./resume_training.sh
```

This will:
- Check for existing checkpoint
- Show checkpoint info (episode, steps, win rate)
- Ask how many hours to continue training
- Resume automatically

### Option 3: Manual Resume

Specify exact checkpoint path:

```bash
python src/training/train_ppo_potential.py \
    --resume checkpoints/ppo_from_bc/latest_model.pt \
    --output_dir checkpoints/ppo_from_bc \
    --training_hours 2.0
```

---

## 📊 What Gets Restored

When resuming from a checkpoint, the script restores:

| Component | Restored | Details |
|-----------|----------|---------|
| **Policy Network** | ✅ | All weights and parameters |
| **Value Network** | ✅ | All weights and parameters |
| **Policy Optimizer** | ✅ | Adam state (momentum, etc.) |
| **Value Optimizer** | ✅ | Adam state (momentum, etc.) |
| **Episode Counter** | ✅ | Continues from last episode |
| **Total Steps** | ✅ | Cumulative step count |
| **Best Win Rate** | ✅ | Best performance so far |
| **Max Tiles Ever** | ✅ | Maximum exploration achieved |

**Result:** Training continues seamlessly as if it never stopped! 🎉

---

## 🎯 Use Cases

### 1. Training Interrupted

Your training was interrupted (power loss, crash, Ctrl+C):

```bash
# Just resume!
python src/training/train_ppo_potential.py --auto_resume --output_dir checkpoints/ppo_from_bc --training_hours 1.0
```

### 2. Want to Train Longer

You ran 1 hour, but want 2 more hours:

```bash
# First run (1 hour)
python src/training/train_ppo_potential.py --training_hours 1.0

# Continue for 2 more hours
python src/training/train_ppo_potential.py --auto_resume --training_hours 2.0
```

**Total training:** 3 hours cumulative

### 3. Experiment with Different Durations

Test short runs, then extend if promising:

```bash
# Test run: 15 minutes
python src/training/train_ppo_potential.py --training_hours 0.25

# If looking good, continue for 1 hour
python src/training/train_ppo_potential.py --auto_resume --training_hours 1.0

# If still improving, go overnight
python src/training/train_ppo_potential.py --auto_resume --training_hours 8.0
```

### 4. Resume from Different Checkpoint

```bash
# Resume from best model instead of latest
python src/training/train_ppo_potential.py \
    --resume checkpoints/ppo_from_bc/best_model.pt \
    --output_dir checkpoints/ppo_from_best \
    --training_hours 2.0
```

---

## 📋 Command Line Arguments

### New Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--resume` | str | None | Path to specific checkpoint to resume from |
| `--auto_resume` | flag | False | Automatically resume from latest checkpoint in output_dir |

### Existing Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bc_checkpoint` | str | checkpoints/bc/best_model.pt | BC checkpoint for warm start |
| `--output_dir` | str | checkpoints/ppo_potential | Where to save checkpoints |
| `--training_hours` | float | 1.0 | How many hours to train |
| `--from_scratch` | flag | False | Train from scratch (no BC init) |

---

## 🔍 Examples

### Example 1: Basic Resume

```bash
cd "/Users/sujithjulakanti/Desktop/DRL Project"
source venv313/bin/activate

# Start training
python src/training/train_ppo_potential.py --training_hours 1.0

# Later: resume for 1 more hour
python src/training/train_ppo_potential.py --auto_resume --training_hours 1.0
```

**Output:**
```
🔄 Auto-resume enabled: Found checkpoint at checkpoints/ppo_potential/latest_model.pt

🔄 RESUMING from checkpoint: checkpoints/ppo_potential/latest_model.pt
✅ Loaded policy and value networks from checkpoint
   Resuming from episode 150
   Total steps so far: 71,234
   Best win rate: 52.0%
   Max tiles ever: 58

🚀 PPO WITH POTENTIAL-BASED REWARD SHAPING 🚀
⏱️  Training Duration: 1.0 hours
🔄 Training Mode: RESUMING from episode 150
   Previous steps: 71,234
   Previous best win rate: 52.0%
```

### Example 2: Resume with Monitoring

**Terminal 1:**
```bash
python src/training/train_ppo_potential.py --auto_resume --training_hours 2.0
```

**Terminal 2:**
```bash
python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_potential
```

### Example 3: Different Output Directory

```bash
# Start in one directory
python src/training/train_ppo_potential.py --output_dir checkpoints/run1 --training_hours 1.0

# Resume and save to different directory (branches off)
python src/training/train_ppo_potential.py \
    --resume checkpoints/run1/latest_model.pt \
    --output_dir checkpoints/run1_extended \
    --training_hours 2.0
```

### Example 4: Check Checkpoint Info Before Resuming

```bash
python -c "
import torch
ckpt = torch.load('checkpoints/ppo_from_bc/latest_model.pt', map_location='cpu')
print(f'Episode: {ckpt[\"episode\"]}')
print(f'Steps: {ckpt[\"total_steps\"]:,}')
print(f'Win rate: {ckpt.get(\"best_win_rate\", 0):.1%}')
print(f'Max tiles: {ckpt.get(\"max_tiles_ever\", 1)}')
"

# Then resume
python src/training/train_ppo_potential.py --auto_resume --training_hours 1.0
```

---

## ⚠️ Important Notes

### 1. Hyperparameters Are Fixed

When resuming, these are **loaded from checkpoint** (can't change):
- Policy and value network weights
- Optimizer states (momentum, etc.)

These **stay as originally set**:
- Learning rates
- Entropy coefficient
- Batch size
- PPO epochs

### 2. Training Hours Are Additive

If you run:
1. `--training_hours 1.0` → trains 1 hour
2. `--auto_resume --training_hours 2.0` → trains 2 MORE hours

**Total:** 3 hours of training

### 3. Episode Counter Continues

Resume from episode 150 → next episode is 151, 152, etc.

### 4. Checkpoints Are Overwritten

Resuming saves to the same `latest_model.pt` and `best_model.pt` in output_dir.

To keep original checkpoint, use different output_dir:
```bash
python src/training/train_ppo_potential.py \
    --resume checkpoints/ppo_from_bc/latest_model.pt \
    --output_dir checkpoints/ppo_continued \
    --training_hours 2.0
```

---

## 🐛 Troubleshooting

### Error: "No such file or directory"

**Problem:** Checkpoint doesn't exist

**Solution:**
```bash
# Check what checkpoints exist
ls checkpoints/*/latest_model.pt

# Use correct path
python src/training/train_ppo_potential.py --resume checkpoints/ppo_from_bc/latest_model.pt
```

### Error: "Key mismatch" or "Unexpected key"

**Problem:** Checkpoint is from different model architecture

**Solution:** Make sure you're resuming a PPO checkpoint (not BC or DQN):
```bash
# Check checkpoint type
python -c "
import torch
ckpt = torch.load('checkpoints/ppo_from_bc/latest_model.pt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
# Should have: policy_state_dict, value_state_dict
"
```

### Training Seems to Start Over

**Problem:** Not using `--auto_resume` or `--resume`

**Solution:**
```bash
# WITHOUT resume (starts fresh)
python src/training/train_ppo_potential.py --training_hours 1.0

# WITH resume (continues)
python src/training/train_ppo_potential.py --auto_resume --training_hours 1.0
```

### Can't Resume from BC Checkpoint

**Problem:** BC checkpoints don't have value network or PPO state

**Solution:** BC checkpoints are for **warm start only**, not resumption:
```bash
# Start from BC (creates PPO checkpoint)
python src/training/train_ppo_potential.py --bc_checkpoint checkpoints/bc/best_model.pt --training_hours 1.0

# Then resume from PPO checkpoint
python src/training/train_ppo_potential.py --auto_resume --training_hours 1.0
```

---

## 📊 Checkpoint File Structure

### BC Checkpoint (for warm start)
```python
{
    'model_state_dict': {...},  # DuelingDQN weights
    'accuracy': 0.2775,
    # ... BC-specific fields
}
```

### PPO Checkpoint (for resumption)
```python
{
    'episode': 150,
    'total_steps': 71234,
    'policy_state_dict': {...},
    'value_state_dict': {...},
    'policy_optimizer_state_dict': {...},
    'value_optimizer_state_dict': {...},
    'best_win_rate': 0.52,
    'max_tiles_ever': 58,
    'config': {...}
}
```

---

## ✨ Benefits

### 1. **Flexibility**
- Train in sessions (1 hour today, 2 hours tomorrow)
- Stop/start anytime without losing progress
- Experiment with different durations

### 2. **Safety**
- Recover from crashes/interruptions
- Checkpoint saved every 50 episodes
- Always have latest and best models

### 3. **Efficiency**
- No wasted compute redoing training
- Continue from best model
- Extend training only if needed

### 4. **Experimentation**
- Quick test runs first
- Extend promising runs
- Branch from checkpoints to try different settings

---

## 🎯 Recommended Workflow

### Day 1: Initial Training
```bash
# Train for 1 hour to test
python src/training/train_ppo_potential.py --training_hours 1.0
```

### Day 2: Check Results & Continue
```bash
# Check performance
python src/evaluation/diagnose_agent_behavior.py checkpoints/ppo_potential/latest_model.pt

# If promising, continue for 2 more hours
python src/training/train_ppo_potential.py --auto_resume --training_hours 2.0
```

### Day 3: Final Push
```bash
# If still improving, go longer
python src/training/train_ppo_potential.py --auto_resume --training_hours 5.0
```

### Result
**Total:** 8 hours of training, done in 3 manageable sessions! 🎉

---

## 📝 Summary

| Feature | Command |
|---------|---------|
| **Auto-resume** | `--auto_resume` |
| **Specific checkpoint** | `--resume path/to/checkpoint.pt` |
| **Helper script** | `./resume_training.sh` |
| **Check checkpoint** | `python -c "import torch; print(torch.load('...').keys())"` |

**Key Points:**
- ✅ All training state is preserved
- ✅ Training hours are additive
- ✅ Episode counter continues
- ✅ Works with monitoring script
- ✅ Safe to interrupt and resume anytime

**Ready to resume training!** 🚀
