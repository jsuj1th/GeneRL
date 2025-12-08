# Quick Start Guide - Three Ways to Train PPO

## 🎯 Goal: Train PPO and see if it overcomes BC's poor exploration

---

## ✅ Option 1: Just Training (Simplest)

**See training progress in real-time:**

```bash
./train_visible.sh
```

**What you see:**
```
🎮 Starting episode 1...
  🎲 Opponent: Random
  🏁 Result: LOST ✗ | Steps: 234 | Max Tiles: 15
  💰 Total Reward: -0.872 (potential-based)
  📊 Policy Loss: 0.3421 | Value Loss: 1.2345

🎮 Starting episode 10...
  🤖 Opponent: BC Baseline
  🏁 Result: WON ✓ | Steps: 189 | Max Tiles: 42
  💰 Total Reward: +1.234 (potential-based)
  
================================================================================
📊 Episode   10 Summary
================================================================================
  Reward:      +0.5
  Win Rate:    40% (last 10)
  Avg Tiles:   32.5 (max ever: 45)
  Avg Cities:  2.3
  Total Steps: 2,547
  Time:        0.15h / 1.00h
================================================================================
```

**Pros:**
- ✅ See ALL training progress
- ✅ Simple, one command
- ✅ Can stop with Ctrl+C

**Cons:**
- ❌ No visual game replays
- ❌ Can't see agent actually playing

---

## ⭐ Option 2: Training + Monitoring (Best!)

**Terminal 1 - Training Progress:**
```bash
./train_with_visible_progress.sh
```

This shows you:
- Training progress every 10 episodes
- Win rates, max tiles, rewards
- Policy/value losses

**Terminal 2 - Game Replays (open separately):**
```bash
cd "/Users/sujithjulakanti/Desktop/DRL Project"
source venv313/bin/activate
python src/evaluation/monitor_exploration.py \
  --checkpoint_dir checkpoints/ppo_from_bc \
  --interval 300
```

This shows you:
- Actual games being played (PyGame window)
- Exploration statistics
- Action diversity analysis
- Every 5 minutes

**Pros:**
- ✅ See training progress (Terminal 1)
- ✅ Watch games visually (Terminal 2)
- ✅ Best of both worlds!

**Cons:**
- ⚠️ Need to open 2 terminals

---

## 🔧 Option 3: Background Training (Original)

**Good for long runs when you don't want to watch:**

```bash
cd "/Users/sujithjulakanti/Desktop/DRL Project"
source venv313/bin/activate

# Start training in background
python src/training/train_ppo_potential.py \
  --bc_checkpoint checkpoints/bc/best_model.pt \
  --output_dir checkpoints/ppo_from_bc \
  --training_hours 1.0 \
  > logs/ppo_training.log 2>&1 &

# Monitor logs
tail -f logs/ppo_training.log
```

**Pros:**
- ✅ Runs in background
- ✅ Can close terminal
- ✅ Good for overnight training

**Cons:**
- ❌ Less interactive
- ❌ Have to tail logs

---

## 📊 What to Look For in Training Progress

### Early Training (Episodes 1-100):
```
  Avg Tiles:   12.5 (max ever: 18)    ← Like BC, poor exploration
  Win Rate:    20%                     ← Losing most games
  Action diversity: 18%                ← Repetitive
```

### Mid Training (Episodes 200-400):
```
  Avg Tiles:   28.3 (max ever: 42)    ← Improving! ✅
  Win Rate:    45%                     ← Winning more
  Action diversity: 25%                ← More varied
```

### Late Training (Episodes 500+):
```
  Avg Tiles:   45.7 (max ever: 68)    ← Great exploration! 🎉
  Win Rate:    65%                     ← Strong performance
  Action diversity: 32%                ← Very diverse
```

---

## 🎯 My Recommendation

**Use Option 2** (Training + Monitoring):

1. Open **Terminal 1**, run:
   ```bash
   ./train_with_visible_progress.sh
   ```

2. Open **Terminal 2**, run:
   ```bash
   cd "/Users/sujithjulakanti/Desktop/DRL Project"
   source venv313/bin/activate
   python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_bc --interval 300
   ```

3. Watch:
   - **Terminal 1:** Numbers going up (tiles, win rate)
   - **Terminal 2:** Agent actually playing games

4. After 1 hour, compare:
   - **Initial:** ~15 tiles, 17% diversity (like BC)
   - **Final:** ~45 tiles, 30% diversity (much better!)

This answers your question: **"Does PPO overcome BC's poor exploration?"**

---

## 🚀 Ready to Start?

**Simplest (just training):**
```bash
./train_visible.sh
```

**Best (training + games):**
```bash
# Terminal 1
./train_with_visible_progress.sh

# Terminal 2 (open new terminal window)
cd "/Users/sujithjulakanti/Desktop/DRL Project" && source venv313/bin/activate && python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_bc --interval 300
```

Choose based on whether you want to see games or just numbers! 🎮
