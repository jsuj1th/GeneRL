# Monitoring PPO Training - Live Replay Viewing

## 🎯 Goal

Watch PPO agent **during training** to see if it's learning to explore better than BC baseline.

## 🚀 Three Ways to Monitor

### Option 1: Quick Start (Easiest) ⭐

**Watch games every 5 minutes during training:**

```bash
./quick_start_with_monitoring.sh
```

This will:
1. Start PPO training (1 hour)
2. Automatically show games every 5 minutes
3. Display exploration statistics
4. Let you see if agent is improving!

Press `Ctrl+C` to stop monitoring (training continues in background).

---

### Option 2: Manual Monitoring

**Start training first:**
```bash
cd "/Users/sujithjulakanti/Desktop/DRL Project"
source venv313/bin/activate

python src/training/train_ppo_potential.py \
  --bc_checkpoint checkpoints/bc/best_model.pt \
  --output_dir checkpoints/ppo_from_bc \
  --training_hours 1.0 &
```

**Then in another terminal, start monitoring:**
```bash
cd "/Users/sujithjulakanti/Desktop/DRL Project"
source venv313/bin/activate

python src/evaluation/monitor_exploration.py \
  --checkpoint_dir checkpoints/ppo_from_bc \
  --interval 300
```

---

### Option 3: Compare BC vs From-Scratch (Both)

**Run full comparison:**
```bash
chmod +x run_ppo_comparison.sh
./run_ppo_comparison.sh
```

**Then monitor BOTH in separate terminals:**

Terminal 1 (BC warm start):
```bash
python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_bc
```

Terminal 2 (From scratch):
```bash
python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_scratch
```

---

## 📊 What You'll See

Every 5 minutes, the monitor will:

1. **Load latest checkpoint** from training
2. **Play 3 games** with visual rendering (you can watch!)
3. **Show exploration statistics:**
   ```
   📊 CHECKPOINT UPDATE - Episode 50
   ════════════════════════════════════════
     Total steps: 15,000
     Best win rate: 30.0%
     Max tiles ever: 45
   
   🎮 Playing 3 evaluation games with visualization...
   
     Game 1/3: WIN | Steps: 234 | Max tiles: 42 | Cities: 3 | Action diversity: 28.2%
     Game 2/3: LOSS | Steps: 189 | Max tiles: 38 | Cities: 2 | Action diversity: 25.1%
     Game 3/3: WIN | Steps: 267 | Max tiles: 48 | Cities: 4 | Action diversity: 31.5%
   
     📈 Summary:
        Win rate: 2/3 (67%)
        Avg max tiles: 42.7
        Avg cities: 3.0
        Avg action diversity: 28.3%
   
     🔍 Exploration Analysis:
        ✅ GOOD: Agent exploring 43 tiles on average!
        ✅ GOOD: High action diversity (28.3%)
   
     ⏰ Next check in 5 minutes...
   ```

4. **Compare with BC baseline:**
   - BC typically: 10-15 tiles, 17% action diversity
   - Good PPO: 30-50 tiles, 25-35% action diversity

---

## 🔍 What to Look For

### ✅ **GOOD Signs** (PPO is working!)
- ✅ Tiles explored **increasing over time** (20 → 30 → 40+)
- ✅ Action diversity **improving** (20% → 25% → 30%+)
- ✅ Cities captured **consistently** (2-4 per game)
- ✅ Win rate **trending up**

### ⚠️ **WARNING Signs** (PPO struggling)
- ⚠️ Tiles explored **stuck at 10-15** (like BC)
- ⚠️ Action diversity **not improving** (<20%)
- ⚠️ No cities captured
- ⚠️ Win rate stagnant

### ❌ **BAD Signs** (BC warm start failed)
- ❌ Tiles explored **decreasing**
- ❌ Action diversity **dropping**
- ❌ Agent getting worse over time

---

## 📈 Expected Progress Timeline

| Time | Episode | Expected Behavior |
|------|---------|-------------------|
| **0 min** | 1-10 | Like BC: 10-15 tiles, low diversity |
| **10 min** | 50-100 | Starting to explore: 15-25 tiles |
| **20 min** | 100-200 | Clear improvement: 25-35 tiles |
| **30 min** | 200-300 | Good exploration: 30-45 tiles |
| **60 min** | 400-600 | Strong agent: 40-60 tiles, consistent wins |

---

## 🎮 Monitoring Controls

### Change Check Interval
```bash
# Check every 10 minutes instead of 5
python src/evaluation/monitor_exploration.py \
  --checkpoint_dir checkpoints/ppo_from_bc \
  --interval 600
```

### Stop Monitoring
Press `Ctrl+C` (training continues in background)

### Resume Monitoring
```bash
python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_bc
```

### Check Training Progress
```bash
tail -f logs/ppo_training.log
```

---

## 💡 Interpretation Guide

### If PPO Shows Good Exploration:
→ **Answer:** Yes, PPO can overcome BC's poor exploration!  
→ **Action:** Continue training, may even extend to 2-3 hours

### If PPO Stays Like BC:
→ **Answer:** BC initialization is holding PPO back  
→ **Action:** Try training from scratch instead

### If From-Scratch Better Than BC Warm Start:
→ **Answer:** BC was a liability, its habits hurt more than help  
→ **Action:** Always train PPO from scratch for this game

---

## 📝 Example Session

```bash
$ ./quick_start_with_monitoring.sh

==================================================
🚀 QUICK START: PPO WITH LIVE MONITORING
==================================================

This will:
  1. Start PPO training (BC warm start)
  2. Monitor and show games every 5 minutes
  3. Display exploration statistics

You'll be able to watch the agent learn to explore!
==================================================

📊 Starting training at 2025-12-07 18:00:00

1️⃣  Starting PPO training (BC warm start)...
   Training PID: 12345
   Training logs: logs/ppo_training.log

⏳ Waiting for first checkpoint...
✅ First checkpoint detected!

2️⃣  Starting live monitoring...
   You'll see games played every 5 minutes
   Press Ctrl+C to stop monitoring (training will continue)

==================================================

👀 MONITORING PPO TRAINING PROGRESS
══════════════════════════════════════════════════
Checkpoint directory: checkpoints/ppo_from_bc
Check interval: 300 seconds (5.0 minutes)
══════════════════════════════════════════════════

⏳ Waiting for training to start...

📊 CHECKPOINT UPDATE - Episode 30
══════════════════════════════════════════════════
  Total steps: 8,432
  Best win rate: 25.0%
  Max tiles ever: 38

🎮 Playing 3 evaluation games with visualization...

  [You see the game being played with PyGame visualization]

  Game 1/3: WIN | Steps: 245 | Max tiles: 35 | Cities: 2 | Action diversity: 24.5%
  ...
```

---

## 🎯 Your Question Answered

**Q:** "Can I see replays during training to compare them?"

**A:** ✅ **YES!** Run `./quick_start_with_monitoring.sh`

You'll see:
1. Initial games (looks like BC - poor exploration)
2. Mid-training games (improving exploration)
3. Final games (strong exploration, if working)

This lets you **visually confirm** whether PPO is overcoming BC's limitations!

---

## 🔥 Ready to Start?

```bash
# Easiest way - just run this!
./quick_start_with_monitoring.sh
```

Sit back and watch as PPO learns to explore the map! 🎮🚀
