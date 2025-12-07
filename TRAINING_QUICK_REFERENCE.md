# 🎮 Generals.io Deep RL Training - Quick Reference

## 📊 Current Status
**Training Started:** December 6, 2024 @ 7:04 PM  
**Status:** ✅ **RUNNING SUCCESSFULLY**  
**Current Progress:** Epoch 1/100 (Validation phase)

---

## 📈 Training Progress (Real-time)

### Epoch 1 Results (Just Completed):
- **Training Loss:** ~1.58 (started at 1.67)
- **Training Accuracy:** ~28% (good start!)
- **Validation Loss:** ~1.54 (even better than training!)
- **Training Time:** ~47 seconds per epoch

### Performance Metrics:
- ✅ **Speed:** ~8.5 iterations/second
- ✅ **GPU:** Using MPS (Metal on M4 Mac)
- ✅ **Memory:** Stable
- ✅ **Estimated completion:** 60-90 minutes total

---

## 🚀 Quick Commands

### Check Training Progress
```bash
python3 check_training.py
```

### View TensorBoard (Real-time Graphs)
```bash
tensorboard --logdir logs/bc
# Then open: http://localhost:6006
```

### Check if Training is Running
```bash
ps aux | grep train_bc
```

### Stop Training (if needed)
```bash
pkill -f train_bc
```

---

## 📁 Important Files

### Checkpoints (Auto-saved)
- `checkpoints/bc/latest_model.pt` - Most recent checkpoint
- `checkpoints/bc/best_model.pt` - Best performing model
- `checkpoints/bc/training_results.json` - Final results (when complete)

### Logs
- `logs/bc/20251206-190410/` - TensorBoard logs with training curves

### Training Data
- `data/processed/train/data.npz` - 39,065 training examples
- `data/processed/val/data.npz` - 4,772 validation examples

---

## 🎯 What's Happening

The model is learning to imitate expert Generals.io gameplay through **Behavior Cloning**:

1. **Input:** Game state (7 channels, 30×30 map)
   - Terrain, player tiles, enemy tiles, armies, fog of war

2. **Output:** Action prediction (3,600 possible actions)
   - Source tile (30×30) × direction (4)

3. **Learning:** Supervised learning from 39K expert game states

4. **Goal:** Match expert actions as closely as possible

---

## 📊 Expected Training Behavior

### Early Epochs (1-10):
- Loss decreases rapidly (~1.6 → ~1.2)
- Accuracy improves quickly (~25% → ~40%)
- Model learns basic patterns

### Middle Epochs (11-30):
- Loss decreases slowly (~1.2 → ~0.9)
- Accuracy plateaus (~40-50%)
- Model refines decisions

### Late Epochs (31+):
- Loss stabilizes or increases slightly
- Early stopping likely triggers
- Model has converged

---

## ✅ Success Criteria

**Training is successful if:**
- ✅ Validation accuracy > 40%
- ✅ Validation loss < 1.0
- ✅ Model doesn't overfit (train loss << val loss)

**Current Progress:**
- Validation accuracy: ~29% (Epoch 1)
- Validation loss: ~1.54 (Epoch 1)
- ✨ On track!

---

## ⏭️ Next Steps (After Training)

### 1. Evaluate the Model
```bash
python src/evaluation/evaluate.py
```

### 2. (Optional) DQN Fine-tuning
```bash
python src/training/train_dqn.py --bc_checkpoint checkpoints/bc/best_model.pt
```

### 3. Test Against AI
- Load the model in a Generals.io bot
- Play against other bots or humans
- Measure win rate

---

## 🔍 Monitoring Tips

1. **Check progress every 10-15 minutes** with `python3 check_training.py`
2. **Watch for validation loss** - should decrease steadily
3. **Training should complete in 60-90 minutes** on M4 Air
4. **Don't worry if accuracy seems low** - 40-50% is excellent for this task!

---

## 🐛 Troubleshooting

### Training seems stuck?
```bash
# Check if process is alive
ps aux | grep train_bc

# Check latest output
tail -f logs/bc/20251206-190410/events.out.tfevents.*
```

### Want to restart?
```bash
# Kill current training
pkill -f train_bc

# Restart from scratch
python3 src/training/train_bc.py \
  --data_dir data/processed/train \
  --val_dir data/processed/val \
  --output_dir checkpoints/bc \
  --batch_size 256 \
  --epochs 100 \
  --patience 10 \
  --num_workers 0
```

### System running hot?
- Normal! Training is GPU-intensive
- M4 Mac has excellent cooling
- Consider reducing batch size to 128 if needed

---

## 📝 Notes

- **Model size:** 2.1M parameters (~8MB on disk)
- **Dataset:** 39K training + 4.7K validation examples
- **Training time:** ~1 hour estimated (much faster than originally estimated 12-15 hours!)
- **Why so fast?** Smaller dataset (39K vs 400K+) and efficient M4 GPU

---

**Last Updated:** December 6, 2024 @ 7:10 PM  
**Training Terminal ID:** `0f051a04-5062-4163-8d71-af606d884617`
