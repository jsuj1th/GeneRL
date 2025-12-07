# Behavior Cloning Training Status

**Started:** December 6, 2024 at 7:04 PM
**Status:** 🔄 IN PROGRESS

---

## Training Configuration

### Data
- **Training samples:** 39,065 (from 9,858 replays)
- **Validation samples:** 4,772 (from 1,231 replays)
- **State dimensions:** (7, 30, 30)
  - 7 channels: terrain, my_tiles, enemy_tiles, my_armies, enemy_armies, neutral_armies, fog
  - 30×30 map (padded to fixed size)
- **Action space:** 3,600 actions (30×30 tiles × 4 directions)

### Model Architecture
- **Type:** Dueling DQN with CNN backbone
- **Parameters:** 2,078,033 (~2M)
- **CNN channels:** [32, 64, 64, 128]
- **Device:** MPS (Metal Performance Shaders on M4 Mac)

### Hyperparameters
- **Batch size:** 256
- **Learning rate:** 0.001 (1e-3)
- **Weight decay:** 0.0001 (1e-4)
- **Max epochs:** 100
- **Early stopping patience:** 10 epochs
- **Optimizer:** Adam
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)

### Performance
- **Training speed:** ~8.5 iterations/second
- **Time per epoch:** ~18 seconds
- **Estimated total time:** ~30 minutes for 100 epochs (if no early stopping)
- **Expected completion:** Much faster than initially estimated (12-15 hours was overestimate)

---

## How to Monitor Training

### Option 1: Quick Check (Terminal)
```bash
python3 check_training.py
```

### Option 2: TensorBoard (Real-time Graphs)
```bash
tensorboard --logdir logs/bc
# Then open: http://localhost:6006
```

### Option 3: Check Output Files
- Latest checkpoint: `checkpoints/bc/latest_model.pt`
- Best checkpoint: `checkpoints/bc/best_model.pt`
- Final results: `checkpoints/bc/training_results.json`

---

## Expected Behavior

### First Epoch
- ✅ Loss starts around ~1.6-1.7
- ✅ Accuracy starts around ~15-25%
- ✅ Both should improve rapidly

### During Training
- Loss should decrease steadily
- Accuracy should increase (target: >40% on validation)
- Learning rate may decrease if validation loss plateaus

### Completion
- Training will stop automatically when:
  1. Validation loss doesn't improve for 10 consecutive epochs (early stopping)
  2. OR 100 epochs are completed
  3. OR training is manually stopped

---

## What Comes Next

After BC training completes:

1. **Evaluate the model** (5-10 minutes)
   ```bash
   python src/evaluation/evaluate.py
   ```

2. **[OPTIONAL] DQN Fine-tuning** (if time permits)
   - Use the BC model as initialization
   - Improve through self-play with DQN
   ```bash
   python src/training/train_dqn.py --bc_checkpoint checkpoints/bc/best_model.pt
   ```

---

## Troubleshooting

### If training seems stuck:
```bash
ps aux | grep train_bc.py  # Check if process is running
```

### If you need to stop training:
```bash
pkill -f train_bc.py
```
The latest checkpoint will be saved and can be resumed.

### If training crashes:
- Check logs in `logs/bc/` directory
- Reduce batch size if memory issues
- Ensure data files are not corrupted

---

## Training Logs Location
- TensorBoard logs: `logs/bc/20251206-190410/`
- Contains:
  - Training/validation loss curves
  - Training/validation accuracy curves
  - Learning rate schedule
  - Batch-level metrics (every 100 batches)

---

**Note:** Initial time estimate of 12-15 hours was conservative. With M4's MPS acceleration and the actual dataset size (39K samples), training is much faster - approximately **30-60 minutes** total.
