# Preprocessing Status

## Current Progress (Running)

**Started:** December 6, 2024
**Dataset:** 18,803 replays (Train: 15,042 | Val: 1,880 | Test: 1,881)

### Processing Status

✅ **Train Split:** In Progress (~23% complete, ~2.5 min remaining)
⏳ **Val Split:** Pending
⏳ **Test Split:** Pending

### Technical Details

- **Padding:** All states padded to 30×30 (max map size)
- **Channels:** 7 (terrain, my_tiles, enemy_tiles, my_armies, enemy_armies, neutral_armies, fog)
- **Action Space:** map_size × 4 directions
- **Progress Reports:** Every 1,000 replays

### Expected Output

```
data/processed/
├── train/
│   └── data.npz (states, actions, masks)
├── val/
│   └── data.npz
└── test/
    └── data.npz
```

### Estimated Timeline

- **Train:** ~3 minutes (15,042 replays)
- **Val:** ~15 seconds (1,880 replays)
- **Test:** ~15 seconds (1,881 replays)
- **Total:** ~4 minutes

## What's Next

After preprocessing completes:

1. **Behavior Cloning Training** (~12-15 hours on M4 Air)
   ```bash
   python src/training/train_bc.py
   ```

2. **Evaluation**
   ```bash
   python src/evaluation/evaluate.py
   ```

## Issues Resolved

✅ Fixed inhomogeneous array shape error by padding all states to fixed 30×30 size
✅ Added detailed progress indicators every 1,000 replays
✅ Improved error reporting (shows first 10 errors)
