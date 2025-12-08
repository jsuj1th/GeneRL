# ✅ Monitoring Script Bug Fix Complete

**Date:** December 7, 2024  
**Bug:** `'Action' object has no attribute 'row'` error in monitoring script  
**Status:** **FIXED AND VERIFIED** ✅

---

## 🐛 The Problem

The monitoring script (`src/evaluation/monitor_exploration.py`) was crashing when trying to track unique actions during gameplay:

```python
# OLD CODE (BROKEN) - Line 175
action_key = (agent_action.row, agent_action.col, agent_action.direction)
unique_actions.add(action_key)
```

**Error:**
```
AttributeError: 'Action' object has no attribute 'row'
```

The script was trying to access attributes on the `Action` object returned by the generals-bots library, but the Action object either:
- Uses different attribute names
- Is immutable/doesn't expose those attributes
- Is a different type than expected

---

## 🔧 The Solution

### 1. Modified `PPOAgent` class to track action indices

Added `last_action_idx` attribute to store the action index before conversion to Action object:

```python
class PPOAgent(Agent):
    def __init__(self, policy_net, device, deterministic=False, name="PPO_Agent"):
        super().__init__(id=name)
        # ...existing code...
        self.last_action_idx = None  # Track action index for statistics
    
    def reset(self):
        self.last_action_idx = None
    
    def act(self, observation, action_mask):
        # ...existing code...
        self.last_action_idx = action_idx  # Store for tracking
        return self.index_to_action(action_idx, observation.shape[1:])
```

### 2. Modified `play_game_with_stats()` to use action indices

Changed from accessing Action object attributes to using the tracked action index:

```python
# NEW CODE (WORKING)
# Track unique actions (use action index as string)
if hasattr(agent, 'last_action_idx') and agent.last_action_idx is not None:
    unique_actions.add(str(agent.last_action_idx))
```

This approach:
- ✅ Uses action indices (integers) instead of Action object attributes
- ✅ Converts to string for set storage (same as `diagnose_agent_behavior.py`)
- ✅ Avoids any dependency on Action object internals
- ✅ Consistent with working diagnostic script

---

## ✅ Verification

Created and ran test script (`test_monitoring.py`) that successfully:

1. ✅ Loaded PPO checkpoint from `checkpoints/ppo_from_bc/`
2. ✅ Loaded BC opponent checkpoint
3. ✅ Played a full game without any Action attribute errors
4. ✅ Collected all statistics correctly

**Test Results:**
```
✅ SUCCESS! No Action attribute errors!

======================================================================
📊 GAME STATISTICS
======================================================================
Result: LOSS
Steps: 1984
Max tiles explored: 129
Max army: 7652
Cities captured: 0
Unique actions: 319
Action diversity: 16.1%
Avg tiles: 68.4
Tiles growth: 128
======================================================================

✅ Monitoring script is working correctly!
```

---

## 📁 Files Modified

### 1. `/Users/sujithjulakanti/Desktop/DRL Project/src/evaluation/monitor_exploration.py`

**Changes:**
- Added `self.last_action_idx = None` to `PPOAgent.__init__()`
- Modified `PPOAgent.reset()` to reset action index
- Modified `PPOAgent.act()` to store action index before returning Action
- Changed `play_game_with_stats()` to track actions using index instead of attributes

**Lines changed:** ~3 locations

---

## 🚀 Ready to Use

The monitoring script is now fully functional! You can use it with:

### Option 1: Manual monitoring
```bash
# In terminal 1: Start training
python src/training/train_ppo_potential.py

# In terminal 2: Monitor progress (checks every 5 minutes)
python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_bc
```

### Option 2: Use shell script
```bash
./quick_start_with_monitoring.sh
```

---

## 📊 What the Monitor Shows

Every 5 minutes (or when a new checkpoint is saved), the monitor will:

1. **Load latest checkpoint** from training
2. **Play 3 visual games** against BC opponent
3. **Display statistics:**
   - Win/Loss/Draw results
   - Steps taken
   - Max tiles explored (expansion metric)
   - Cities captured
   - Action diversity (unique actions %)
   - Territory growth

4. **Compare with BC baseline:**
   - BC Agent: ~15 tiles explored, 17% action diversity
   - Target: 30-50 tiles, 25-35% diversity

---

## 🎯 Next Steps

1. ✅ **Bug is fixed** - Monitoring script works correctly
2. **Start training run** - Use potential-based reward shaping
3. **Monitor exploration** - Watch if PPO explores more than BC (15 tiles → 30+ tiles target)
4. **Evaluate results** - After 1 hour, check if PPO overcomes BC's poor exploration

---

## 📝 Key Insights from This Bug

**Why tracking action indices works better:**

1. **Robust**: Action index is always an integer, no parsing needed
2. **Consistent**: Same approach used in working `diagnose_agent_behavior.py`
3. **Complete**: Action index uniquely identifies the action (cell + direction)
4. **Simple**: No dependency on Action object implementation details

**Alternative approaches that could work:**
- Track `(source_row, source_col, direction_idx)` before creating Action
- Use action_idx directly without converting to Action first
- Store action history as indices in agent

**Why accessing Action attributes failed:**
- The generals-bots `Action` class may use different internal representation
- Attributes might be private or use different names
- The Action might be a namedtuple or dataclass with different fields

---

## ✨ Summary

**Before:** Monitoring script crashed with `'Action' object has no attribute 'row'`  
**After:** Monitoring script works perfectly, tracks actions by index  
**Impact:** Can now watch agent exploration during training in real-time  
**Verified:** Test script confirms no errors, statistics collected correctly  

The bug is **completely fixed** and ready for production use! 🎉
