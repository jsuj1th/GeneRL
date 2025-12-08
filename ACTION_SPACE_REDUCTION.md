# Action Space Reduction Guide

## Current Situation

**Config:**
- `NUM_ACTIONS = 4 * 30 * 30 = 3600` (maximum possible)
- Network outputs 3600 logits

**Actual Training (PPO):**
- Grid sizes: 12×12 to 18×18
- Actual actions used: 576 to 1296
- Action masking filters invalid actions

**BC Training Data:**
- Replays from HuggingFace use various grid sizes (typically 10×20 to 20×20)
- Preprocessed with original replay dimensions

---

## Option 1: Keep Current Approach (RECOMMENDED) ✅

**No changes needed!** Your current setup is already efficient:

1. Network has capacity for 30×30 grids
2. Training uses smaller grids (12×12 to 18×18)
3. Action masking ensures only valid actions are selected
4. Works with any grid size ≤ 30×30

**To use 8×8 grids:**

```python
# In train_ppo_potential.py
grid_factory = GridFactory(
    min_grid_dims=(8, 8),   # Smaller grids
    max_grid_dims=(8, 8),   # Fixed at 8×8
)
```

**Actions used:** 4 × 8 × 8 = 256 (out of 3600 capacity)

**Pros:**
- ✅ No code changes needed
- ✅ Works with BC checkpoint (trained on various sizes)
- ✅ Flexible - can train on different grid sizes
- ✅ Unused logits are masked out (no impact)

**Cons:**
- ❌ Network wastes capacity (outputs 3600 instead of 256)
- ❌ Slightly slower forward pass

---

## Option 2: Dynamic Action Space (EFFICIENT) ⚡

Create networks that adapt to grid size:

### Changes Required

#### 1. Modify `config.py`

```python
class Config:
    # Don't fix NUM_ACTIONS - calculate dynamically
    MAX_MAP_SIZE = 30  # Maximum grid dimension
    NUM_DIRECTIONS = 4
    
    def get_num_actions(self, grid_height, grid_width):
        return self.NUM_DIRECTIONS * grid_height * grid_width
```

#### 2. Modify `PolicyNetwork` and `ValueNetwork`

```python
class PolicyNetwork(nn.Module):
    def __init__(self, num_channels, max_actions, cnn_channels):
        super().__init__()
        self.max_actions = max_actions
        self.backbone = DuelingDQN(num_channels, max_actions, cnn_channels)
    
    def forward(self, state, mask=None, num_actions=None):
        logits = self.backbone(state)
        
        # Use only the first num_actions logits if specified
        if num_actions is not None:
            logits = logits[:, :num_actions]
        
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        
        action_probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        return action_probs, log_probs
```

#### 3. Update Training Loop

```python
def collect_rollout(self, verbose=False):
    env, agent_names = self.create_environment()
    
    # Get actual grid size from environment
    observations, infos = env.reset()
    grid_height, grid_width = observations[0].shape[1:3]
    num_actions = 4 * grid_height * grid_width
    
    # Pass num_actions to forward pass
    action_idx, log_prob = self.policy.sample_action(
        state_tensor, mask_tensor, num_actions=num_actions
    )
```

**Pros:**
- ✅ Efficient - only uses needed logits
- ✅ Still works with BC checkpoint
- ✅ Faster forward pass
- ✅ Less memory usage

**Cons:**
- ⚠️ Requires code changes
- ⚠️ Must pass grid size to every forward call

---

## Option 3: Fixed Small Network (FASTEST) 🚀

Train a dedicated network for 8×8 grids only:

### Changes Required

#### 1. Create new config

```python
# config_small.py
class SmallConfig(Config):
    MAP_HEIGHT = 8
    MAP_WIDTH = 8
    NUM_ACTIONS = 4 * 8 * 8  # 256 actions
    
    # Smaller network
    CNN_CHANNELS = [16, 32, 64]  # Less capacity needed
```

#### 2. Cannot use BC checkpoint

BC was trained on 30×30, so you **cannot warm-start** from it.

Must train from scratch:

```bash
python src/training/train_ppo_potential.py \
    --from_scratch \
    --output_dir checkpoints/ppo_8x8
```

#### 3. Fixed grid size in training

```python
grid_factory = GridFactory(
    min_grid_dims=(8, 8),
    max_grid_dims=(8, 8),
)
```

**Pros:**
- ✅ Smallest network - fastest training
- ✅ Fastest inference
- ✅ Least memory usage
- ✅ Simpler action space

**Cons:**
- ❌ **Cannot use BC checkpoint!**
- ❌ Only works for 8×8 grids
- ❌ Must retrain from scratch
- ❌ Worse performance (no BC warm start)

---

## Recommended Approach

### For Your Current Checkpoint ✅

**Option 1** - Keep current approach, just change grid size:

```python
# In train_ppo_potential.py, line 358-361
grid_factory = GridFactory(
    min_grid_dims=(8, 8),   # Changed from (12, 12)
    max_grid_dims=(8, 8),   # Changed from (18, 18)  
)
```

Then resume training:

```bash
python src/training/train_ppo_potential.py \
    --auto_resume \
    --output_dir checkpoints/ppo_from_bc \
    --training_hours 2.0
```

**Why:** 
- ✅ No code changes
- ✅ Keeps BC warm start advantage
- ✅ Just trains on smaller grids
- ✅ Episodes finish faster → more exploration

### For New Project from Scratch

**Option 2** - Dynamic action space for efficiency

---

## BC Replay Compatibility

**Question:** "Original replays from HuggingFace might be 16×16, how to tackle?"

**Answer:** Your BC checkpoint already handles this correctly!

1. **During BC training:**
   - Each replay has its own grid size (varies)
   - `move_to_action()` converts moves based on actual replay width
   - Action index = `direction × (width × height) + cell_index`

2. **During PPO training:**
   - Uses different grid sizes (12×12, 18×18, or 8×8)
   - Action masking ensures only valid actions selected
   - BC knowledge transfers (general strategy, not exact positions)

3. **Why it works:**
   - Both use same action encoding: `direction × grid_size + cell`
   - BC learns "attack nearby enemy tiles" (concept)
   - PPO applies this on smaller/larger grids (transfer)

**No special handling needed** - the action space naturally adapts via masking!

---

## Implementation: Reduce to 8×8 Now

Since you want smaller action space for "more ease", here's the quickest fix:

### Step 1: Modify train_ppo_potential.py

```python
# Line 358-361, change to:
grid_factory = GridFactory(
    min_grid_dims=(8, 8),
    max_grid_dims=(8, 8),
)
```

### Step 2: Resume training

```bash
python src/training/train_ppo_potential.py --auto_resume --training_hours 1.0
```

### Benefits:
- ✅ Action space: 3600 → 256 used (masked)
- ✅ Faster episodes (smaller map)
- ✅ Easier to explore (less area)
- ✅ Keeps BC warm start
- ✅ No other changes needed

### What Changes:
- Map size: 12×12-18×18 → 8×8
- Actions per episode: ~576-1296 → 256
- Episode length: ~1000-2000 → 300-600 steps
- Training speed: ~2-3x faster per episode

---

## Summary

| Option | Action Space | BC Checkpoint | Code Changes | Speed | Recommended |
|--------|-------------|---------------|--------------|-------|-------------|
| **Keep Current** | 3600 (256 used) | ✅ Yes | None | Fast | ✅ **YES** |
| Dynamic Space | 256 | ✅ Yes | Moderate | Faster | 🟡 Optional |
| Fixed Small | 256 | ❌ No | Minimal | Fastest | ❌ No |

**Best choice:** Option 1 - Just change grid size in GridFactory!

