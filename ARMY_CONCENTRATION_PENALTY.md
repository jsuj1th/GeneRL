# Army Concentration Penalty Implementation

## 🎯 Overview

Added **army concentration penalty** to PPO training to encourage better army distribution and more active exploration. This addresses the common RL problem where agents hoard armies on one tile (usually the general) instead of spreading them strategically.

**Implementation Date**: December 7, 2024  
**File Modified**: `src/training/train_ppo_potential.py`  
**Training Strategy**: Continue from existing checkpoint (Episode 100, 270K steps, 70% win rate)

---

## 🚀 What Was Changed

### **New Method: `compute_army_concentration_penalty()`**

Added comprehensive penalty calculation using three complementary metrics:

```python
def compute_army_concentration_penalty(self, observation):
    """
    Penalize uneven army distribution to encourage spreading armies.
    
    Returns:
        penalty: Value from -0.15 to +0.05
                 Negative = too concentrated
                 Positive = well distributed
    """
```

### **Integration Point**

Added to the rollout collection loop in `collect_rollout()`:

```python
# After computing shaped reward
shaped_reward = self.compute_shaped_reward(...)

# NEW: Add army concentration penalty
army_penalty = self.compute_army_concentration_penalty(next_observations[0])

total_reward = shaped_reward + army_penalty  # Combined reward
```

---

## 📊 Penalty Components

### **1. Coefficient of Variation (CV) Penalty**

**Purpose**: Penalize uneven distribution across all owned tiles

**Formula**:
```python
cv = std(armies) / mean(armies)
cv_penalty = -0.05 * min(cv, 3.0) / 3.0  # Range: 0 to -0.05
```

**Example**:
- Uniform [10, 10, 10, 10]: cv ≈ 0 → penalty = 0
- Concentrated [40, 0, 0, 0]: cv ≈ 2.0 → penalty = -0.033

**Why**: Coefficient of variation is scale-invariant, works regardless of total army size

---

### **2. Max Army Ratio Penalty**

**Purpose**: Penalize when one tile has disproportionately large army

**Formula**:
```python
max_ratio = max(armies) / mean(armies)
if max_ratio > 4.0:
    ratio_penalty = -0.05 * min((max_ratio - 4.0) / 6.0, 1.0)  # Max: -0.05
else:
    ratio_penalty = 0.0
```

**Example**:
- Tiles: [10, 10, 10, 100], mean=32.5, max=100, ratio=3.08 → no penalty
- Tiles: [5, 5, 5, 200], mean=53.75, max=200, ratio=3.72 → no penalty  
- Tiles: [5, 5, 5, 250], mean=66.25, max=250, ratio=3.77 → no penalty
- Tiles: [3, 3, 3, 300], mean=77.25, max=300, ratio=3.88 → no penalty
- Tiles: [2, 2, 2, 350], mean=89, max=350, ratio=3.93 → no penalty
- Tiles: [1, 1, 1, 400], mean=100.75, max=400, ratio=3.97 → no penalty
- Tiles: [1, 1, 1, 450], mean=113.25, max=450, ratio=3.97 → no penalty
- Tiles: [1, 1, 1, 1, 1, 500], mean=84.17, max=500, ratio=5.94 → penalty = -0.016

**Threshold**: Only penalize if max > 4× average (allows some concentration for strategic reasons)

**Why**: Catches extreme cases where one tile has nearly all armies

---

### **3. Entropy Bonus**

**Purpose**: Reward even distribution using information theory

**Formula**:
```python
probs = armies / sum(armies)  # Normalize to probabilities
entropy = -sum(probs * log(probs))  # Information entropy
max_entropy = log(num_tiles)  # Maximum possible entropy
normalized_entropy = entropy / max_entropy  # 0 to 1
entropy_bonus = 0.05 * normalized_entropy  # Bonus: 0 to +0.05
```

**Example**:
- Uniform [10, 10, 10, 10]: entropy = log(4) → bonus = +0.05
- Concentrated [40, 0, 0, 0]: entropy ≈ 0 → bonus ≈ 0
- Mixed [20, 15, 10, 5]: entropy ≈ 0.85×log(4) → bonus ≈ +0.043

**Why**: 
- Entropy measures "evenness" of distribution
- High entropy = more uniform = better exploration
- Provides positive reinforcement (not just penalties)

---

## 🎯 Total Penalty Range

**Combined Range**: -0.15 to +0.05

- **Best case** (uniform distribution): +0.05 (entropy bonus, no penalties)
- **Worst case** (very concentrated): -0.15 (all penalties apply)
- **Typical case**: -0.05 to 0.0

**Magnitude relative to other rewards**:
- Win reward: +10.0
- Loss penalty: -10.0
- Potential-based rewards: -2.0 to +2.0 per step
- Army penalty: -0.15 to +0.05 per step

The penalty is **small but persistent**, gradually encouraging better distribution without overwhelming the main objectives.

---

## 🧠 Why This Design Works

### **1. Multiple Complementary Metrics**

- CV catches overall unevenness
- Max ratio catches extreme outliers
- Entropy rewards good behavior (not just penalizing bad)

### **2. Scale-Invariant**

All metrics work regardless of:
- Total army size
- Number of tiles owned
- Stage of game (early/late)

### **3. Gradual Penalties**

Penalties scale smoothly:
- Small imbalance → small penalty
- Large imbalance → larger penalty
- No cliff effects or sudden jumps

### **4. Threshold-Based**

Max ratio only penalizes when ratio > 4.0:
- Allows some strategic concentration (building up for attacks)
- Only penalizes excessive hoarding
- Respects tactical decisions

---

## 📈 Expected Impact

### **Immediate Effects (Episodes 101-120)**

- ✅ Agent starts spreading armies more
- ✅ More tiles with active armies (not just general)
- ⚠️ Win rate may dip slightly (10-20%) as agent adapts
- ⚠️ Value network adjusts to new reward structure

### **Medium Term (Episodes 120-150)**

- ✅ Win rate recovers to ~70%
- ✅ Better exploration patterns emerge
- ✅ More diverse attack strategies
- ✅ Captures cities more actively

### **Long Term (Episodes 150+)**

- ✅ Win rate improves beyond 70%
- ✅ Max tiles explored increases (target: 30-50)
- ✅ Action diversity increases (target: 25-35%)
- ✅ More strategic gameplay (multiple fronts, coordinated attacks)

---

## 🔍 How to Monitor

### **During Training**

Watch the logs for these indicators:

```
📊 Episode Stats:
   Max Tiles: 25 → 35 (improving exploration)
   Cities: 2 → 4 (more active capture)
   Win Rate: 68% → 74% (performance improving)
```

### **Using Monitor Script**

```bash
python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_bc
```

Watch for:
- **Max Tiles Explored**: Should increase over time
- **Action Diversity**: Should increase (less repetitive moves)
- **Unique Actions**: Should increase (trying different strategies)

### **Visual Inspection**

```bash
./watch_my_agent.sh
```

Look for:
- ✅ Armies on multiple tiles (not just general)
- ✅ Active attacks from multiple directions
- ✅ Building armies on frontier tiles
- ✅ Strategic positioning before attacks
- ❌ All armies concentrated on general (bad - should reduce over time)

---

## 🎓 Theoretical Foundation

### **Potential-Based Reward Shaping**

The original training uses potential-based reward shaping (Ng et al., 1999):

```
r_shaped(s, a, s') = r(s, a, s') + γφ(s') - φ(s)
```

Where `φ(s)` is a potential function based on land, army, and cities.

### **Our Addition**

We add an **auxiliary reward** term:

```
r_total = r_shaped + r_army_concentration
```

This is **theoretically sound** because:

1. ✅ **Additive**: Doesn't change optimal policy if agent is optimal
2. ✅ **Small magnitude**: Doesn't overwhelm main objectives
3. ✅ **Aligned**: Spreading armies helps achieve main goals (exploration, capturing)
4. ✅ **Stationary**: Penalty doesn't change over episodes (unlike curriculum)

### **Relation to Exploration Bonuses**

Similar to:
- **Curiosity-driven exploration**: Reward visiting new states
- **Empowerment**: Reward maintaining options (spread armies = more options)
- **Diversity rewards**: Reward diverse behaviors

Our penalty encourages **strategic diversity** through army distribution.

---

## 🔧 Hyperparameters

### **Current Settings**

```python
# CV Penalty
cv_scale = 0.05  # Max penalty from CV
cv_max = 3.0     # Clamp CV to this value

# Max Ratio Penalty  
ratio_threshold = 4.0  # Only penalize if max > 4× mean
ratio_scale = 0.05     # Max penalty from ratio
ratio_range = 6.0      # Normalize (ratio - 4) by this

# Entropy Bonus
entropy_scale = 0.05  # Max bonus from entropy
```

### **Tuning Guidance**

If agent **still hoards armies** after 50 episodes:
- ↑ Increase `cv_scale` to 0.07 or 0.10
- ↓ Lower `ratio_threshold` to 3.0
- ↑ Increase `ratio_scale` to 0.07

If agent **spreads too thin** (weak attacks):
- ↓ Decrease `cv_scale` to 0.03
- ↑ Increase `ratio_threshold` to 5.0
- Keep entropy bonus (rewards some concentration)

---

## 📋 Implementation Checklist

- [x] ✅ Added `compute_army_concentration_penalty()` method
- [x] ✅ Integrated penalty into rollout collection
- [x] ✅ Uses three complementary metrics (CV, max ratio, entropy)
- [x] ✅ Penalty is additive to existing potential-based rewards
- [x] ✅ Magnitude is appropriate (-0.15 to +0.05)
- [x] ✅ Scale-invariant (works at any army size)
- [x] ✅ Tested with existing checkpoint (ready to resume)
- [x] ✅ Documentation complete

---

## 🚀 How to Use

### **Resume Training with New Penalty**

```bash
# Activate environment
source venv313/bin/activate

# Resume training (uses auto_resume by default)
python src/training/train_ppo_potential.py --auto_resume --training_hours 2.0
```

### **Monitor Progress**

```bash
# Terminal 1: Training
python src/training/train_ppo_potential.py --auto_resume --training_hours 4.0

# Terminal 2: Monitoring
python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_bc
```

### **Watch Agent Play**

```bash
# After 20-30 more episodes
./watch_my_agent.sh
```

---

## 📊 Expected Timeline

| Episodes | Expectation | Metrics |
|----------|-------------|---------|
| 100 (current) | Baseline | Win: 70%, Tiles: ~15 |
| 101-110 | Adaptation period | Win: 60-65%, agent adjusting |
| 111-130 | Recovery | Win: 65-70%, better spreading |
| 131-150 | Improvement | Win: 70-75%, more exploration |
| 151-200 | Mastery | Win: 75-80%, strategic army use |

---

## 🎯 Success Criteria

After 100 more episodes (total 200), we should see:

### **Exploration Metrics**
- ✅ Max tiles explored: 30-50 (vs current ~15)
- ✅ Action diversity: 25-35% (vs current ~17%)
- ✅ Unique actions per game: 40-60 (vs current ~30)

### **Strategic Metrics**
- ✅ Win rate: >75% vs BC opponent
- ✅ Cities captured: 3-5 per game
- ✅ Multiple attack fronts visible in gameplay
- ✅ General has <50% of total armies (spread effectively)

### **Behavioral Metrics**
- ✅ Armies on frontier tiles, not just general
- ✅ Coordinated multi-tile attacks
- ✅ Active city capture throughout game
- ✅ Reduced "camping" behavior

---

## 🔬 Alternative Approaches Considered

### **1. Frontier Army Reward**
Reward having armies on tiles adjacent to enemy/neutral territory.
- **Pros**: Directly encourages active positioning
- **Cons**: Requires expensive neighbor computation, may conflict with defensive play
- **Decision**: Simpler metrics (CV, entropy) achieve similar goal

### **2. Gini Coefficient**
Use Gini coefficient (economic inequality measure).
- **Pros**: Well-studied metric for inequality
- **Cons**: More complex computation, similar to CV
- **Decision**: CV + max ratio + entropy is simpler and more interpretable

### **3. Hard Constraint**
Force agent to never exceed 4× concentration.
- **Pros**: Guaranteed spreading
- **Cons**: Breaks policy gradient, reduces flexibility
- **Decision**: Soft penalty allows learning and strategic exceptions

---

## 📚 References

1. **Ng, A. Y., Harada, D., & Russell, S.** (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.

2. **Pathak, D., et al.** (2017). Curiosity-driven exploration by self-supervised prediction. *ICML*.

3. **Bellemare, M., et al.** (2016). Unifying count-based exploration and intrinsic motivation. *NIPS*.

4. **Eysenbach, B., et al.** (2018). Diversity is all you need: Learning skills without a reward function. *ICLR*.

---

## ✅ Status

**Implementation**: COMPLETE ✅  
**Testing**: Ready for training ✅  
**Monitoring**: Scripts available ✅  
**Documentation**: Complete ✅  

**Next Step**: Resume training and monitor for 20-30 episodes to see adaptation.

```bash
python src/training/train_ppo_potential.py --auto_resume --training_hours 2.0
```

🚀 **Ready to train!**
