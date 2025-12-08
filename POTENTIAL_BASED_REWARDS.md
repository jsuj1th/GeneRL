# Potential-Based Reward Shaping (From Paper)

## Current Implementation (WRONG ❌)

We're using **additive dense rewards**:
```python
reward = base_reward + territory_bonus + army_bonus + city_bonus
```

This changes the optimal policy!

## Correct Implementation (From Paper ✓)

### Formula
```
r_shaped(s, a, s') = r_original(s, a, s') + γ·φ(s') - φ(s)
```

Where:
- `r_original` = win/loss only (+1/-1)
- `γ` = discount factor (0.99)
- `φ(s)` = potential function

### Potential Function
```
φ(s) = 0.3·φ_land(s) + 0.3·φ_army(s) + 0.4·φ_castle(s)
```

Where each sub-potential is:
```
φ_x(s) = log(x_agent / x_enemy) / log(max_ratio)
```

For x ∈ {land, army, castle (cities)}

### Why This Matters

**Potential-based shaping:**
- ✅ **Preserves optimal policy** (proven mathematically)
- ✅ Uses ratios (relative advantage) not absolutes
- ✅ Logarithmic scale handles large differences
- ✅ Normalized by max_ratio for bounded values

**Our additive rewards:**
- ❌ Changes optimal policy
- ❌ Uses absolute values (1 tile = 1 tile regardless of context)
- ❌ Linear scale (100 tiles >> 10 tiles)
- ❌ Unbounded growth

## Example

### State s:
- Agent: 10 tiles, 50 army, 2 cities
- Enemy: 20 tiles, 100 army, 3 cities

### State s' (after action):
- Agent: 12 tiles (+2), 55 army (+5), 3 cities (+1)
- Enemy: 20 tiles, 100 army, 3 cities

### Old Method (Additive):
```python
reward = +2 * 1.0 (territory) + 5 * 0.05 (army) + 1 * 10.0 (city)
       = 2.0 + 0.25 + 10.0 = 12.25
```

### Correct Method (Potential-based):
```python
# State s potential
φ_land(s) = log(10/20) / log(100) = log(0.5) / log(100) = -0.15
φ_army(s) = log(50/100) / log(100) = log(0.5) / log(100) = -0.15
φ_castle(s) = log(2/3) / log(100) = log(0.67) / log(100) = -0.09

φ(s) = 0.3*(-0.15) + 0.3*(-0.15) + 0.4*(-0.09) = -0.126

# State s' potential
φ_land(s') = log(12/20) / log(100) = log(0.6) / log(100) = -0.11
φ_army(s') = log(55/100) / log(100) = log(0.55) / log(100) = -0.13  
φ_castle(s') = log(3/3) / log(100) = log(1.0) / log(100) = 0.0

φ(s') = 0.3*(-0.11) + 0.3*(-0.13) + 0.4*(0.0) = -0.072

# Shaped reward (assuming r_original = 0 for non-terminal)
r_shaped = 0 + 0.99 * (-0.072) - (-0.126)
         = 0 + (-0.071) + 0.126
         = 0.055
```

Much smaller, more balanced reward!

## Implementation

See `train_ppo_potential.py` for correct implementation.

## References

- Paper: "Artificial Generals Intelligence: Mastering Generals.io with RL"
- Original: Ng et al., "Policy Invariance Under Reward Transformations", 1999
