# 🎯 DQN Fine-tuning Implementation Guide

Based on the paper "Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning"

## 📚 Key Findings from Paper

### 1. **Environment Setup**
- **GitHub:** https://github.com/strakam/generals-bots
- **Compatibility:** Gymnasium and PettingZoo interfaces
- **Performance:** Thousands of frames per second on commodity hardware
- **Observation:** Fog-of-war, 8-cell Moore neighborhood visibility

### 2. **Training Methodology**
```
Phase 1: Behavior Cloning (3 hours)
├─ Dataset: 16,320 high-quality games (≥70 star rating)
├─ Filtering: <500 turns, newest patch only
└─ Result: Strong but narrow baseline

Phase 2: Self-Play with PPO (36 hours on H100)
├─ Algorithm: Proximal Policy Optimization
├─ Opponent Pool: N=3 most recent models
├─ Update Trigger: 45% win rate vs pool
└─ Result: Top 0.003% of human leaderboard
```

### 3. **Reward Shaping** (Critical for Success!)

The paper uses **potential-based reward shaping** to avoid sparse rewards:

```python
# Potential function (weighted combination)
ϕ(s) = 0.3 * ϕ_land(s) + 0.3 * ϕ_army(s) + 0.4 * ϕ_castle(s)

# Each sub-potential is a log-ratio
ϕ_x(s) = log(x_agent / x_enemy) / log(max_ratio)

# Shaped reward
r_shaped(s, a, s') = r_original(s, a, s') + γ*ϕ(s') - ϕ(s)
```

**Where:**
- `x_agent`: Your land/army/castle count
- `x_enemy`: Opponent's land/army/castle count
- `max_ratio`: Normalization constant to bound in [-1, 1]
- `r_original`: Simple win (+1) / lose (-1) reward

**Why it works:**
- ✅ Preserves optimal policies (proven in Ng et al., 1999)
- ✅ Provides dense feedback every step
- ✅ Encourages material advantage
- ✅ Prevents overly aggressive/defensive play

### 4. **Key Components**

#### Memory Augmentation
- Agent needs memory stack to track discovered information under fog-of-war
- Encodes what agent has already seen

#### Network Architecture
- U-Net backbone for spatial feature extraction
- Separate policy and value heads
- Output: H×W×9 tensor (pass or move all/half in 4 directions per cell)

#### Training Details
- **GAE (λ=0.95):** For long episodes with limited batch size
- **Opponent Selection:** arg max policy (deterministic), not stochastic
- **Success Metric:** 54.82% win rate vs Human.exe (prior SOTA)

---

## 🚀 Implementation Path Forward

### Option 1: Use Official Environment (Recommended)
```bash
# Install generals-bots environment
pip install generals

# The environment provides:
# - Gymnasium/PettingZoo interface
# - Fast simulation (1000s FPS)
# - Built-in opponent pool
# - Replay recording
```

### Option 2: Simplified DQN (What We Can Do Now)
Since we have:
- ✅ BC model trained
- ✅ Preprocessed data
- ✅ Understanding of reward shaping

We can implement:
1. **Load BC model as initialization**
2. **Use existing replays as opponent demonstrations**
3. **Implement potential-based reward shaping**
4. **Train with Double DQN** (simpler than PPO)

---

## 💡 Practical Implementation

### Step 1: Install Environment
```bash
pip install generals
```

### Step 2: Implement Reward Shaping
```python
def potential_function(state, agent_id):
    """Calculate potential based on material advantage"""
    agent_land = count_land(state, agent_id)
    enemy_land = count_land(state, enemy_id)
    agent_army = count_army(state, agent_id)
    enemy_army = count_army(state, enemy_id)
    agent_castle = count_castles(state, agent_id)
    enemy_castle = count_castles(state, enemy_id)
    
    # Log-ratio features (bounded to [-1, 1])
    max_ratio = 10.0  # Adjust based on typical game values
    
    phi_land = np.log(agent_land / max(enemy_land, 1)) / np.log(max_ratio)
    phi_army = np.log(agent_army / max(enemy_army, 1)) / np.log(max_ratio)
    phi_castle = np.log(agent_castle / max(enemy_castle, 1)) / np.log(max_ratio)
    
    # Weighted combination
    return 0.3 * phi_land + 0.3 * phi_army + 0.4 * phi_castle

def shaped_reward(s, a, s_next, gamma=0.99):
    """Potential-based reward shaping"""
    r_original = get_original_reward(s, a, s_next)  # +1 win, -1 lose, 0 otherwise
    phi_s = potential_function(s)
    phi_s_next = potential_function(s_next)
    
    return r_original + gamma * phi_s_next - phi_s
```

### Step 3: Self-Play Training Loop
```python
def train_dqn_with_self_play(bc_model, num_episodes=10000):
    # Load BC model as initialization
    q_network = load_bc_model(bc_model)
    target_network = copy.deepcopy(q_network)
    
    # Opponent pool (start with BC model)
    opponent_pool = [copy.deepcopy(bc_model)]
    
    for episode in range(num_episodes):
        # Select random opponent from pool
        opponent = random.choice(opponent_pool)
        
        # Play game
        env = make_generals_env()
        state = env.reset()
        
        episode_transitions = []
        done = False
        
        while not done:
            # Agent action
            action = select_action(q_network, state, epsilon)
            
            # Opponent action
            opp_action = select_action(opponent, state, epsilon=0)  # Deterministic
            
            # Environment step
            next_state, reward, done, info = env.step([action, opp_action])
            
            # Calculate shaped reward
            shaped_r = shaped_reward(state, action, next_state)
            
            # Store transition
            episode_transitions.append((state, action, shaped_r, next_state, done))
            
            state = next_state
        
        # Train on episode
        for transition in episode_transitions:
            replay_buffer.push(transition)
            
        if len(replay_buffer) > min_buffer_size:
            train_step(q_network, target_network, replay_buffer)
        
        # Update opponent pool every N episodes
        if episode % 100 == 0:
            win_rate = evaluate_vs_pool(q_network, opponent_pool)
            if win_rate > 0.45:  # Paper uses 45% threshold
                opponent_pool.append(copy.deepcopy(q_network))
                if len(opponent_pool) > 3:  # Keep pool size at 3
                    opponent_pool.pop(0)
```

---

## 📊 Expected Results

Based on paper findings:

| Method | Performance |
|--------|-------------|
| BC only | 1874 Elo |
| BC + Self-Play | ~1950 Elo |
| BC + Self-Play + Reward Shaping | ~2000 Elo |
| BC + Self-Play + Reward Shaping + Population | **2052 Elo** |

**Our current BC model:** ~1874 Elo (estimated)  
**After DQN fine-tuning:** ~2000+ Elo (estimated)  
**Improvement:** +7-10% win rate

---

## ⏱️ Time Estimates

From paper (H100 GPU):
- BC training: 3 hours
- Self-play PPO: 36 hours

Our setup (M4 Mac):
- BC training: ✅ **14 minutes** (done!)
- DQN fine-tuning: ~4-6 hours (estimated)

---

## 🎯 Next Steps

1. **Install environment:** `pip install generals`
2. **Test environment:** Run simple game loop
3. **Implement reward shaping:** Code potential function
4. **Update train_dqn.py:** Add self-play loop
5. **Run training:** 4-6 hours
6. **Evaluate:** Test against Human.exe bot

---

## 📖 References

**Paper:** "Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning"
- Authors: Matej Straka, Martin Schmid
- Code: https://github.com/strakam/generals-bots
- arXiv: 2507.06825v2

**Key Algorithms:**
- PPO: Schulman et al., 2017
- GAE: Schulman et al., 2018
- Reward Shaping: Ng et al., 1999

**Baseline Bots:**
- Human.exe: Rule-based SOTA (2018 Elo)
- Flobot: Classic baseline (1500 Elo)
- HASP: RL agent (1710 Elo)
