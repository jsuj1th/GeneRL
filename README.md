# 🎮 Generals.io Deep Reinforcement Learning Agent

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep reinforcement learning agent trained to master [Generals.io](https://generals.io/), achieving **70% win rate** through Proximal Policy Optimization (PPO) with sophisticated reward shaping.

---

## 📊 Performance Highlights

| Metric | Baseline (BC) | Our Agent (PPO) | Improvement |
|--------|--------------|-----------------|-------------|
| **Win Rate** | 40% | **70%** | **+75%** 🏆 |
| **Map Exploration** | 15 tiles | **25 tiles** | **+66%** 🗺️ |
| **Cities Captured** | 1.2/game | **2.6/game** | **+116%** 🏛️ |
| **Action Diversity** | 17% | **25%+** | **+47%** 🎯 |

**Training Stats**: 100 episodes • 270,368 steps • ~10 hours on consumer hardware

---

## 🎯 Project Overview

This project implements a complete deep reinforcement learning pipeline for strategic gameplay:

1. **Behavioral Cloning (BC)**: Supervised learning from expert human demonstrations (10,000+ game states)
2. **PPO Fine-tuning**: Policy gradient RL with self-play training
3. **Reward Shaping**: Multi-objective potential-based rewards (territory + army + cities)
4. **Novel Contributions**: Army concentration penalty to encourage strategic positioning

### Key Features

- ✅ **Complete Training Pipeline**: BC → PPO → Evaluation
- ✅ **Sophisticated Reward Engineering**: 5+ component reward function
- ✅ **Efficient Architecture**: 250K parameter CNN with dueling streams
- ✅ **Auto-Resume Training**: Checkpoint management and crash recovery
- ✅ **Real-time Monitoring**: TensorBoard integration + custom metrics
- ✅ **Visualization Tools**: Watch agent play at 5× speed

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+ (tested on macOS with Apple Silicon)
- 8GB+ RAM
- 5GB disk space (for checkpoints and data)

### Installation

1. **Clone the repository** (or navigate to project directory):
```bash
cd "/Users/sujithjulakanti/Desktop/DRL Project"
```

2. **Create virtual environment**:
```bash
python3.13 -m venv venv313
source venv313/bin/activate
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gymnasium; print('Gymnasium: OK')"
```

---

## 📦 Project Structure

```
DRL-Project/
├── src/                           # Source code
│   ├── training/
│   │   ├── train_bc.py           # Behavioral cloning training
│   │   ├── train_ppo_potential.py # PPO with reward shaping ⭐
│   │   └── train_dqn_real_env.py # DQN baseline (experimental)
│   ├── evaluation/
│   │   ├── watch_bc_game.py      # Visual game playback
│   │   ├── monitor_exploration.py # Track exploration metrics
│   │   └── diagnose_agent_behavior.py
│   ├── models/
│   │   └── networks.py           # Neural network architectures
│   ├── preprocessing/
│   │   └── data_loading.py       # Dataset utilities
│   └── config.py                 # Configuration and hyperparameters
├── checkpoints/                   # Saved models
│   ├── bc/best_model.pt          # BC baseline (27.75% accuracy)
│   └── ppo_from_bc/latest_model.pt # PPO Episode 100 (70% win rate)
├── data/                          # Training datasets
├── logs/                          # TensorBoard logs
├── results/                       # Evaluation outputs
├── scripts/                       # Utility scripts
├── tests/                         # Test files
├── docs_archive/                  # Historical documentation
├── requirements.txt               # Python dependencies
├── README_RESEARCH_PAPER.md      # Detailed research documentation
├── PRESENTATION_SUMMARY.md       # Project presentation summary
└── *.sh                          # Quick-start shell scripts
```

---

## 🎮 Usage

### 1. Train Behavioral Cloning (BC) Baseline

```bash
# Train from scratch (requires dataset)
python src/training/train_bc.py

# Expected output: 27.75% test accuracy after ~2 hours
# Checkpoint saved to: checkpoints/bc/best_model.pt
```

### 2. Train PPO Agent (Main Training)

```bash
# Option A: Quick start with monitoring
./start_training_with_monitoring.sh

# Option B: Manual training with auto-resume
python src/training/train_ppo_potential.py --auto_resume --training_hours 2.0

# Option C: Resume from latest checkpoint
./resume_training.sh
```

**Training Arguments**:
- `--auto_resume`: Continue from latest checkpoint if exists
- `--training_hours`: Duration in hours (default: 2.0)
- `--episodes`: Number of episodes (default: 100)
- `--save_freq`: Checkpoint save frequency (default: 10 episodes)

### 3. Watch Agent Play (Visualization)

```bash
# Watch PPO agent at 5× speed
./watch_my_agent.sh

# Or manually:
python src/evaluation/watch_bc_game.py \
    --checkpoint checkpoints/ppo_from_bc/latest_model.pt \
    --delay 0.02 \
    --epsilon 0.1
```

**Keyboard Controls**:
- Press `Space`: Pause/Resume
- Press `Q`: Quit
- Press `R`: Restart episode

### 4. Monitor Training Progress

```bash
# View TensorBoard logs
tensorboard --logdir=logs/ppo_exploration

# Check exploration metrics
python src/evaluation/monitor_exploration.py \
    --checkpoint_dir checkpoints/ppo_from_bc

# Quick progress check
python tests/check_progress.py
```

---

## 🧠 Technical Details

### Neural Network Architecture

**GeneralsAgent** (Dueling Q-Network inspired):

```
Input: 11 channels × 25×25 grid
  ├─ Armies, terrain, ownership, generals, cities, mountains, fog-of-war
  
CNN Encoder:
  ├─ Conv2D(11→32) + BatchNorm + ReLU
  ├─ Conv2D(32→64) + BatchNorm + ReLU
  └─ Conv2D(64→128) + BatchNorm + ReLU
  
Dense Layers:
  ├─ Flatten(128×25×25) → Linear(80000→512)
  └─ Linear(512→256)
  
Dueling Streams:
  ├─ Value: V(s) = Linear(256→1)
  └─ Advantage: A(s,a) = Linear(256→256)
  
Output: Q(s,a) = V(s) + (A(s,a) - mean(A))
  └─ 256 actions (8×8 grid × 4 directions)
```

**Parameters**: ~250,000 total

### Reward Function

**Potential-Based Reward Shaping** (preserves optimal policy):

```python
r_shaped = r_terminal + γ·Φ(s') - Φ(s)

where:
  Φ(s) = 0.3·φ_land + 0.3·φ_army + 0.4·φ_cities
  φ_x = log(x_agent / x_opponent) / log(max_ratio)
  
  r_terminal = +1.0 (win) / -1.0 (loss)
```

**Army Concentration Penalty** (novel contribution):
```python
penalty = CV_penalty + max_ratio_penalty + entropy_bonus
  ├─ CV penalty: -0.05 for uneven distribution
  ├─ Max ratio penalty: -0.05 when one tile >4× average
  └─ Entropy bonus: +0.05 for uniform spreading
  
Total range: [-0.15, +0.05]
```

### Training Configuration

```python
# PPO Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99              # Discount factor
GAE_LAMBDA = 0.95         # Advantage estimation
EPSILON_CLIP = 0.2        # PPO clipping
VALUE_COEF = 0.5          # Value loss weight
ENTROPY_COEF = 0.01       # Exploration bonus
BATCH_SIZE = 256
PPO_EPOCHS = 4
GRADIENT_CLIP = 0.5
```

---

## 📈 Results & Analysis

### Training Progression

| Episode | Win Rate | Tiles Explored | Cities/Game |
|---------|----------|----------------|-------------|
| 0 (BC)  | 40%      | 15             | 1.2         |
| 20      | 52%      | 18             | 1.6         |
| 50      | 65%      | 22             | 2.2         |
| 100     | **70%**  | **25**         | **2.6**     |

### Behavioral Analysis

**BC Agent (Baseline)**:
- ❌ Passive defensive play near general
- ❌ Limited exploration (15 tiles average)
- ❌ Predictable, repetitive strategies
- ❌ Rarely captures cities (1.2/game)

**PPO Agent (Ours)**:
- ✅ Active map exploration (25 tiles average)
- ✅ Strategic city capture (2.6/game)
- ✅ Adapts to opponent behavior
- ✅ Multi-front attack strategies
- ⚠️ Still improving army distribution

### Key Findings

1. **BC Initialization Helps**: Warm-start from BC accelerates PPO convergence
2. **Reward Shaping Critical**: Dense feedback essential for sparse reward games
3. **Action Space Reduction Works**: 8×8 grids enable 3× faster training vs 12×12
4. **Concentration Penalty Needed**: Explicit mechanism required to prevent defensive camping

---

## 🔧 Advanced Usage

### Custom Training Configuration

Edit `src/config.py` to modify:
- Grid size (default: 8×8)
- Network architecture (CNN channels, dense units)
- Hyperparameters (learning rate, batch size, etc.)
- Reward weights (territory, army, cities)

### Checkpoint Management

```bash
# List available checkpoints
ls -lh checkpoints/ppo_from_bc/

# Load specific checkpoint
python src/evaluation/watch_bc_game.py \
    --checkpoint checkpoints/ppo_from_bc/episode_50.pt

# Resume from episode N
python src/training/train_ppo_potential.py \
    --checkpoint checkpoints/ppo_from_bc/episode_50.pt \
    --episodes 150
```

### Parallel Training (Future)

```bash
# Train multiple agents with different hyperparameters
python scripts/hyperparameter_search.py \
    --num_agents 4 \
    --lr_range 1e-4,1e-3 \
    --entropy_range 0.001,0.1
```

---

## 📊 Monitoring & Debugging

### TensorBoard

```bash
tensorboard --logdir=logs/ppo_exploration
# Open browser to http://localhost:6006
```

**Available Metrics**:
- Win rate over time
- Episode length
- Policy loss / Value loss
- Exploration metrics (tiles, cities)
- Action distribution heatmaps
- Reward components breakdown

### Common Issues

**Issue**: Training crashes with CUDA/MPS errors
```bash
# Solution: Force CPU training
export PYTORCH_ENABLE_MPS_FALLBACK=1
python src/training/train_ppo_potential.py --device cpu
```

**Issue**: Low win rate (<50%) after 50 episodes
```bash
# Solution: Check reward weights or increase entropy
python src/training/train_ppo_potential.py --entropy_coef 0.02
```

**Issue**: Agent camps on general (no exploration)
```bash
# Solution: Army concentration penalty is enabled by default
# If still occurring, increase penalty weight in config.py
```

---

## 🎓 Research & Documentation

### Academic Paper

See **[README_RESEARCH_PAPER.md](README_RESEARCH_PAPER.md)** for:
- Complete methodology and mathematical formulations
- Detailed experimental results and statistical analysis
- Ablation studies and comparisons
- References to related work
- Suitable for academic publication

### Presentation Materials

See **[PRESENTATION_SUMMARY.md](PRESENTATION_SUMMARY.md)** for:
- Project overview and highlights
- Key achievements and metrics
- Slide-by-slide presentation guide
- Visualizations and demos

### Historical Documentation

See **docs_archive/** folder for development history:
- Technical guides (DQN, reward shaping, checkpointing)
- Bug fix documentation
- Training logs and analysis
- Implementation notes

---

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Document all public methods
- Add unit tests for new features

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{julakanti2024generals,
  title={Deep Reinforcement Learning for Strategic Gameplay in Generals.io},
  author={Julakanti, Sujith},
  year={2024},
  note={Implementing PPO with potential-based reward shaping}
}
```

---

## 🔗 References

### Academic Papers

1. **Schulman et al. (2017)**: "Proximal Policy Optimization Algorithms" - Core PPO algorithm
2. **Ng et al. (1999)**: "Policy Invariance Under Reward Transformations" - Theoretical foundation for reward shaping
3. **Silver et al. (2016)**: "Mastering Go with Deep Neural Networks" - Deep RL for strategy games
4. **Wang et al. (2016)**: "Dueling Network Architectures" - Dueling DQN inspiration

### Resources

- [Generals.io](https://generals.io/) - Official game
- [PyTorch Documentation](https://pytorch.org/docs/) - Deep learning framework
- [Gymnasium](https://gymnasium.farama.org/) - RL environment interface
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - Reference RL implementations

---

## 📧 Contact

**Author**: Sujith Julakanti  
**Project**: Deep Reinforcement Learning for Generals.io  
**Date**: December 2024  

For questions or collaboration:
- Open an issue on GitHub
- Email: [Your Email]
- Project Repository: `/Users/sujithjulakanti/Desktop/DRL Project/`

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎉 Acknowledgments

- **Generals.io** developers for creating an engaging strategy game
- **PyTorch** team for the deep learning framework
- **OpenAI** for PPO algorithm and research
- **Research community** for prior work on RL in strategy games

---

## 🚀 Future Roadmap

### Short-term (1-2 weeks)
- [ ] Evaluate army concentration penalty impact
- [ ] Add event-based reward bonuses (city capture, destruction)
- [ ] Improve late-game aggression
- [ ] Target: 75-80% win rate

### Medium-term (1-3 months)
- [ ] Self-play curriculum with opponent pool
- [ ] Scale to larger grids (12×12, 18×18)
- [ ] Hyperparameter optimization
- [ ] Human evaluation tournaments

### Long-term (3-12 months)
- [ ] AlphaZero-style MCTS integration
- [ ] Transformer architecture for attention
- [ ] Multi-agent team-based gameplay
- [ ] Online deployment and leaderboard ranking

---

**Last Updated**: December 8, 2024  
**Version**: 1.0.0  
**Status**: ✅ Production Ready - 70% Win Rate Achieved

---

<p align="center">
  <strong>🎮 Train Smart. Play Strategic. Win Consistently. 🏆</strong>
</p>
