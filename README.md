# 🎮 Generals.io Deep RL Project

## Overview
Training a Generals.io agent using **50,000 filtered replays** from Hugging Face with Behavior Cloning + DQN fine-tuning on MacBook M4 Air.

**Dataset**: https://huggingface.co/datasets/strakammm/generals_io_replays  
**Strategy**: Use filtered subset (50k out of 347k) for efficient training with limited compute

## ⏱️ Timeline (1.5 days)
- **Hours 0-4**: Download & preprocess 50k replays (40k train / 5k val / 5k test)
- **Hours 4-24**: Intensive behavior cloning training  
- **Hours 24-36**: Light DQN fine-tuning + evaluation

## 📁 Project Structure
```
├── data/
│   ├── raw/              # Downloaded replays (train/val/test splits)
│   └── processed/        # Preprocessed .npz shards
├── src/
│   ├── config.py         # Hyperparameters and settings
│   ├── preprocessing/    # Download & preprocess replays
│   ├── models/           # Neural network architectures
│   ├── training/         # BC and DQN training loops
│   ├── agents/           # Agent implementations
│   └── evaluation/       # Evaluation utilities
├── checkpoints/          # Saved models
├── logs/                 # TensorBoard logs
└── results/              # Evaluation metrics
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Replays (2-3 hours)
```bash
python src/preprocessing/download_replays.py \
    --output_dir data/raw \
    --num_replays 50000 \
    --seed 42
```

This downloads 50k replays and splits them:
- `data/raw/train/` - 40,000 replays (80%)
- `data/raw/val/` - 5,000 replays (10%)
- `data/raw/test/` - 5,000 replays (10%)

### 3. Preprocess Replays (2-3 hours)
```bash
python src/preprocessing/preprocess_replays.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --num_workers 8
```

Converts replays to training-ready tensors (`.npz` shards)

### 4. Behavior Cloning Training (18 hours)
```bash
python src/training/train_bc.py \
    --data_dir data/processed/train \
    --val_dir data/processed/val \
    --output_dir checkpoints/bc \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.0003
```

Monitor training:
```bash
tensorboard --logdir logs/bc
```

### 5. DQN Fine-tuning (8 hours)
```bash
python src/training/train_dqn.py \
    --bc_checkpoint checkpoints/bc/best_model.pt \
    --output_dir checkpoints/dqn \
    --training_hours 8 \
    --human_data_ratio 0.1
```

### 6. Evaluation
```bash
python src/evaluation/evaluate.py \
    --model_path checkpoints/dqn/final_model.pt \
    --num_games 100
```

## 📊 Expected Results

After 1.5 days on M4 MacBook Air:

| Metric | BC Model | DQN Fine-tuned |
|--------|----------|----------------|
| Action Accuracy | 60-70% | 65-75% |
| Win Rate vs Random | 70-80% | 80-90% |
| Avg Territory | 25-35% | 30-40% |
| Avg Survival Time | 150-200 turns | 180-220 turns |

## ⚙️ Configuration

Key hyperparameters in `src/config.py`:

```python
# Data
MAP_HEIGHT = 20
MAP_WIDTH = 20
NUM_ACTIONS = 1600  # 4 directions × 400 tiles

# Training
BC_BATCH_SIZE = 1024
BC_LEARNING_RATE = 0.0003
BC_EPOCHS = 100

DQN_BATCH_SIZE = 512
DQN_LEARNING_RATE = 0.0001
REPLAY_BUFFER_SIZE = 100000
```

## 🔑 Key Features

- ✅ **Efficient data pipeline**: Streaming from disk, no full dataset in RAM
- ✅ **Action masking**: Ensures only valid moves are selected
- ✅ **Dense reward shaping**: Territory control + survival incentives
- ✅ **Human replay regularization**: Mix human transitions in DQN buffer
- ✅ **M4 optimized**: Large batches, efficient CNN architecture
- ✅ **Proper splits**: 80/10/10 train/val/test with reproducible seeds

## 🎯 Optimization for M4 Air

1. **Limited dataset**: 50k replays instead of full 347k
2. **Streaming data**: Load batches from disk, not all into RAM
3. **CPU-friendly**: Small CNN, no complex attention mechanisms
4. **Short BC training**: Focus on imitation, not RL exploration
5. **Minimal DQN**: Few hours of fine-tuning, not full convergence

## 📝 Notes

- The preprocessing step processes each split separately to maintain data isolation
- Validation set used for early stopping during BC training
- Test set held out completely until final evaluation
- All random operations use fixed seeds for reproducibility

## 🐛 Troubleshooting

**Out of memory during training?**
- Reduce `batch_size` in config
- Reduce `num_workers` for data loading
- Use fewer replays (`--num_replays 20000`)

**Dataset download too slow?**
- Check internet connection
- Try smaller subset first (`--num_replays 10000`)
- Use HuggingFace cache if available

**Training not improving?**
- Check TensorBoard logs for NaN losses
- Verify action masking is working
- Ensure preprocessed data format matches model input

## 📚 References

- [Artificial Generals Intelligence Paper](2507.06825v2.pdf)
- [Generals.io Environment](https://generals.io)
- [HuggingFace Dataset](https://huggingface.co/datasets/strakammm/generals_io_replays)
