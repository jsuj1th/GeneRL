#!/bin/bash

# Quickstart script for Generals.io DRL Project
# This runs the full pipeline end-to-end

set -e  # Exit on error

echo "======================================"
echo "🎮 Generals.io DRL Training Pipeline"
echo "======================================"
echo ""

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import torch" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Step 1: Download replays
echo ""
echo "======================================"
echo "Step 1/5: Download Replays (2-3 hours)"
echo "======================================"
python src/preprocessing/download_replays.py \
    --output_dir data/raw \
    --num_replays 50000 \
    --seed 42

# Step 2: Preprocess replays
echo ""
echo "======================================"
echo "Step 2/5: Preprocess Replays (2-3 hours)"
echo "======================================"
python src/preprocessing/preprocess_replays.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --num_workers 8

# Step 3: Behavior cloning
echo ""
echo "======================================"
echo "Step 3/5: Behavior Cloning (18 hours)"
echo "======================================"
python src/training/train_bc.py \
    --data_dir data/processed/train \
    --val_dir data/processed/val \
    --output_dir checkpoints/bc \
    --batch_size 1024 \
    --epochs 100

# Step 4: DQN fine-tuning
echo ""
echo "======================================"
echo "Step 4/5: DQN Fine-tuning (8 hours)"
echo "======================================"
python src/training/train_dqn.py \
    --bc_checkpoint checkpoints/bc/best_model.pt \
    --output_dir checkpoints/dqn \
    --training_hours 8

# Step 5: Evaluation
echo ""
echo "======================================"
echo "Step 5/5: Evaluation"
echo "======================================"
python src/evaluation/evaluate.py \
    --model_path checkpoints/dqn/final_model.pt \
    --num_games 100

echo ""
echo "======================================"
echo "✅ Pipeline Complete!"
echo "======================================"
echo "Results saved to: results/"
echo "View logs: tensorboard --logdir logs/"
