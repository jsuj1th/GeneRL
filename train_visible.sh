#!/bin/bash

# Simple: Just show training progress in real-time
# No monitoring, just pure training visibility

echo "=================================================="
echo "🚀 PPO TRAINING - VISIBLE PROGRESS"
echo "=================================================="
echo ""
echo "Training will run for 1 hour"
echo "You'll see real-time progress updates"
echo ""
echo "Press Ctrl+C to stop early"
echo "=================================================="
echo ""

# Activate virtual environment
cd "/Users/sujithjulakanti/Desktop/DRL Project"
source venv313/bin/activate

# Create output directory
mkdir -p checkpoints/ppo_from_bc

echo "📊 Starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Run training with visible output
python src/training/train_ppo_potential.py \
  --bc_checkpoint checkpoints/bc/best_model.pt \
  --output_dir checkpoints/ppo_from_bc \
  --training_hours 1.0

echo ""
echo "✅ Training complete!"
