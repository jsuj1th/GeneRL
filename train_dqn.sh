#!/bin/bash
# DQN Self-Play Training Launcher
# Usage: ./train_dqn.sh [hours]

set -e

TRAINING_HOURS=${1:-4}

echo "======================================================================"
echo "🎮 DQN Self-Play Training"
echo "======================================================================"
echo "Training duration: ${TRAINING_HOURS} hours"
echo "BC checkpoint: checkpoints/bc/best_model.pt"
echo "Output directory: checkpoints/dqn"
echo ""
echo "Press Ctrl+C to stop training at any time"
echo "======================================================================"
echo ""

# Run training
python3 src/training/train_dqn_self_play.py \
  --bc_checkpoint checkpoints/bc/best_model.pt \
  --train_dir data/processed/train \
  --val_dir data/processed/val \
  --output_dir checkpoints/dqn \
  --training_hours ${TRAINING_HOURS}

echo ""
echo "======================================================================"
echo "✅ Training complete!"
echo "======================================================================"
echo ""
echo "📊 View results:"
echo "  - Best model: checkpoints/dqn/best_model.pt"
echo "  - Training summary: checkpoints/dqn/training_summary.json"
echo ""
echo "📈 View training curves:"
echo "  tensorboard --logdir logs/dqn_self_play"
echo ""
echo "🔍 Evaluate model:"
echo "  python3 src/evaluation/evaluate.py \\"
echo "    --model_path checkpoints/dqn/best_model.pt \\"
echo "    --test_dir data/processed/test \\"
echo "    --output_file results/dqn_evaluation.json"
echo ""
echo "======================================================================"
