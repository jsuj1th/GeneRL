#!/bin/bash

# Quick Start: Train and Monitor PPO with BC Warm Start
# This will start training AND open a monitoring window to watch progress

echo "=================================================="
echo "🚀 QUICK START: PPO WITH LIVE MONITORING"
echo "=================================================="
echo ""
echo "This will:"
echo "  1. Start PPO training (BC warm start)"
echo "  2. Monitor and show games every 5 minutes"
echo "  3. Display exploration statistics"
echo ""
echo "You'll be able to watch the agent learn to explore!"
echo "=================================================="
echo ""

# Activate virtual environment
cd "/Users/sujithjulakanti/Desktop/DRL Project"
source venv313/bin/activate

# Create output directory
mkdir -p checkpoints/ppo_from_bc
mkdir -p logs

echo "📊 Starting training at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Start training in foreground (you'll see the progress!)
echo "1️⃣  Starting PPO training (BC warm start)..."
echo "   You'll see training progress in real-time"
echo "   Training will run for 1 hour"
echo ""
echo "=================================================="
echo ""

python src/training/train_ppo_potential.py \
  --bc_checkpoint checkpoints/bc/best_model.pt \
  --output_dir checkpoints/ppo_from_bc \
  --training_hours 1.0

echo ""
echo "=================================================="
echo "⚠️  Monitoring stopped"
echo "=================================================="
echo ""
echo "Training is still running in background (PID: $TRAIN_PID)"
echo ""
echo "Options:"
echo "  • Resume monitoring: python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_bc"
echo "  • View training logs: tail -f logs/ppo_training.log"
echo "  • Stop training: kill $TRAIN_PID"
echo ""
echo "Training will complete in ~1 hour from start time."
echo "=================================================="
