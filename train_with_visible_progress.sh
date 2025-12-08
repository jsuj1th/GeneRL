#!/bin/bash

# Train PPO with visible training progress
# Open a separate terminal for monitoring

echo "=================================================="
echo "🚀 PPO TRAINING WITH MONITORING"
echo "=================================================="
echo ""
echo "This script will:"
echo "  1. Start PPO training (you'll see progress)"
echo "  2. Show you how to monitor in a separate terminal"
echo ""
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
echo "=================================================="
echo "OPTION 1: Watch training progress here (RECOMMENDED)"
echo "=================================================="
echo ""
echo "To also watch game replays during training:"
echo ""
echo "1️⃣  Open a NEW terminal window"
echo "2️⃣  Copy and paste this command:"
echo ""
echo "    cd /Users/sujithjulakanti/Desktop/DRL\\ Project && source venv313/bin/activate && python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_bc --interval 300"
echo ""
echo "3️⃣  Press Enter in that terminal"
echo ""
echo "You'll see:"
echo "  • Training progress HERE (this window)"
echo "  • Game replays THERE (other window)"
echo ""
echo "=================================================="
echo ""
echo "⏱️  Training starts in 5 seconds..."
sleep 5
echo ""
echo "🎮 TRAINING STARTED - Watch the progress below!"
echo "=================================================="
echo ""

# Run training in foreground (visible progress)
python src/training/train_ppo_potential.py \
  --bc_checkpoint checkpoints/bc/best_model.pt \
  --output_dir checkpoints/ppo_from_bc \
  --training_hours 1.0

echo ""
echo "=================================================="
echo "✅ TRAINING COMPLETE!"
echo "=================================================="
echo ""
echo "Results saved to: checkpoints/ppo_from_bc/"
echo ""
