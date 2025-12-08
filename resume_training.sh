#!/bin/bash

# Resume PPO Training from Previous Checkpoint
# This script continues training from where it left off

echo "======================================================================"
echo "🔄 RESUME PPO TRAINING FROM CHECKPOINT"
echo "======================================================================"
echo ""
echo "This will continue training from your previous checkpoint."
echo ""

# Check if checkpoint exists
CHECKPOINT_DIR="checkpoints/ppo_from_bc"
CHECKPOINT_FILE="$CHECKPOINT_DIR/latest_model.pt"

if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "❌ No checkpoint found at: $CHECKPOINT_FILE"
    echo ""
    echo "Available checkpoint directories:"
    ls -d checkpoints/*/ 2>/dev/null || echo "  (none)"
    echo ""
    echo "To resume training, you need a checkpoint file."
    echo "Use one of these options:"
    echo ""
    echo "1. Resume from specific checkpoint:"
    echo "   python src/training/train_ppo_potential.py --resume checkpoints/ppo_potential/latest_model.pt --training_hours 2.0"
    echo ""
    echo "2. Auto-resume from output directory:"
    echo "   python src/training/train_ppo_potential.py --auto_resume --output_dir checkpoints/ppo_potential --training_hours 2.0"
    echo ""
    exit 1
fi

echo "✅ Found checkpoint: $CHECKPOINT_FILE"
echo ""

# Get checkpoint info
python3 -c "
import torch
checkpoint = torch.load('$CHECKPOINT_FILE', map_location='cpu')
print(f\"📊 Checkpoint Info:\")
print(f\"   Episode: {checkpoint['episode']}\")
print(f\"   Total steps: {checkpoint['total_steps']:,}\")
print(f\"   Best win rate: {checkpoint.get('best_win_rate', 0.0):.1%}\")
print(f\"   Max tiles ever: {checkpoint.get('max_tiles_ever', 1)}\")
print()
"

# Ask for training duration
echo "How many hours do you want to continue training?"
read -p "Hours (default: 1.0): " HOURS
HOURS=${HOURS:-1.0}

echo ""
echo "======================================================================"
echo "Starting training..."
echo "  Resuming from: $CHECKPOINT_FILE"
echo "  Training for: $HOURS hours"
echo "  Output dir: $CHECKPOINT_DIR"
echo "======================================================================"
echo ""

# Activate virtual environment
source venv313/bin/activate

# Resume training
python src/training/train_ppo_potential.py \
    --resume "$CHECKPOINT_FILE" \
    --output_dir "$CHECKPOINT_DIR" \
    --training_hours "$HOURS"

echo ""
echo "======================================================================"
echo "✅ Training session complete!"
echo "======================================================================"
