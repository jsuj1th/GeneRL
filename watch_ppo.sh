#!/bin/bash

# Watch PPO Agent Play Generals.io
# This script runs visual games with your trained PPO agent

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║              🎮 WATCH PPO AGENT PLAY GENERALS.IO                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will open a visual window showing your PPO agent playing Generals.io"
echo ""

# Check if checkpoint exists
CHECKPOINT="checkpoints/ppo_from_bc/latest_model.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -la checkpoints/*/latest_model.pt 2>/dev/null || echo "  (none found)"
    echo ""
    echo "Please specify a checkpoint with:"
    echo "  python src/evaluation/watch_bc_game.py --checkpoint <path>"
    exit 1
fi

echo "✅ Found checkpoint: $CHECKPOINT"
echo ""

# Get checkpoint info
python3 -c "
import torch
try:
    checkpoint = torch.load('$CHECKPOINT', map_location='cpu')
    if 'policy_state_dict' in checkpoint:
        # PPO checkpoint
        print('📊 PPO Checkpoint Info:')
        print(f\"   Episode: {checkpoint.get('episode', 'N/A')}\")
        print(f\"   Total steps: {checkpoint.get('total_steps', 0):,}\")
        print(f\"   Best win rate: {checkpoint.get('best_win_rate', 0):.1%}\")
        print(f\"   Max tiles: {checkpoint.get('max_tiles_ever', 1)}\")
    else:
        # BC checkpoint
        print('📊 BC Checkpoint Info:')
        print(f\"   Test accuracy: {checkpoint.get('test_accuracy', 0):.1%}\")
    print()
except Exception as e:
    print(f'Error loading checkpoint: {e}')
    print()
" 2>/dev/null

echo "⚙️  Options:"
echo "   • Games: 3"
echo "   • Opponent: Same agent (mirror match)"
echo "   • Delay: 0.1s per step"
echo "   • Grid: 8×8"
echo ""
echo "Press Ctrl+C to stop at any time"
echo "Close the game window to skip to next game"
echo ""
read -p "Press Enter to start watching..."

# Activate environment and run
source venv313/bin/activate

python src/evaluation/watch_bc_game.py \
    --checkpoint "$CHECKPOINT" \
    --num_games 3 \
    --opponent bc \
    --delay 0.1

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                           ✅ DONE WATCHING                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
