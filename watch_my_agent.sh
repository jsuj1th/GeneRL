#!/bin/bash

# Watch Your Trained PPO Agent Play Generals.io
# Simple script to visualize how your agent plays

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║              🎮 WATCH YOUR TRAINED PPO AGENT PLAY                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if checkpoint exists
CHECKPOINT="checkpoints/ppo_from_bc/latest_model.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -lh checkpoints/*/latest_model.pt 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "✅ Found PPO checkpoint: $CHECKPOINT"
echo ""

# Get checkpoint info
python3 -c "
import torch
try:
    checkpoint = torch.load('$CHECKPOINT', map_location='cpu')
    if 'policy_state_dict' in checkpoint:
        print('📊 PPO Agent Info:')
        print(f\"   Episode: {checkpoint.get('episode', 'N/A')}\")
        print(f\"   Total steps: {checkpoint.get('total_steps', 0):,}\")
        print(f\"   Best win rate: {checkpoint.get('best_win_rate', 0):.1%}\")
    print()
except Exception as e:
    print(f'Note: Could not load checkpoint details')
    print()
" 2>/dev/null

echo "⚙️  Settings:"
echo "   • Games: 3"
echo "   • Grid Size: 8×8"
echo "   • Opponent: Same agent (mirror match)"
echo "   • Exploration: 10% (epsilon=0.1)"
echo "   • Delay: 0.02s per step (5x speed!)"
echo ""
echo "🎮 Visual game window will open showing:"
echo "   🟦 Blue = Your agent's territory"
echo "   🟥 Red = Opponent's territory"
echo "   ⚪ Gray = Neutral tiles"
echo "   🏔️  Mountains (obstacles)"
echo "   🏛️  Cities (bonus army)"
echo "   👑 Generals (must protect!)"
echo ""
echo "💡 Tips:"
echo "   • Close window to skip to next game"
echo "   • Press Ctrl+C to stop all games"
echo ""
read -p "Press Enter to start watching..."
echo ""

# Activate virtual environment
source venv313/bin/activate

# Run the watch script
python src/evaluation/watch_bc_game.py \
    --checkpoint "$CHECKPOINT" \
    --num_games 3 \
    --opponent bc \
    --epsilon 0.1 \
    --delay 0.02

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                           ✅ DONE WATCHING                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
