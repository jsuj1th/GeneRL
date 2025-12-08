#!/bin/bash
# Compare exploitation vs exploration behavior

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║           🔍 COMPARE EXPLOITATION vs EXPLORATION 🔍                       ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This script will show you the difference between:"
echo "  1. Pure exploitation (epsilon=0.0) - Always greedy"
echo "  2. With exploration (epsilon=0.1) - 10% random"
echo ""
echo "You'll see how exploration makes the agent more interesting to watch!"
echo ""

read -p "Press Enter to start..."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 PART 1: PURE EXPLOITATION (epsilon=0.0)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "The agent will ALWAYS pick the best action (greedy)"
echo "Watch how it plays - note the strategy"
echo ""
read -p "Press Enter to watch 2 games with epsilon=0.0..."

python src/evaluation/watch_bc_game.py \
    --epsilon 0.0 \
    --num_games 2 \
    --delay 0.05 \
    --opponent random

echo ""
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎲 PART 2: WITH EXPLORATION (epsilon=0.1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "The agent will pick the best action 90% of the time"
echo "But 10% of the time, it will pick a RANDOM action"
echo ""
echo "Watch for:"
echo "  • More variety in gameplay"
echo "  • Occasional unexpected moves"
echo "  • How it recovers from random actions"
echo ""
read -p "Press Enter to watch 2 games with epsilon=0.1..."

python src/evaluation/watch_bc_game.py \
    --epsilon 0.1 \
    --num_games 2 \
    --delay 0.05 \
    --opponent random

echo ""
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                          📊 COMPARISON DONE                               ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Questions to think about:"
echo "  1. Did you notice more variety with epsilon=0.1?"
echo "  2. Did exploration hurt the win rate significantly?"
echo "  3. Which was more interesting to watch?"
echo ""
echo "💡 TIP: For evaluation, use epsilon=0.0 (true performance)"
echo "💡 TIP: For watching/demos, use epsilon=0.1 (more interesting)"
echo ""
