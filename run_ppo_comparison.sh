#!/bin/bash

# Train PPO with potential-based rewards - Two variants in parallel
# 1. Warm start from BC (test if PPO can overcome BC's poor exploration)
# 2. From scratch (baseline without BC bias)

echo "=================================================="
echo "🚀 PPO TRAINING COMPARISON EXPERIMENT"
echo "=================================================="
echo ""
echo "This will run TWO training sessions in parallel:"
echo ""
echo "1️⃣  PPO with BC Warm Start"
echo "   - Starts with BC weights (27.75% accuracy)"
echo "   - Tests if PPO can overcome BC's lack of exploration"
echo "   - Output: checkpoints/ppo_from_bc/"
echo ""
echo "2️⃣  PPO from Scratch  "
echo "   - Random initialization (no BC bias)"
echo "   - Pure PPO learning with exploration"
echo "   - Output: checkpoints/ppo_from_scratch/"
echo ""
echo "Both will train for 1 hour with:"
echo "  ✓ Potential-based reward shaping (from paper)"
echo "  ✓ High entropy bonus (0.1 for exploration)"
echo "  ✓ Curriculum learning (80% random → 20% random)"
echo "  ✓ Small grids (12x12 to 18x18 for faster episodes)"
echo ""
echo "=================================================="
echo ""

# Activate virtual environment
source venv313/bin/activate

# Create output directories
mkdir -p checkpoints/ppo_from_bc
mkdir -p checkpoints/ppo_from_scratch

echo "📊 Starting training at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Run both in background
echo "1️⃣  Starting PPO with BC warm start..."
python src/training/train_ppo_potential.py \
  --bc_checkpoint checkpoints/bc/best_model.pt \
  --output_dir checkpoints/ppo_from_bc \
  --training_hours 1.0 \
  > logs/ppo_from_bc.log 2>&1 &

PID_BC=$!
echo "   PID: $PID_BC (logs: logs/ppo_from_bc.log)"
echo ""

echo "2️⃣  Starting PPO from scratch..."
python src/training/train_ppo_potential.py \
  --bc_checkpoint checkpoints/bc/best_model.pt \
  --output_dir checkpoints/ppo_from_scratch \
  --training_hours 1.0 \
  --from_scratch \
  > logs/ppo_from_scratch.log 2>&1 &

PID_SCRATCH=$!
echo "   PID: $PID_SCRATCH (logs: logs/ppo_from_scratch.log)"
echo ""

echo "=================================================="
echo "✅ Both training sessions started!"
echo "=================================================="
echo ""
echo "📊 Monitor progress:"
echo "   • BC warm start:  tail -f logs/ppo_from_bc.log"
echo "   • From scratch:   tail -f logs/ppo_from_scratch.log"
echo ""
echo "⏱️  Both will complete in ~1 hour"
echo ""
echo "🔍 Compare results after training:"
echo "   • BC warm start:  checkpoints/ppo_from_bc/training_summary.json"
echo "   • From scratch:   checkpoints/ppo_from_scratch/training_summary.json"
echo ""
echo "=================================================="

# Optional: Start monitoring in separate terminals
echo "🔍 OPTIONAL: Monitor training progress in real-time"
echo ""
echo "Open 2 new terminal windows and run:"
echo ""
echo "Terminal 1 (Monitor BC warm start):"
echo "  cd /Users/sujithjulakanti/Desktop/DRL\ Project"
echo "  source venv313/bin/activate"
echo "  python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_bc --interval 300"
echo ""
echo "Terminal 2 (Monitor from scratch):"
echo "  cd /Users/sujithjulakanti/Desktop/DRL\ Project"
echo "  source venv313/bin/activate"
echo "  python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_scratch --interval 300"
echo ""
echo "=================================================="

# Wait for both to complete
wait $PID_BC $PID_SCRATCH

echo ""
echo "=================================================="
echo "✅ TRAINING COMPLETE!"
echo "=================================================="
echo ""

# Compare results
echo "📊 RESULTS COMPARISON:"
echo ""

if [ -f checkpoints/ppo_from_bc/training_summary.json ]; then
    echo "1️⃣  PPO with BC Warm Start:"
    cat checkpoints/ppo_from_bc/training_summary.json | grep -E "(best_win_rate|final_win_rate|total_episodes|max_tiles_ever)" | sed 's/^/   /'
else
    echo "1️⃣  PPO with BC Warm Start: ❌ Failed (check logs/ppo_from_bc.log)"
fi

echo ""

if [ -f checkpoints/ppo_from_scratch/training_summary.json ]; then
    echo "2️⃣  PPO from Scratch:"
    cat checkpoints/ppo_from_scratch/training_summary.json | grep -E "(best_win_rate|final_win_rate|total_episodes|max_tiles_ever)" | sed 's/^/   /'
else
    echo "2️⃣  PPO from Scratch: ❌ Failed (check logs/ppo_from_scratch.log)"
fi

echo ""
echo "=================================================="
echo ""
echo "📈 Interpretation Guide:"
echo ""
echo "If BC Warm Start > From Scratch:"
echo "  → BC initialization helps! BC's knowledge + PPO's exploration = WIN"
echo ""
echo "If From Scratch > BC Warm Start:"
echo "  → BC was a liability. Its poor exploration habits held PPO back."
echo ""
echo "If Both ~similar:"
echo "  → BC initialization doesn't matter. PPO can learn from either."
echo ""
echo "=================================================="
