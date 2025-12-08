#!/bin/bash

# Quick Start: PPO Training with Live Monitoring
# This script starts PPO training and monitoring in an integrated way

echo "======================================================================"
echo "🚀 PPO TRAINING WITH LIVE MONITORING - QUICK START"
echo "======================================================================"
echo ""
echo "This will:"
echo "  1. Start PPO training with potential-based reward shaping"
echo "  2. Automatically monitor progress every 5 minutes"
echo "  3. Show visual games with exploration statistics"
echo ""
echo "✅ Bug Fix Applied: Action attribute error is fixed!"
echo ""
echo "You need 2 terminal windows for this:"
echo "  Terminal 1: Training (runs continuously)"
echo "  Terminal 2: Monitoring (checks every 5 min)"
echo ""
echo "======================================================================"
echo ""

# Check which terminal we're in
if [ -z "$MONITOR_TERMINAL" ]; then
    # This is terminal 1 - training
    echo "📝 TERMINAL 1 SETUP (Training)"
    echo "======================================================================"
    echo ""
    echo "Starting PPO training with potential-based reward shaping..."
    echo "The agent will learn from BC initialization and improve exploration."
    echo ""
    echo "⚙️  Training Configuration:"
    echo "   - Algorithm: PPO with potential-based rewards"
    echo "   - Initialization: BC warm start (27.75% accuracy)"
    echo "   - Grid size: 12x12 to 18x18"
    echo "   - Max steps per episode: 5,000"
    echo "   - Entropy bonus: 0.1 (high exploration)"
    echo "   - Checkpoint: Every 10 episodes"
    echo ""
    echo "🎯 Goal: Improve exploration from BC's 15 tiles → 30-50 tiles"
    echo ""
    echo "======================================================================"
    echo ""
    echo "To start monitoring in another terminal, run:"
    echo "  MONITOR_TERMINAL=1 ./quick_start_with_monitoring.sh"
    echo ""
    echo "======================================================================"
    echo ""
    
    read -p "Press Enter to start training..."
    
    # Activate virtual environment
    source venv313/bin/activate
    
    # Start training
    python src/training/train_ppo_potential.py
    
else
    # This is terminal 2 - monitoring
    echo "👀 TERMINAL 2 SETUP (Monitoring)"
    echo "======================================================================"
    echo ""
    echo "This will monitor training progress every 5 minutes by:"
    echo "  1. Loading the latest checkpoint"
    echo "  2. Playing 3 visual games vs BC opponent"
    echo "  3. Showing exploration statistics"
    echo "  4. Comparing with BC baseline (15 tiles, 17% diversity)"
    echo ""
    echo "✅ The Action attribute bug has been fixed!"
    echo ""
    echo "======================================================================"
    echo ""
    echo "Make sure training is running in Terminal 1 first!"
    echo ""
    
    read -p "Press Enter to start monitoring..."
    
    # Activate virtual environment
    source venv313/bin/activate
    
    # Start monitoring
    python src/evaluation/monitor_exploration.py --checkpoint_dir checkpoints/ppo_from_bc
fi
