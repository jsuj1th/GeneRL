#!/bin/bash

# Quick start script for Generals.io RL training
# Run this on your M4 MacBook to train a competitive agent in 1.5 days

set -e  # Exit on error

echo "=========================================="
echo "Generals.io RL Training Pipeline"
echo "Dataset: 347k replays from HuggingFace"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration (edit these if needed)
MAX_REPLAYS=10000  # Use 10k for testing, None for full 347k
SHARD_SIZE=500
NUM_WORKERS=4
BC_EPOCHS=20
DQN_EPISODES=5000

echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Max replays: $MAX_REPLAYS (change to 'None' for full dataset)"
echo "  Workers: $NUM_WORKERS"
echo "  BC epochs: $BC_EPOCHS"
echo "  DQN episodes: $DQN_EPISODES"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo "Please run: python -m venv venv"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
pip install -q -r requirements.txt

echo ""
echo "=========================================="
echo "STEP 0: Inspect Dataset (5 minutes)"
echo "=========================================="
read -p "Inspect HuggingFace dataset structure? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python inspect_dataset.py
    echo ""
    read -p "Does the dataset format look correct? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Please update src/preprocessing/download_replays.py based on the output${NC}"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "STEP 1: Download & Preprocess (4-6 hours)"
echo "=========================================="
read -p "Download and preprocess replays? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p data/processed
    
    if [ "$MAX_REPLAYS" = "None" ]; then
        echo -e "${YELLOW}Downloading ALL 347k replays (this will take a while)...${NC}"
        python src/preprocessing/download_replays.py \
            --output_dir data/processed \
            --max_replays None \
            --shard_size $SHARD_SIZE \
            --num_workers $NUM_WORKERS \
            --min_turns 50 \
            --max_turns 1000
    else
        echo -e "${YELLOW}Downloading $MAX_REPLAYS replays for testing...${NC}"
        python src/preprocessing/download_replays.py \
            --output_dir data/processed \
            --max_replays $MAX_REPLAYS \
            --shard_size $SHARD_SIZE \
            --num_workers $NUM_WORKERS \
            --min_turns 50 \
            --max_turns 1000
    fi
    
    echo -e "${GREEN}✓ Preprocessing complete!${NC}"
fi

echo ""
echo "=========================================="
echo "STEP 2: Behavior Cloning (12-18 hours)"
echo "=========================================="
read -p "Train BC model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p checkpoints/bc logs
    
    echo -e "${YELLOW}Starting BC training (this will take 12-18 hours)...${NC}"
    echo "Monitor with: tensorboard --logdir logs/"
    
    python src/train_bc.py \
        --data_dir data/processed \
        --output_dir checkpoints/bc \
        --batch_size 1024 \
        --num_epochs $BC_EPOCHS \
        --learning_rate 0.0003 \
        --num_workers $NUM_WORKERS
    
    echo -e "${GREEN}✓ BC training complete!${NC}"
fi

echo ""
echo "=========================================="
echo "STEP 3: DQN Fine-tuning (4-6 hours)"
echo "=========================================="
read -p "Fine-tune with DQN? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -f "checkpoints/bc/best_model.pt" ]; then
        echo -e "${RED}Error: BC checkpoint not found!${NC}"
        echo "Please complete Step 2 first."
        exit 1
    fi
    
    mkdir -p checkpoints/dqn
    
    echo -e "${YELLOW}Starting DQN fine-tuning (this will take 4-6 hours)...${NC}"
    
    python src/train_dqn.py \
        --bc_checkpoint checkpoints/bc/best_model.pt \
        --output_dir checkpoints/dqn \
        --num_episodes $DQN_EPISODES \
        --batch_size 512 \
        --replay_buffer_size 100000 \
        --human_data_ratio 0.1
    
    echo -e "${GREEN}✓ DQN fine-tuning complete!${NC}"
fi

echo ""
echo "=========================================="
echo "STEP 4: Evaluation"
echo "=========================================="
read -p "Evaluate final model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -f "checkpoints/dqn/best_model.pt" ]; then
        echo -e "${YELLOW}Warning: DQN checkpoint not found, using BC model${NC}"
        CHECKPOINT="checkpoints/bc/best_model.pt"
    else
        CHECKPOINT="checkpoints/dqn/best_model.pt"
    fi
    
    echo -e "${YELLOW}Evaluating model: $CHECKPOINT${NC}"
    
    python src/evaluate.py \
        --checkpoint $CHECKPOINT \
        --num_games 100
    
    echo -e "${GREEN}✓ Evaluation complete!${NC}"
fi

echo ""
echo "=========================================="
echo "🎉 TRAINING PIPELINE COMPLETE! 🎉"
echo "=========================================="
echo ""
echo "Your models are saved in:"
echo "  - checkpoints/bc/best_model.pt"
echo "  - checkpoints/dqn/best_model.pt"
echo ""
echo "View training logs with:"
echo "  tensorboard --logdir logs/"
echo ""
echo "Next steps:"
echo "  - Analyze results in logs/"
echo "  - Try different hyperparameters"
echo "  - Test against human players"
echo ""
