"""
Configuration for Generals.io Deep RL training.
Optimized for MacBook M4 Air with 50k filtered replays.

Full dataset has 347k replays, but we use 50k for efficient training with limited compute.
"""

class Config:
    # ============================================================================
    # Data Configuration
    # ============================================================================
    TOTAL_REPLAYS = 50_000   # Filtered subset from 347k
    TRAIN_REPLAYS = 40_000   # 80% for training
    VAL_REPLAYS = 5_000      # 10% for validation
    TEST_REPLAYS = 5_000     # 10% for testing
    
    # Replay filtering (if metadata available)
    MIN_GAME_LENGTH = 50      # Filter out very short games
    MAX_GAME_LENGTH = 1000    # Filter out extremely long games
    
    # ============================================================================
    # Game State Configuration
    # ============================================================================
    MAP_HEIGHT = 30
    MAP_WIDTH = 30
    
    # State channels
    NUM_CHANNELS = 7  # [terrain, my_tiles, enemy_tiles, my_armies, 
                      #  enemy_armies, neutral_armies, fog]
    
    # Action space: 4 directions * (map_height * map_width) source tiles
    NUM_ACTIONS = 4 * MAP_HEIGHT * MAP_WIDTH  # 3600 actions
    
    # ============================================================================
    # Model Architecture
    # ============================================================================
    # CNN backbone
    CNN_CHANNELS = [32, 64, 64, 128]
    KERNEL_SIZE = 3
    
    # Dueling DQN heads
    VALUE_HIDDEN = 256
    ADVANTAGE_HIDDEN = 512
    
    # ============================================================================
    # Behavior Cloning Configuration
    # ============================================================================
    BC_BATCH_SIZE = 1024  # Large batches for M4 efficiency
    BC_LEARNING_RATE = 1e-3
    BC_WEIGHT_DECAY = 1e-4
    BC_EPOCHS = 100
    BC_PATIENCE = 10  # Early stopping
    
    # Data augmentation
    BC_USE_AUGMENTATION = True
    BC_FLIP_HORIZONTAL = True
    BC_FLIP_VERTICAL = True
    
    # ============================================================================
    # DQN Configuration
    # ============================================================================
    DQN_BATCH_SIZE = 512
    DQN_LEARNING_RATE = 1e-4
    DQN_GAMMA = 0.99
    DQN_TAU = 0.005  # Soft update for target network
    
    # Replay buffer
    DQN_BUFFER_SIZE = 100_000
    DQN_HUMAN_REPLAY_RATIO = 0.2  # 20% human transitions for regularization
    DQN_MIN_BUFFER_SIZE = 10_000  # Start training after this many transitions
    
    # Exploration
    EPSILON_START = 0.1  # Start low since we have BC initialization
    EPSILON_END = 0.01
    EPSILON_DECAY_STEPS = 50_000
    
    # Training
    DQN_UPDATE_FREQUENCY = 4  # Update every 4 steps
    DQN_TARGET_UPDATE_FREQUENCY = 1000  # Update target network
    
    # ============================================================================
    # Reward Shaping
    # ============================================================================
    REWARD_TERRITORY = 0.1      # Per tile controlled
    REWARD_ARMY_GROWTH = 0.05   # Per army unit gained
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0
    REWARD_SURVIVAL = 0.5       # Per turn survived
    REWARD_GENERAL_KILL = 50.0
    REWARD_CITY_CAPTURE = 10.0
    
    # ============================================================================
    # Training Configuration
    # ============================================================================
    DEVICE = "mps"  # Use Metal Performance Shaders on M4
    NUM_WORKERS = 8  # For data loading
    PIN_MEMORY = False  # Not needed for MPS
    
    # Checkpointing
    SAVE_FREQUENCY = 1000  # Save every N updates
    EVAL_FREQUENCY = 5000  # Evaluate every N updates
    
    # Logging
    LOG_DIR = "logs"
    TENSORBOARD_DIR = "runs"
    
    # ============================================================================
    # Evaluation Configuration
    # ============================================================================
    EVAL_GAMES = 100
    EVAL_OPPONENTS = ["random", "expander", "bc_baseline"]
    
    # ============================================================================
    # Optimization for M4
    # ============================================================================
    # Use mixed precision training if supported
    USE_AMP = True
    
    # Gradient clipping
    GRAD_CLIP = 1.0
    
    # Preprocessing
    PREPROCESS_BATCH_SIZE = 1000  # Games to process at once
    COMPRESSION = "gzip"  # For .npz files
