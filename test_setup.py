#!/usr/bin/env python3
"""
Quick test to verify project setup.
Run this to check if all dependencies are installed correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import torch
        print(f"  ✓ torch (version {torch.__version__})")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    try:
        from datasets import load_dataset
        print("  ✓ datasets (HuggingFace)")
    except ImportError as e:
        print(f"  ✗ datasets: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("  ✓ tqdm")
    except ImportError as e:
        print(f"  ✗ tqdm: {e}")
        return False
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("  ✓ tensorboard")
    except ImportError as e:
        print(f"  ✗ tensorboard: {e}")
        return False
    
    return True


def test_project_structure():
    """Test if project structure is correct."""
    print("\nTesting project structure...")
    
    required_files = [
        "src/config.py",
        "src/models/networks.py",
        "src/preprocessing/download_replays.py",
        "src/preprocessing/preprocess_replays.py",
        "src/preprocessing/dataset.py",
        "src/training/train_bc.py",
        "src/training/train_dqn.py",
        "src/evaluation/evaluate.py",
        "requirements.txt",
        "README.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist


def test_config():
    """Test if config can be loaded."""
    print("\nTesting configuration...")
    
    try:
        sys.path.append(str(Path("src")))
        from config import Config
        
        config = Config()
        print(f"  ✓ Config loaded")
        print(f"    - Total replays: {config.TOTAL_REPLAYS:,}")
        print(f"    - Train replays: {config.TRAIN_REPLAYS:,}")
        print(f"    - Val replays: {config.VAL_REPLAYS:,}")
        print(f"    - Test replays: {config.TEST_REPLAYS:,}")
        print(f"    - Map size: {config.MAP_HEIGHT}x{config.MAP_WIDTH}")
        print(f"    - Action space: {config.NUM_ACTIONS}")
        return True
    except Exception as e:
        print(f"  ✗ Config failed: {e}")
        return False


def test_model():
    """Test if model can be instantiated."""
    print("\nTesting model...")
    
    try:
        import torch
        sys.path.append(str(Path("src")))
        from config import Config
        from models.networks import DuelingDQN
        
        config = Config()
        model = DuelingDQN(
            num_channels=config.NUM_CHANNELS,
            num_actions=config.NUM_ACTIONS,
            cnn_channels=config.CNN_CHANNELS
        )
        
        # Test forward pass
        batch_size = 4
        dummy_input = torch.randn(batch_size, config.NUM_CHANNELS, 
                                   config.MAP_HEIGHT, config.MAP_WIDTH)
        output = model(dummy_input)
        
        assert output.shape == (batch_size, config.NUM_ACTIONS)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model instantiated")
        print(f"    - Parameters: {params:,}")
        print(f"    - Input shape: {dummy_input.shape}")
        print(f"    - Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Model failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device():
    """Test available compute device."""
    print("\nTesting compute device...")
    
    try:
        import torch
        
        if torch.backends.mps.is_available():
            print("  ✓ MPS (Apple Silicon) available")
            device = torch.device("mps")
        elif torch.cuda.is_available():
            print("  ✓ CUDA available")
            device = torch.device("cuda")
        else:
            print("  ⚠ Using CPU (training will be slower)")
            device = torch.device("cpu")
        
        # Test tensor operation on device
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = x @ y
        
        print(f"    - Device: {device}")
        print(f"    - Test computation successful")
        return True
    except Exception as e:
        print(f"  ✗ Device test failed: {e}")
        return False


def main():
    print("="*60)
    print("Generals.io DRL Project - Setup Verification")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("Configuration", test_config()))
    results.append(("Model", test_model()))
    results.append(("Device", test_device()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*60)
    if all_passed:
        print("✅ All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Test download script with small sample:")
        print("     python src/preprocessing/download_replays.py --num_replays 100")
        print("  2. Run full pipeline:")
        print("     ./quickstart.sh")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - Import errors: source venv/bin/activate")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
