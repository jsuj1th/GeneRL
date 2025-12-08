#!/usr/bin/env python3
"""
Test winner detection logic with all possible winner values.
"""

import numpy as np

# Simulate different winner values from the environment
agent_names = ["PPO_Agent", "Opponent"]

test_cases = [
    ("Agent wins (string)", "PPO_Agent"),
    ("Agent wins (True)", True),
    ("Agent wins (np.True_)", np.True_),
    ("Agent wins (np.array(True))", np.array(True)),
    ("Opponent wins (string)", "Opponent"),
    ("Opponent wins (False)", False),
    ("Opponent wins (np.False_)", np.False_),
    ("Opponent wins (np.array(False))", np.array(False)),
    ("Draw (None)", None),
]

print("="*70)
print("🧪 TESTING WINNER DETECTION LOGIC")
print("="*70)
print()

for test_name, winner_value in test_cases:
    print(f"Test: {test_name}")
    print(f"  Value: {winner_value}")
    print(f"  Type: {type(winner_value)}")
    
    # Apply the conversion logic
    winner = winner_value
    if hasattr(winner, 'item'):
        winner = winner.item()
    
    print(f"  After .item(): {winner} (type: {type(winner)})")
    
    # Apply the detection logic
    if winner == agent_names[0] or winner == True:
        agent_won = True
        result = "WON ✓"
    elif winner is None:
        agent_won = None
        result = "DRAW ⚖️"
    else:
        agent_won = False
        result = "LOST ✗"
    
    print(f"  Detected as: {result}")
    print(f"  agent_won: {agent_won}")
    
    # Check if detection is correct
    expected = None
    if test_name.startswith("Agent wins"):
        expected = True
    elif test_name.startswith("Opponent wins"):
        expected = False
    elif test_name.startswith("Draw"):
        expected = None
    
    if agent_won == expected:
        print(f"  ✅ CORRECT")
    else:
        print(f"  ❌ WRONG! Expected agent_won={expected}, got {agent_won}")
    
    print()

print("="*70)
print("✅ Test complete")
print("="*70)
