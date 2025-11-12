# DQN Pong Agent

Deep Q-Network implementation for playing Pong using PyTorch.

## Overview

This DQN agent learns to play Pong through reinforcement learning, using a neural network to approximate Q-values for state-action pairs.

## Files

- `agent.py` - Main DQN agent with experience replay and target network
- `network.py` - Neural network architecture (3-layer MLP)
- `train.py` - Training script with enhanced metrics tracking
- `test.py` - Test trained models
- `plot.py` - Visualization utilities for training results
- `simple_replay_buffer.py` - Experience replay buffer implementation

## Features

- **Experience Replay**: Stores and samples past experiences for stable learning
- **Target Network**: Separate target network updated periodically for stability
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Enhanced Metrics**: Tracks rewards, win rates, Q-values, losses, and training time
- **Camping Detection**: Prevents reward hacking through position-based penalties

## How to Run

### Training
```bash
cd pong-ai/agents/dqn
python train.py
```

### Testing
```bash
cd pong-ai/agents/dqn
python test.py
```

## Configuration

Key hyperparameters in `agent.py`:
- Learning rate: 0.001
- Batch size: 32
- Memory size: 100,000
- Epsilon decay: 0.99995
- Target network update: every 1000 steps

## Results

- Models saved to: `../../models/dqn/`
- Training plots: `../../results/plots/dqn/`
- Logs: `../../results/logs/dqn/`

## Performance Metrics

The agent tracks:
- Episode rewards and moving averages
- Win rate against CPU
- Q-value estimates
- Training loss
- Time vs reward progression
- Natural game ending rates