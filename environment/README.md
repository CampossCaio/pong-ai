# Pong Environment

Custom Pong environment built specifically for AI training and comparison between DQN and NEAT algorithms.

## Overview

This environment implements a Gymnasium-compatible Pong game designed to provide a fair and challenging testbed for reinforcement learning and evolutionary algorithms.

## Files

- `pong.py` - Main environment implementation with Gymnasium interface

## Key Features

### Gymnasium Compatibility
- Standard `reset()`, `step()`, `render()` interface
- Discrete action space: [0: stay, 1: up, 2: down]
- 12-dimensional observation space with normalized values

### Observation Space (12 features)
1. Player Y position (normalized)
2. Player speed
3. Ball X position (normalized)
4. Ball Y position (normalized)
5. Ball X velocity
6. Ball Y velocity
7. Relative X distance to ball
8. Relative Y distance to ball
9. Distance to ball
10. Ball approaching speed
11. Score difference
12. Game progress

### Enhanced Reward System

**Basic Rewards:**
- +15/+20 points for scoring (difficulty dependent)
- -15/-20 points for conceding
- Small step penalty (-0.001) in normal mode

**Strategic Bonuses:**
- Angle change bonus: +1-2 points for strategic ball deflection
- Speed bonus: +0.5-1 points for aggressive play
- Positioning bonus: +0.1-0.2 points for good defensive positioning

**End-game Rewards:**
- +25/+30 points for winning
- -25/-30 points for losing

### Anti-Camping System

To prevent reward hacking where agents exploit environment bugs:
- **Camping Detection**: Monitors if ball stays near player paddle (x < 100) for >50 steps
- **Penalty**: -1.0 reward per step when camping detected
- **Purpose**: Forces agents to learn genuine Pong strategies

### Difficulty Levels

**Easy Mode:**
- Slower, imperfect CPU opponent with random errors
- No step penalty for easier learning
- Higher rewards for ball interactions
- Max 5 points to win

**Normal Mode:**
- Perfect CPU opponent
- Step penalty to encourage efficiency
- Standard reward values
- Max 2 points to win

### Rendering Modes

**Human Mode (`render_mode="human"`):**
- Visual pygame window
- Real-time gameplay at 60 FPS
- Score display and game graphics

**Headless Mode (`render_mode=None`):**
- No visual output
- Maximum training speed
- Same game logic and physics

## Usage

### Basic Usage
```python
from environment.pong import Pong

# Create environment
env = Pong(render_mode=None, difficulty="easy")

# Reset environment
obs, info = env.reset()

# Take action
action = 1  # Move up
obs, reward, terminated, truncated, info = env.step(action)
```

### Training Integration
```python
# For DQN training
env = Pong(render_mode=None, difficulty="easy")

# For NEAT evaluation
env = Pong(render_mode=None, difficulty="easy")

# For human testing
env = Pong(render_mode="human", difficulty="easy")
```