# Pong AI: DQN vs NEAT Comparison

A comprehensive comparison between Deep Q-Network (DQN) and NeuroEvolution of Augmenting Topologies (NEAT) algorithms for learning to play Pong.

## Project Overview

This project implements and compares two different AI approaches for mastering the classic Pong game:

- **DQN**: Deep reinforcement learning using neural networks and experience replay
- **NEAT**: Evolutionary algorithm that evolves both network topology and weights

**🎮 Custom Environment**: We developed a specialized Pong environment using Pygame, designed specifically for AI training and fair algorithm comparison. This environment includes advanced features like reward hacking prevention, strategic reward shaping, and comprehensive state representation.

## Project Structure

```
pong-ai/
├── agents/
│   ├── dqn/                 # Deep Q-Network implementation
│   │   ├── agent.py         # DQN agent with experience replay
│   │   ├── network.py       # Neural network architecture
│   │   ├── train.py         # Training script
│   │   ├── test.py          # Testing script
│   │   └── plot.py          # Visualization utilities
│   └── neat/                # NEAT implementation
│       ├── evolve_pong.py   # Evolution script
│       ├── test_neat_agent.py # Testing script
│       ├── config           # NEAT parameters
│       └── visualize.py     # Network visualization
├── environment/
│   └── pong.py             # 🎮 Custom-built Pong environment for AI training
├── models/                 # Trained models (gitignored)
├── results/                # Training results and plots (gitignored)
├── utils/
│   └── metrics.py          # Performance metrics utilities
├── compare_models.py       # Head-to-head comparison script
└── requirements.txt        # Python dependencies
```

## Custom Environment Features

**🏗️ Custom Implementation**: We developed a complete Pong environment using Pygame specifically for this AI comparison project.

Our custom environment includes:

- **Gymnasium-compatible interface** for easy integration
- **Enhanced reward system** with strategic bonuses
- **Difficulty levels** (easy/normal) for progressive learning
- **Camping detection** to prevent reward hacking
- **Comprehensive state representation** (12-dimensional observation space)
- **Headless mode** for fast training without graphics

## Quick Start

### Installation
```bash
cd pong-ai
pip install -r requirements.txt
```

### Train DQN Agent
```bash
cd agents/dqn
python train.py
```

### Train NEAT Agent
```bash
cd agents/neat
python evolve_pong.py
```

### Compare Both Models
```bash
python compare_models.py
```

## Key Features

### DQN Implementation
- Experience replay buffer for stable learning
- Target network for reduced correlation
- Epsilon-greedy exploration strategy
- Advanced metrics tracking (Q-values, losses, win rates)

### NEAT Implementation
- Automatic topology evolution
- Speciation for diversity maintenance
- Fitness-based selection and reproduction
- Network visualization capabilities

### Environment Enhancements
- **Strategic reward shaping**: Bonuses for angle changes, speed increases, positioning
- **Anti-camping system**: Detects and penalizes reward hacking behaviors
- **Difficulty scaling**: Adjustable CPU opponent strength
- **Comprehensive metrics**: Win rates, scores, episode lengths, natural endings

## Results and Analysis

Both algorithms generate:
- **Training curves**: Reward progression over time
- **Performance metrics**: Win rates, average scores, training time
- **Visual comparisons**: Side-by-side performance analysis
- **Model persistence**: Saved models for future testing

Results are automatically saved to:
- `results/plots/dqn/` - DQN training visualizations
- `results/plots/neat/` - NEAT evolution visualizations  
- `results/plots/comparison/` - Head-to-head comparisons
- `models/` - Trained model files

## Research Questions

This project explores:
1. **Learning efficiency**: Which algorithm learns faster?
2. **Final performance**: Which achieves better gameplay?
3. **Stability**: Which is more consistent across runs?
4. **Interpretability**: How do the learned strategies differ?
5. **Computational requirements**: Training time and resource usage

## Dependencies

- Python 3.8+
- PyTorch (for DQN)
- NEAT-Python (for NEAT)
- Pygame (for environment)
- Gymnasium (for RL interface)
- Matplotlib (for visualization)
- NumPy (for numerical operations)

## Contributing

Feel free to experiment with:
- Different network architectures
- Alternative reward functions
- New hyperparameter configurations
- Additional comparison metrics
- Extended environment features

## References

1. **NEAT Algorithm**: Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary computation*, 10(2), 99-127. [https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf](https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf)

2. **NEAT-Python Library**: CodeReclaimers. NEAT-Python: A pure Python implementation of NEAT. [https://github.com/CodeReclaimers/neat-python](https://github.com/CodeReclaimers/neat-python)

3. **NEAT for 2D Games**: Selvan, J. P., & Game, P. S. Playing a 2D Game Indefinitely using NEAT and Reinforcement Learning. *arXiv preprint*. [https://arxiv.org/pdf/2207.14140](https://arxiv.org/pdf/2207.14140)

4. **Neuroevolution in Gaming**: O'Connor, J., Parker, G. B., & Bugti, M. Learning Dark Souls Combat Through Pixel Input With Neuroevolution. *arXiv preprint*. [https://arxiv.org/pdf/2507.03793v1](https://arxiv.org/pdf/2507.03793v1)

5. **Double DQN**: van Hasselt, H., Guez, A., & Silver, D. Deep Reinforcement Learning with Double Q-learning. *arXiv preprint*. [https://arxiv.org/pdf/1509.06461](https://arxiv.org/pdf/1509.06461)