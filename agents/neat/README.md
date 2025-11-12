# NEAT Pong Agent

NeuroEvolution of Augmenting Topologies implementation for playing Pong.

## Overview

This NEAT agent evolves neural networks through genetic algorithms, automatically discovering optimal network topologies and weights for playing Pong.

## Files

- `evolve_pong.py` - Main evolution script with fitness evaluation
- `test_neat_agent.py` - Test evolved agents
- `config` - NEAT configuration parameters
- `visualize.py` - Network visualization and training plots
- `episode_reporter.py` - Custom reporter for episode tracking

## Features

- **Topology Evolution**: Automatically evolves network structure
- **Speciation**: Maintains diversity through species separation
- **Fitness-Based Selection**: Selects best performers for reproduction
- **Network Visualization**: Generates visual representations of evolved networks
- **Time Tracking**: Monitors training time vs fitness progression
- **Checkpointing**: Saves progress every 10 generations

## How to Run

### Training
```bash
cd pong-ai/agents/neat
python evolve_pong.py
```

### Testing
```bash
cd pong-ai/agents/neat
python test_neat_agent.py
```

## Configuration

Key parameters in `config`:
- Population size: 150
- Max generations: 100
- Fitness threshold: 300
- Episodes per genome: 10
- Mutation rates: Various (see config file)

## Results

- Best genomes: `checkpoints/winner.pkl`
- Models: `../../models/neat/`
- Fitness plots: `fitness_stats.svg`
- Species plots: `species.svg`
- Network diagrams: `winner-net.gv`

## Evolution Process

1. **Initialize**: Random population of minimal networks
2. **Evaluate**: Test each genome on multiple Pong episodes
3. **Select**: Choose best performers based on fitness
4. **Reproduce**: Create offspring with mutations/crossover
5. **Speciate**: Group similar networks to maintain diversity
6. **Repeat**: Continue until solution found or max generations

## Fitness Function

Agents are evaluated on:
- Average reward across multiple episodes
- Consistency across different game scenarios
- Ability to score points and defend effectively