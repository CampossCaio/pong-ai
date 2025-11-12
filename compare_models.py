#!/usr/bin/env python3

import sys
import os
import pickle
import torch
import neat
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path


sys.path.append(os.path.dirname(__file__))
from environment.pong import Pong
from agents.dqn.agent import PongAgent

def test_dqn_model(model_path, episodes=50):
    env = Pong(render_mode=None, difficulty="easy")
    agent = PongAgent(state_dim=12, action_dim=3)
    
    # Load model
    if os.path.exists(model_path):
        agent.load(model_path)
        agent.exploration_rate = 0
    else:
        print(f"DQN model not found at {model_path}")
        return None
    
    results = []
    start_time = time.time()
    
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        results.append({
            'reward': episode_reward,
            'player_points': info['player_points'],
            'cpu_points': info['cpu_points'],
            'steps': steps,
            'win': 1 if info['player_points'] > info['cpu_points'] else 0
        })
    
    env.close()
    test_time = time.time() - start_time
    
    return {
        'results': results,
        'test_time': test_time,
        'model_type': 'DQN'
    }

def test_neat_model(model_path, episodes=50):
    env = Pong(render_mode=None, difficulty="easy")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"NEAT model not found at {model_path}")
        return None
    
    with open(model_path, 'rb') as f:
        winner = pickle.load(f)
    

    config_path = os.path.join(os.path.dirname(__file__), 'agents', 'neat', 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    results = []
    start_time = time.time()
    
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            output = net.activate(state)
            action = int(np.argmax(output))
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        results.append({
            'reward': episode_reward,
            'player_points': info['player_points'],
            'cpu_points': info['cpu_points'],
            'steps': steps,
            'win': 1 if info['player_points'] > info['cpu_points'] else 0
        })
    
    env.close()
    test_time = time.time() - start_time
    
    return {
        'results': results,
        'test_time': test_time,
        'model_type': 'NEAT'
    }

def compare_results(dqn_data, neat_data):
    if not dqn_data or not neat_data:
        print("Missing model data for comparison")
        return
    
    def calc_metrics(data):
        results = data['results']
        return {
            'avg_reward': np.mean([r['reward'] for r in results]),
            'avg_player_score': np.mean([r['player_points'] for r in results]),
            'avg_cpu_score': np.mean([r['cpu_points'] for r in results]),
            'win_rate': np.mean([r['win'] for r in results]) * 100,
            'avg_steps': np.mean([r['steps'] for r in results]),
            'test_time': data['test_time']
        }
    
    dqn_metrics = calc_metrics(dqn_data)
    neat_metrics = calc_metrics(neat_data)
    

    print("\n=== MODEL COMPARISON ===")
    print(f"{'Metric':<20} {'DQN':<10} {'NEAT':<10}")
    print("-" * 40)
    print(f"{'Avg Reward':<20} {dqn_metrics['avg_reward']:<10.2f} {neat_metrics['avg_reward']:<10.2f}")
    print(f"{'Win Rate %':<20} {dqn_metrics['win_rate']:<10.1f} {neat_metrics['win_rate']:<10.1f}")
    print(f"{'Avg Player Score':<20} {dqn_metrics['avg_player_score']:<10.1f} {neat_metrics['avg_player_score']:<10.1f}")
    print(f"{'Avg CPU Score':<20} {dqn_metrics['avg_cpu_score']:<10.1f} {neat_metrics['avg_cpu_score']:<10.1f}")
    print(f"{'Avg Steps':<20} {dqn_metrics['avg_steps']:<10.1f} {neat_metrics['avg_steps']:<10.1f}")
    print(f"{'Test Time (s)':<20} {dqn_metrics['test_time']:<10.2f} {neat_metrics['test_time']:<10.2f}")
    
  
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    

    dqn_rewards = [r['reward'] for r in dqn_data['results']]
    neat_rewards = [r['reward'] for r in neat_data['results']]
    
    ax1.plot(dqn_rewards, 'b-', alpha=0.7, label='DQN')
    ax1.plot(neat_rewards, 'r-', alpha=0.7, label='NEAT')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    

    models = ['DQN', 'NEAT']
    win_rates = [dqn_metrics['win_rate'], neat_metrics['win_rate']]
    ax2.bar(models, win_rates, color=['blue', 'red'], alpha=0.7)
    ax2.set_title('Win Rate Comparison')
    ax2.set_ylabel('Win Rate (%)')
    ax2.grid(True, alpha=0.3)
    

    dqn_scores = [r['player_points'] for r in dqn_data['results']]
    neat_scores = [r['player_points'] for r in neat_data['results']]
    
    ax3.plot(dqn_scores, 'b-', alpha=0.7, label='DQN')
    ax3.plot(neat_scores, 'r-', alpha=0.7, label='NEAT')
    ax3.set_title('Player Scores')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    metrics = ['Avg Reward', 'Win Rate', 'Avg Score']
    dqn_vals = [dqn_metrics['avg_reward'], dqn_metrics['win_rate'], dqn_metrics['avg_player_score']]
    neat_vals = [neat_metrics['avg_reward'], neat_metrics['win_rate'], neat_metrics['avg_player_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, dqn_vals, width, label='DQN', alpha=0.7)
    ax4.bar(x + width/2, neat_vals, width, label='NEAT', alpha=0.7)
    ax4.set_title('Performance Metrics')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('DQN vs NEAT Performance Comparison')
    plt.tight_layout()
    
    # Save plot
    results_dir = Path("results/plots/comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(results_dir / f"dqn_vs_neat_comparison_{timestamp}.png")
    plt.show()

def main():
    print("=== DQN vs NEAT Model Comparison ===")
    
    dqn_model = "models/dqn/pong_ai_79.0.chkpt" # change to your DQN model path
    neat_model = "models/neat/winner.pkl" # change to your NEAT model path
    
    episodes = 100
    print(f"Testing both models with {episodes} episodes each...")
    

    print("\nTesting DQN model...")
    dqn_data = test_dqn_model(dqn_model, episodes)
    
    print("Testing NEAT model...")
    neat_data = test_neat_model(neat_model, episodes)
    
    compare_results(dqn_data, neat_data)

if __name__ == "__main__":
    main()