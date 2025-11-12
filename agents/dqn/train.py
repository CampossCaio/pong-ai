#!/usr/bin/env python3

import torch
import numpy as np
from pathlib import Path
import datetime
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from environment.pong import Pong
from agents.dqn.agent import PongAgent
from agents.dqn.plot import plot_results, plot_time_vs_reward

def train_dqn(episodes=200, difficulty="easy"):

    save_dir = Path("../../results/logs/dqn") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path("../../models/dqn")
    models_dir.mkdir(parents=True, exist_ok=True)
    

    env = Pong(render_mode=None, difficulty=difficulty)
    agent = PongAgent(state_dim=12, action_dim=3, save_dir=save_dir)
    
    print(f"Training DQN for {episodes} episodes (difficulty: {difficulty})")
    
    # metrics tracking
    results = []
    losses = []
    q_values = []
    wins = []
    
    # Time tracking
    start_time = time.time()
    time_reward_data = []  # For plotting time vs reward
    
    # Training loop
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.cache(state, next_state, action, reward, terminated)
            
            # Learn and collect metrics
            q, loss = agent.learn()
            if q is not None:
                q_values.append(q)
            if loss is not None:
                losses.append(loss)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if terminated or truncated:
                break
        
        # Store episode results
        win = 1 if info['player_points'] > info['cpu_points'] else 0
        wins.append(win)
        
        results.append({
            'reward': episode_reward,
            'player_points': info['player_points'],
            'cpu_points': info['cpu_points'],
            'steps': episode_steps
        })
        
        # Enhanced logging every 10 episodes
        if episode % 10 == 0:
            recent_episodes = min(10, episode + 1)
            avg_reward = np.mean([r['reward'] for r in results[-recent_episodes:]])
            avg_steps = np.mean([r['steps'] for r in results[-recent_episodes:]])
            win_rate = np.mean(wins[-recent_episodes:]) * 100
            avg_player_score = np.mean([r['player_points'] for r in results[-recent_episodes:]])
            avg_cpu_score = np.mean([r['cpu_points'] for r in results[-recent_episodes:]])
            
            # Loss and Q-value metrics
            avg_loss = np.mean(losses[-100:]) if losses else 0
            avg_q = np.mean(q_values[-100:]) if q_values else 0
            
            # Natural ending rate
            natural_endings = sum(1 for r in results[-recent_episodes:] if r['steps'] < 1000)
            natural_rate = (natural_endings / recent_episodes) * 100
            
            elapsed_time = time.time() - start_time
            
            # Store time vs reward data
            time_reward_data.append({
                'time_minutes': elapsed_time / 60,
                'avg_reward': avg_reward,
                'episode': episode
            })
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {avg_reward:6.2f} | "
                  f"Steps: {avg_steps:5.1f} | "
                  f"Win%: {win_rate:5.1f} | "
                  f"Score: {avg_player_score:.1f}-{avg_cpu_score:.1f} | "
                  f"Natural%: {natural_rate:4.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Q: {avg_q:5.2f} | "
                  f"ε: {agent.exploration_rate:.3f} | "
                  f"Time: {elapsed_time/60:.1f}min")
    
    env.close()
    
    # Final statistics
    final_episodes = min(50, len(results))
    final_win_rate = np.mean(wins[-final_episodes:]) * 100
    final_avg_reward = np.mean([r['reward'] for r in results[-final_episodes:]])
    final_avg_score = np.mean([r['player_points'] for r in results[-final_episodes:]])
    
    total_time = time.time() - start_time
    
    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Total Training Time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Model saved to: {save_dir}")
    print(f"Final Win Rate (last {final_episodes} episodes): {final_win_rate:.1f}%")
    print(f"Final Average Reward: {final_avg_reward:.2f}")
    print(f"Final Average Player Score: {final_avg_score:.2f}")
    print(f"Final Epsilon: {agent.exploration_rate:.3f}")
    
    # Plot training results
    print("\nPlotting training results...")
    plot_results(results, "DQN Training Results")
    
    # Plot time vs reward
    if time_reward_data:
        plot_time_vs_reward(time_reward_data, "DQN Time vs Reward")
    
    return results, total_time, time_reward_data

if __name__ == "__main__":
    train_dqn(episodes=400)