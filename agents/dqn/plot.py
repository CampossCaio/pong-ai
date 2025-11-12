#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import datetime

def plot_results(results, title="DQN Results"):
    """Simple plotting of results"""
    rewards = [r['reward'] for r in results]
    player_scores = [r['player_points'] for r in results]
    cpu_scores = [r['cpu_points'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Rewards
    ax1.plot(rewards, alpha=0.6)
    if len(rewards) > 10:
        # Moving average
        window = min(10, len(rewards)//4)
        ma = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        ax1.plot(ma, linewidth=2, label=f'MA({window})')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Scores
    ax2.plot(player_scores, 'b-', label='Player', alpha=0.7)
    ax2.plot(cpu_scores, 'r-', label='CPU', alpha=0.7)
    ax2.set_title('Scores per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save plot in results directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "..", "..", "results", "plots", "dqn")
    os.makedirs(plots_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{title.lower().replace(' ', '_')}_{timestamp}.png"
    plt.savefig(os.path.join(plots_dir, filename))
    
    plt.show()

def plot_time_vs_reward(time_reward_data, title="DQN Time vs Reward"):

    times = [d['time_minutes'] for d in time_reward_data]
    rewards = [d['avg_reward'] for d in time_reward_data]
    episodes = [d['episode'] for d in time_reward_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time vs Reward
    ax1.plot(times, rewards, 'b-', linewidth=2)
    ax1.set_title('Reward vs Training Time')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True, alpha=0.3)
    
    # Episode vs Reward
    ax2.plot(episodes, rewards, 'r-', linewidth=2)
    ax2.set_title('Reward vs Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save plot
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "..", "..", "results", "plots", "dqn")
    os.makedirs(plots_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"time_vs_reward_{timestamp}.png"
    plt.savefig(os.path.join(plots_dir, filename))
    
    plt.show()

if __name__ == "__main__":
    print("This is a plotting utility. Import and use plot_results() function.")