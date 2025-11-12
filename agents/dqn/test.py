#!/usr/bin/env python3

import numpy as np
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from environment.pong import Pong
from agents.dqn.agent import PongAgent

def test_dqn(checkpoint_path, episodes=10, difficulty="easy"):

    env = Pong(render_mode="human", difficulty=difficulty)
    agent = PongAgent(state_dim=12, action_dim=3)
    
    agent.load(checkpoint_path)
    agent.exploration_rate = 0  # No exploration
    
    print(f"Testing DQN for {episodes} episodes")
    print(f"Model: {checkpoint_path}")
    
    results = []
    
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
        
        result = {
            'reward': episode_reward,
            'player_points': info['player_points'],
            'cpu_points': info['cpu_points'],
            'steps': steps
        }
        results.append(result)
        
        if episode < 10 or episode % 10 == 0:
            print(f"Episode {episode+1:3d}: reward={episode_reward:6.2f}, "
                  f"score={info['player_points']}-{info['cpu_points']}, steps={steps}")
    
    env.close()
    
    # Print summary
    wins = sum(1 for r in results if r['player_points'] > r['cpu_points'])
    win_rate = wins / len(results)
    avg_reward = np.mean([r['reward'] for r in results])
    
    print(f"\n=== RESULTS ===")
    print(f"Win Rate: {win_rate:.1%} ({wins}/{episodes})")
    print(f"Average Reward: {avg_reward:.2f}")
    
    return results

if __name__ == "__main__":
    # Update this path to your trained model
    model_path = "../../models/dqn/pong_ai_79.0.chkpt"
    test_dqn(model_path)