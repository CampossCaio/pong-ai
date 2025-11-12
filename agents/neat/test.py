import os
import sys
import pickle
import neat
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from environment.pong import Pong

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Load winner genome
    with open('checkpoints/winner.pkl', 'rb') as f:
        genome = pickle.load(f)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = Pong(render_mode="human", difficulty="easy")
    
    N_EPISODES = 10
    for ep in range(N_EPISODES):
        obs, info = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            output = net.activate(obs)
            action = int(np.argmax(output))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {ep+1}: reward={total_reward:.3f}, score={info['player_points']}-{info['cpu_points']}")

    env.close()

if __name__ == "__main__":
    main()