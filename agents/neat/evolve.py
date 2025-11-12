import os
import sys
import neat
import numpy as np
import multiprocessing
import pickle
import visualize
import time

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from environment.pong import Pong

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config')
N_EPISODES_PER_GENOME = 10
NUM_CORES = multiprocessing.cpu_count()

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = Pong(render_mode=None, difficulty="easy")
    
    episode_scores = []
    
    for _ in range(N_EPISODES_PER_GENOME):
        total_reward = 0.0
        observation, info = env.reset()
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            output = net.activate(observation)
            action = int(np.argmax(output))
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
        episode_scores.append(total_reward)
        
    env.close()
    return np.mean(episode_scores)

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pop.add_reporter(neat.Checkpointer(10, filename_prefix=os.path.join(script_dir, "neat-checkpoint-")))

    # Single-threaded version: It would be better to use multiprocessing, but
    # it would be unfair to DQN, so i decided to keep it simple and single-threaded.
    def evaluate_genomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = eval_genome(genome, config)
    
    # Time tracking
    start_time = time.time()
    
    print(f"Starting training (single-threaded)...")
    pop.run(evaluate_genomes, n=100)
    
    total_time = time.time() - start_time

    winner = stats.best_genome()
    final_mean_reward = stats.get_fitness_mean()[-1] if stats.get_fitness_mean() else 0
    
    if winner:
        print(f"\n=== NEAT TRAINING COMPLETED ===")
        print(f"Total Training Time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        print(f"Best Fitness: {winner.fitness:.2f}")
        print(f"Final Mean Reward: {final_mean_reward:.2f}")

        # Save winner locally in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_checkpoints = os.path.join(script_dir, "checkpoints")
        os.makedirs(local_checkpoints, exist_ok=True)
        winner_path = os.path.join(local_checkpoints, "winner.pkl")
        
        # Save to models directory  
        models_dir = os.path.join(script_dir, "..", "..", "models", "neat")
        os.makedirs(models_dir, exist_ok=True)
        models_path = os.path.join(models_dir, "winner.pkl")
        
        # Save locally
        with open(winner_path, 'wb') as f:
            pickle.dump(winner, f)
        
        # Save to models directory
        with open(models_path, 'wb') as f:
            pickle.dump(winner, f)
            
        print(f"Winner genome saved to {winner_path} and {models_path}")
        
    else:
        print(f"\n=== NEAT TRAINING COMPLETED ===")
        print(f"Total Training Time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        print(f"No solution found after 100 generations.")
        
    # Plot statistics to both local and results directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    visualize.plot_stats(stats, ylog=False, view=False, 
                        filename=os.path.join(script_dir, "fitness_stats.svg"))
    visualize.plot_species(stats, view=False, 
                          filename=os.path.join(script_dir, "species.svg"))
    if winner:
        visualize.draw_net(config, winner, view=False, 
                          filename=os.path.join(script_dir, "winner-net.gv"))
    
    # Also save to results/plots/neat
    plots_dir = os.path.join(script_dir, "..", "..", "results", "plots", "neat")
    os.makedirs(plots_dir, exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    visualize.plot_stats(stats, ylog=False, view=False, 
                        filename=os.path.join(plots_dir, f"fitness_stats_{timestamp}.svg"))
    visualize.plot_species(stats, view=False, 
                          filename=os.path.join(plots_dir, f"species_{timestamp}.svg"))
    if winner:
        visualize.draw_net(config, winner, view=False, 
                          filename=os.path.join(plots_dir, f"winner-net_{timestamp}.gv"))
    
    # Create time vs fitness data using stats
    generations = range(len(stats.most_fit_genomes))
    best_fitness = [c.fitness for c in stats.most_fit_genomes]
    avg_fitness = stats.get_fitness_mean()
    
    time_fitness_data = []
    for i, gen in enumerate(generations):
        time_fitness_data.append({
            'time_minutes': (total_time / len(generations)) * (i + 1) / 60,
            'avg_fitness': avg_fitness[i],
            'best_fitness': best_fitness[i],
            'generation': gen
        })

    print("\nPlotting time vs fitness...")
    visualize.plot_time_vs_fitness(time_fitness_data, "NEAT Time vs Fitness")
    
    return winner, total_time, time_fitness_data

if __name__ == '__main__':
    run(CONFIG_PATH)