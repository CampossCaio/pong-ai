import pygame
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

pygame.init()
score_font = pygame.font.Font(None, 100)

class Pong(gym.Env):
  def __init__(self, width = 800, height = 500, render_mode=None, difficulty="normal"):
    self.width = width
    self.height = height
    self.render_mode = render_mode
    self.difficulty = difficulty
    
    # Gymnasium spaces
    self.action_space = spaces.Discrete(3)  # 0: nothing, 1: up, 2: down
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
    
    if render_mode == "human":
      self.screen = pygame.display.set_mode((self.width, self.height))
      pygame.display.set_caption(f"Pong - {difficulty.capitalize()}!")
      self.clock = pygame.time.Clock()
    else:
      self.screen = None
      self.clock = None
    
    self.ball = pygame.Rect(0, 0, 20, 20)
    self.ball_speed_x = 6
    self.ball_speed_y = 6
    
    self.cpu = pygame.Rect(0, 0, 20, 100)
    self.cpu.midright = (self.width - 20, self.height / 2)
    self.cpu_speed = 10
        
    self.player = pygame.Rect(0, 0, 20, 100)
    self.player.midleft = (20, self.height / 2)
    self.player_speed = 0

    self.cpu_points, self.player_points = 0, 0
    
    self.game_over = False
    self.max_points = 5 if difficulty == "easy" else 2
    self.steps = 0
    self.max_steps = 1000
    
    # Ball camping tracking
    self.camping_count = 0
  
  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    if seed is not None:
      random.seed(seed)
      np.random.seed(seed)
    
    self.cpu_points = 0
    self.player_points = 0
    self.game_over = False
    self.steps = 0
    self._prev_ball_speed_x = 6
    self._prev_ball_speed_y = 6
    self.camping_count = 0
    self._reset_ball()
    self._reset_players()
    
    observation = self._get_observation()
    info = self._get_info()
    
    return observation, info
    
  
  def _reset_ball(self):
    self.ball_speed_y = 6
    
    self.ball.center = (self.width / 2, self.height / 2)
    self.ball_speed_x *= random.choice([-1, 1])
    self.ball_speed_y *= random.choice([-1, 1])
    
  def _reset_players(self):
    self.cpu.midright = (self.width - 20, self.height / 2)
    self.player.midleft = (20, self.height / 2)
    
  def _update_score(self, striker):
    if striker == 'player':
        self.player_points += 1
    if striker == 'cpu':
        self.cpu_points += 1
    self._reset_ball()
    
    # Check if the game is over
    if self.player_points >= self.max_points or self.cpu_points >= self.max_points:
        self.game_over = True
        
  
  def _freeze_screen(self, milliseconds):
    if self.render_mode == "human" and self.screen is not None:
        self.draw()
        pygame.display.update()
        pygame.time.wait(milliseconds)
    
  def draw(self):
    self.screen.fill('black')

    self.cpu_score = score_font.render(str(self.cpu_points), True, 'white')
    self.player_score = score_font.render(str(self.player_points), True, 'white')

    self.screen.blit(self.player_score, (self.width / 4, 20))
    self.screen.blit(self.cpu_score, (3 * self.width / 4, 20))

    pygame.draw.aaline(self.screen, 'white', (self.width / 2, 0), (self.width / 2, self.height))
    pygame.draw.rect(self.screen, 'white', self.ball)
    pygame.draw.rect(self.screen, 'red', self.cpu)
    pygame.draw.rect(self.screen, 'blue', self.player)
    
  
  def _animate_ball(self):
    self.ball.x += self.ball_speed_x
    self.ball.y += self.ball_speed_y

    if self.ball.top <= 0 or self.ball.bottom >= self.height:
        self.ball_speed_y *= -1

    if self.ball.right >= self.width:
        self._update_score('player')

    if self.ball.left <= 0:
        self._update_score('cpu')

    if self.ball.colliderect(self.player) or self.ball.colliderect(self.cpu):
        if self.ball.colliderect(self.player):
            offset = self.ball.centery - self.player.centery
        elif self.ball.colliderect(self.cpu):
            offset = self.ball.centery - self.cpu.centery

        self.ball_speed_x *= -1
        self.ball_speed_y = offset * 0.35
        
  def _animate_player(self):
    self.player.y += self.player_speed

    if self.player.top <= 0:
        self.player.top = 0

    if self.player.bottom >= self.height:
        self.player.bottom = self.height
        
  def _animate_cpu(self):
      
    # CPU is almost perfect in the normal mode, so I needed to make it worse so the agent can learn
    if self.difficulty == "easy":
        # Easier CPU: slower and with random errors
        if self.ball.centery < self.cpu.centery:
            target_speed = -5
        else:
            target_speed = 5
        
        # Add random error
        error = random.uniform(-0.5, 0.5)
        self.cpu_speed = target_speed + error
    else:
        # Normal difficulty: original behavior(Perfect CPU)
        if self.ball.centery < self.cpu.centery:
            self.cpu_speed = -10
        else:
            self.cpu_speed = 10

    self.cpu.y += self.cpu_speed

    if self.cpu.top <= 0:
        self.cpu.top = 0

    if self.cpu.bottom >= self.height:
        self.cpu.bottom = self.height
    
  def step(self, action):
    # Sorry guys, this function has lots of comments, but it is necessary to explain the reward system :(
    # This is the most important part of the environment for training the agents.
    
    self.steps += 1
    

    self._apply_action(action)
    
    # Store previous scores for reward calculation
    prev_player_points = self.player_points
    prev_cpu_points = self.cpu_points
    
    if not self.game_over:
        self._animate_ball()
        self._animate_player()
        self._animate_cpu()
    
    # Enhanced reward system with difficulty scaling
    if self.difficulty == "easy":
        reward = 0
        score_reward = 15
        ball_hit_reward = 1.0
        angle_reward = 2.0
        speed_reward = 1.0
        position_reward = 0.2 
    else:
        reward = -0.001 
        score_reward = 20
        ball_hit_reward = 0.5
        angle_reward = 1.0
        speed_reward = 0.5
        position_reward = 0.1
    
    # Main game outcomes
    if self.player_points > prev_player_points:
        reward = score_reward
    elif self.cpu_points > prev_cpu_points:
        reward = -score_reward
    
    # Ball interaction rewards (same logic, different values)
    if hasattr(self, '_prev_ball_speed_x') and hasattr(self, '_prev_ball_speed_y'):
        # Check if ball was hit by player
        if self._prev_ball_speed_x != self.ball_speed_x and self.ball.x < self.width / 2:
            # Calculate angle change (offensive strategy)
            prev_angle = np.arctan2(self._prev_ball_speed_y, abs(self._prev_ball_speed_x))
            new_angle = np.arctan2(self.ball_speed_y, abs(self.ball_speed_x))
            angle_change = abs(new_angle - prev_angle)
            
            # Reward for hitting ball
            reward += ball_hit_reward
            
            # Bonus for changing angle (strategic play)
            if angle_change > 0.3:  # Significant angle change
                reward += angle_reward
            
            # Bonus for increasing ball speed (aggressive play)
            prev_speed = np.sqrt(self._prev_ball_speed_x**2 + self._prev_ball_speed_y**2)
            new_speed = np.sqrt(self.ball_speed_x**2 + self.ball_speed_y**2)
            if new_speed > prev_speed:
                reward += speed_reward
    
    # Positioning reward (defensive)
    if self.ball_speed_x < 0:  # Ball coming towards player
        alignment_diff = abs(self.ball.centery - self.player.centery)
        if alignment_diff < 30:  # Well positioned
            reward += position_reward
    
   
    self._prev_ball_speed_x = self.ball_speed_x
    self._prev_ball_speed_y = self.ball_speed_y
    
    # Check if ball stuck near player (camping detection)
    ball_near_player = self.ball.x < 100  # Close to player paddle
    if ball_near_player:
        self.camping_count += 1
    else:
        self.camping_count = 0
    
    # Penalize camping behavior: There is a bug in the game. 
    # NEAT always learns to explore it and hack the reward system. So I added this camping detection to penalize it.
    if self.camping_count > 50:
        reward -= 1.0  # Small penalty for camping
    
    
    terminated = self.game_over
    truncated = self.steps >= self.max_steps
    
   
    if terminated or truncated:
        if self.player_points > self.cpu_points:
            # Won the game
            reward += 25 if self.difficulty == "easy" else 30
        elif self.player_points < self.cpu_points:
            # Lost the game
            reward -= 25 if self.difficulty == "easy" else 30
    
    observation = self._get_observation()
    info = self._get_info()
    info['win_rate'] = 1 if self.player_points > self.cpu_points else 0 if self.player_points < self.cpu_points else 0.5
    
    if self.render_mode == "human":
        self.render()
    
    return observation, reward, terminated, truncated, info
  
  def _apply_action(self, action):
    if action == 1:  # Up
        self.player_speed = -10
    elif action == 2:  # Down
        self.player_speed = 10
    else:  # Do nothing
        self.player_speed = 0
  
  def _get_observation(self):
    # Enhanced state with relative positions and distances
    ball_to_player_x = (self.ball.x - self.player.centerx) / self.width
    ball_to_player_y = (self.ball.y - self.player.centery) / self.height
    ball_distance = np.sqrt(ball_to_player_x**2 + ball_to_player_y**2)
    
    # Ball approaching player (negative if moving away)
    ball_approaching = -self.ball_speed_x if self.ball_speed_x < 0 else 0
    
    return np.array([
        self.player.centery / self.height,  # Player Y position
        self.player_speed / 10,             # Player speed
        self.ball.x / self.width,           # Ball X position
        self.ball.y / self.height,          # Ball Y position
        self.ball_speed_x / 6,              # Ball X velocity
        self.ball_speed_y / 6,              # Ball Y velocity
        ball_to_player_x,                   # Relative X distance to ball
        ball_to_player_y,                   # Relative Y distance to ball
        ball_distance,                      # Distance to ball
        ball_approaching / 6,               # Ball approaching speed
        (self.player_points - self.cpu_points) / 2,  # Score difference
        self.steps / self.max_steps         # Game progress
    ], dtype=np.float32)
  
  def _get_info(self):
    return {
        "player_points": self.player_points,
        "cpu_points": self.cpu_points,
        "steps": self.steps
    }
  
  def render(self):
    if self.render_mode == "human" and self.screen is not None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        self.draw()
        pygame.display.update()
        self.clock.tick(60)
  
  def close(self):
    if self.screen is not None:
        pygame.display.quit()
        pygame.quit()