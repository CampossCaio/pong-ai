
import torch
import numpy as np
from agents.dqn.network import PongQNetwork
from agents.dqn.simple_replay_buffer import SimpleReplayBuffer
from pathlib import Path

class PongAgent:
  def __init__(self, state_dim=12, action_dim=3, save_dir=None, checkpoint=None):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.save_dir = save_dir
    
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    self.brain = PongQNetwork(self.state_dim, self.action_dim).float()
    self.brain = self.brain.to(device=self.device)
    
    if checkpoint:
      self.load(checkpoint)
    
    self.exploration_rate = 1.0
    self.exploration_rate_decay = 0.99995
    self.exploration_rate_min = 0.02
    self.curr_step = 0
    self.gamma = 0.95
    
    self.memory = SimpleReplayBuffer(capacity=100000, device=self.device)
    self.batch_size = 64 

    self.save_every = 5e3  # no. of experiences between saving
    self.burnin = 500  # min. experiences before training
    self.learn_every = 1  # learn every step
    self.sync_every = 1000  # sync target network less frequently for stability
    
    self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.0001)
    self.loss_fn = torch.nn.MSELoss()
    

  def act(self, state):
    """
    Given a state, choose an epsilon-greedy action and update value of step.
    """
    
    # EXPLORE
    if np.random.random() < self.exploration_rate:
      action_idx = np.random.randint(self.action_dim)
      
    # EXPLOIT
    else:
      # Handle numpy array directly (Pong returns normalized state)
      if isinstance(state, np.ndarray):
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
      else:
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
      
      action_values = self.brain(state_tensor, model="online")
      action_idx = torch.argmax(action_values, axis=1).item()
      
    # decrease exploration_rate
    self.exploration_rate *= self.exploration_rate_decay
    self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)
    
    # increment step
    self.curr_step += 1
    return action_idx
     

  def cache(self, state, next_state, action, reward, done):
    """
      Store the experience to self.memory (replay buffer)

      Inputs:
      state (numpy array),
      next_state (numpy array),
      action (int),
      reward (float),
      done (bool)
      """
    self.memory.add(state, next_state, action, reward, done)

  def recall(self):
    """
    Retrieve a batch of experiences from memory
    """
    return self.memory.sample(self.batch_size)
  
  def td_estimate(self, state, action):
    current_Q = self.brain(state, model="online")[
      torch.arange(0, len(action), device=self.device), action
    ] # Q_online(s,a)
    return current_Q
  
  @torch.no_grad() # disable gradient calculations here 
  def td_target(self, next_state, reward, done):
    next_state_Q = self.brain(next_state, model="online")
    best_action = torch.argmax(next_state_Q, axis=1)
    next_Q = self.brain(next_state, model="target")[
        torch.arange(0, len(best_action), device=self.device), best_action
    ]
    return (reward + (1 - done.float()) * self.gamma * next_Q).float()
  
  def update_Q_online(self, td_estimate, td_target):
    loss = self.loss_fn(td_estimate, td_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()
  
  def sync_Q_target(self):
    self.brain.target.load_state_dict(self.brain.online.state_dict())
    # Re-freeze target network parameters
    for p in self.brain.target.parameters():
      p.requires_grad = False
    
  
  def save(self):
    if self.save_dir is None:
      return
      
    # Save to logs directory (training checkpoints)
    save_path = (
      self.save_dir / f"pong_ai_{self.curr_step // self.save_every}.chkpt"
    )
    
    # Also save to models directory (final models)
    models_path = Path("../../models/dqn") / f"pong_ai_{self.curr_step // self.save_every}.chkpt"
    models_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_dict = dict(model=self.brain.state_dict(), exploration_rate=self.exploration_rate)
    
    # Save to logs (training checkpoint)
    torch.save(model_dict, save_path)
    
    # Save to models (final model)
    torch.save(model_dict, models_path)
    
    print(f"Pong AI saved to {save_path} and {models_path} at step {self.curr_step}")
    
  
  def load(self, load_path):
    load_path = Path(load_path) if isinstance(load_path, str) else load_path
    if not load_path.exists():
      raise ValueError(f"{load_path} does not exist")

    ckp = torch.load(load_path, map_location=self.device)
    exploration_rate = ckp.get('exploration_rate')
    state_dict = ckp.get('model')

    print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
    self.brain.load_state_dict(state_dict)
    self.exploration_rate = exploration_rate
    
  def learn(self):
    """Update online action value (Q) function with a batch of experiences"""
    if self.curr_step % self.sync_every == 0:
      self.sync_Q_target()
      
    if self.curr_step % self.save_every == 0:
      self.save()
      
    if self.curr_step < self.burnin:
      return None, None
    
    if self.curr_step % self.learn_every != 0:
      return None, None
    
    # Check if we have enough experiences
    if len(self.memory) < self.batch_size:
      return None, None
    
    # Sample from memory
    state, next_state, action, reward, done = self.recall()
    
    # Get TD Estimate
    td_estimate = self.td_estimate(state, action)
    
    # Get TD Target
    td_target = self.td_target(next_state, reward, done)
    
    # Backpropagate loss through Q_online
    loss = self.update_Q_online(td_estimate, td_target)
    
    return (td_estimate.mean().item(), loss)
    