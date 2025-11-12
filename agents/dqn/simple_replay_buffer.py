import random
import numpy as np
import torch
from collections import deque

class SimpleReplayBuffer:
    def __init__(self, capacity=100000, device="cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def add(self, state, next_state, action, reward, done):
        """Add experience to buffer"""
        experience = (state, next_state, action, reward, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to numpy first, then to tensor
        states = torch.tensor(np.array([exp[0] for exp in batch]), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([exp[1] for exp in batch]), dtype=torch.float32).to(self.device)
        actions = torch.tensor([int(exp[2]) for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([float(exp[3]) for exp in batch], dtype=torch.float32).to(self.device)
        dones = torch.tensor([bool(exp[4]) for exp in batch], dtype=torch.bool).to(self.device)
        
        return states, next_states, actions, rewards, dones
    
    def __len__(self):
        return len(self.buffer)