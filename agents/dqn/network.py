import torch
from torch import nn

class PongQNetwork(nn.Module):
    def __init__(self, input_dim=10, output_dim=3):
        super().__init__()
        
        self.online = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        self.target = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # Initialize target with online weights
        self.target.load_state_dict(self.online.state_dict())
        
        # Freeze target network
        for p in self.target.parameters():
            p.requires_grad = False
    
    def forward(self, x, model="online"):
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)