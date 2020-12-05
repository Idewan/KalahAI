import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module):
    def __init__(self, input_size, outputs):
        super(DQN, self).__init__()
        #Number of layers

        self.nn = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, outputs)
        )
    
    def forward(self, x):
        return self.nn(x)
    
    def load(self, weights):
        self.load_state_dict(weights)

    def save(self, name):
        torch.save(self.state_dict(), name)
    