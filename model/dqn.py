import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module):
    def __init__(self, input_size, outputs):
        super(DQN, self).__init__()
        #Number of layers
        self.dense1 = nn.Linear(input_size, input_size)
        self.dense2 = nn.Linear(input_size, input_size)
        self.dense3 = nn.Linear(input_size, input_size)
        self.dense4 = nn.Linear(input_size, input_size)
        self.dense5 = nn.Linear(input_size, input_size)

        self.head = nn.Linear(input_size, outputs)
    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        x = F.relu(self.dense5(x))
        return self.head(x)
    
    def load(self, weights):
        self.load_state_dict(weights)

    def save(self, name):
        torch.save(self.state_dict(), name)
    