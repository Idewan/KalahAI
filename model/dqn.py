import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.module):
    def __init__(self, input_size, outputs):
        super(DQN, self).__init__()
        #Number of layers

        self.head = nn.Linear(input_size, outputs)
    
    def forward(self, x):
        x = F.relu(#TODO)
        #TODO
        return self.head(#TODO)
    
    def load(self, name):
        #TODO

    def save(self, name):
        #TODO
    