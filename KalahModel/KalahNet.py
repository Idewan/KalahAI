import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class KalahNet(nn.Module):
    def __init__(self, holes, dropout):
        super(KalahNet, self).__init__()
        action_n = holes + 1        #Counts
        holes_n = (holes + 1) * 2

        #Layers - fully connected layers
        self.fc1 = nn.Linear(holes_n, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)

        #Probability Distribution
        self.fc5 = nn.Linear(512, action_n)

        #Value
        self.fc6 = nn.Linear(512, 1)

        self.fc_bn_1 = nn.BatchNorm1d(512)
        self.fc_bn_2 = nn.BatchNorm1d(512)
        self.fc_bn_3 = nn.BatchNorm1d(512)
        self.fc_bn_4 = nn.BatchNorm1d(512)

    def forward(self, s):
        s = F.dropout(F.relu(self.fc_bn_1(self.fc1(s))), p=dropout, training=True)
        s = F.dropout(F.relu(self.fc_bn_2(self.fc2(s))), p=dropout, training=True)
        s = F.dropout(F.relu(self.fc_bn_3(self.fc3(s))), p=dropout, training=True)
        s = F.dropout(F.relu(self.fc_bn_4(self.fc4(s))), p=dropout, training=True)

        pi = self.fc5(s)
        v = self.fc6(s)
        
        return F.log_softmax(pi, dim=1), torch.tanh(v)