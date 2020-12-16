import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class KalahNet(nn.Module):
    def __init__(self, holes, dropout):
        super(KalahNet, self).__init__()
        action_n = holes + 1
        holes_n = (holes + 1) * 2

        #Layers - fully connected layers
        self.fc1 = nn.Linear(holes_n, 28)
        self.fc2 = nn.Linear(28, 28)

        self.fc3 = nn.Linear(28, action_n)
        
        self.fc4 = nn.Linear(28, 1)
    
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    # def __init__(self, holes, dropout):
    #     self.dropout = dropout
    #     super(KalahNet, self).__init__()
    #     action_n = holes + 1        #Counts
    #     holes_n = (holes + 1) * 2

    #     #Layers - fully connected layers
    #     self.fc1 = nn.Linear(holes_n, 16) #Try one hidden layer with 28 nodes.
    #     self.fc2 = nn.Linear(16, 16)
    #     self.fc3 = nn.Linear(16, 16)
    #     self.fc4 = nn.Linear(16, 16)

    #     #Probability Distribution
    #     self.fc5 = nn.Linear(16, action_n)

    #     #Value
    #     self.fc6 = nn.Linear(16, 1)

    #     self.fc_bn_1 = nn.BatchNorm1d(16)
    #     self.fc_bn_2 = nn.BatchNorm1d(16)
    #     self.fc_bn_3 = nn.BatchNorm1d(16)
    #     self.fc_bn_4 = nn.BatchNorm1d(16)

    # def forward(self, s):
    #     s = F.dropout(F.relu(self.fc_bn_1(self.fc1(s))), p=self.dropout, training=self.training)
    #     s = F.dropout(F.relu(self.fc_bn_2(self.fc2(s))), p=self.dropout, training=self.training)
    #     s = F.dropout(F.relu(self.fc_bn_3(self.fc3(s))), p=self.dropout, training=self.training)
    #     s = F.dropout(F.relu(self.fc_bn_4(self.fc4(s))), p=self.dropout, training=self.training)

    #     pi = self.fc5(s)
    #     v = self.fc6(s)
        
    #     return F.log_softmax(pi, dim=1), torch.tanh(v)

