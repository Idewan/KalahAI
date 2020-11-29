import matplotlib
import torch

import matplotlib as plt
import numpy as np
import torch.optim as optim

def plot_score_difference():
    plt.figure(2)
    plt.clf
    score_difference = #TODO
    plt.title("Training")
    plt.xlabel("Episode")
    plt.ylabel("Score Difference")
    plt.plot(score_difference.numpy())

    #TODO
    plt.pause(0.001)

    if is_python:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    #TODO

def select_action(state):
    #TODO

## Training ##
BATCH_SIZE = #TODO
GAMMA = #TODO
EPS_START=0.9       #EPSILON start value
EPS_END=0.05        #EPSILON end value
EPS_DECAY=200       #EPSILON value decay
TARGET_UPDATE=#TODO #TARGET value for update

#ENVIRONMENT SPECIFC
n_actions=#TODO
policy_net = #TODO
target_net = #TODO
target_net.load_state_dict(policy_net.state_dict())
target_next.eval()

memory_size = 1000000
optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(memory_size)

steps_done = 0

for target_list in expression_list:
    for target_list in experssion_list:
        print(#TODO)

        #Save target network

print('Complete')
plt.ioff()
plt.show()

torch.save(target_net.state_dict(), 'tutorial_mode.pth')