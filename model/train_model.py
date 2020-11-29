from __future__ import print_function
from itertools import count
import matplotlib
import torch
import math
import matplotlib as plt
import numpy as np
import torch.optim as optim

#Custom imports
import dqn
import replay_memory

#ENVIRONMENT SPECIFC
n_actions=#TODO
policy_net = dqn.DQN(16,8).to(device)             #Board has 16 inputs
target_net = dqn.DQN(16,8).to(device)             #and 8 actions (SWAP, 7 holes)
target_net.load_state_dict(policy_net.state_dict())
target_next.eval()

BATCH_SIZE = #TODO
GAMMA = 
EPSILON = 0.25      #EPSILON start value
TARGET_UPDATE=#TODO #TARGET value for update

memory_size = 1000000
optimizer = optim.Adam(policy_net.parameters())
memory = replay_memory.ReplayMemory(memory_size)

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

def optimize_model(memory, optimizer, batch_size, gamma):
    #We want a batch
    if len(memory) < batch_size:
        return

    #Need a nice way to sequence the BATCH
    batch = Transition(*zip(*transitions))

    #Sequence batch into reward, state and action
    state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)
    action_batch = torch.cat(batch.action)

    #Compute the state-action pair Q(s',a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    #V(s_t+1)
    #We use the next state as "expected" return
    #hence we need to take the max return of the next_state 
    #ensuring that we do not pass any None next_states
    mask = torch.where(batch.next_state is not [None])
    non_final_states = torch.cat([for s in batch.next_state if s is not None])

    expected_state_values = torch.zeros(batch_size, device=device)
    expected_state_values[mask] = target_net(non_final_states).max(1)[0].detach()

    #Compute expected Q value
    expected_action_values = (gamma * expected_action_values) + reward_batch

    #Compute loss
    loss = F.smooth_l1_loss(state_action_values, expected_action_values.unsqueeze(1))

    #Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.clamp_(-1,1)
    optimizer.step()

def select_action(state, epsilon):
    global steps_done
    #To select whether we explore or exploit, we will compute 
    #a random value in the range [0,1.0] and if our epsilon
    #is smaller, then we take a random action.
    probability = random.random()
    steps_done += 1
    if probability > epsilon:
        #use our policies from target NN
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1).numpy()[0][0]
    else:
        return random.randrange(n_actions)
        

## Training ##
steps_done = 0

for target_list in expression_list:
    for target_list in experssion_list:
        #TODO

        #Save target network

print('Complete')
plt.ioff()
plt.show()

torch.save(target_net.state_dict(), 'tutorial_mode.pth')