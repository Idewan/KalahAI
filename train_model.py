from __future__ import print_function

#Custom imports
from model import dqn
from model import replay_memory
from python_agent.kalah_train import Kalah
from python_agent.board import Board
from python_agent.side import Side

import matplotlib
import random
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim


board = Board(7,7)
env = Kalah(board)

is_ipython = 'iniline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scores = []

#ENVIRONMENT SPECIFC
n_actions=env.actionspace_size
policy_net = dqn.DQN(16,8).to(device).double()             #Board has 16 inputs
target_net = dqn.DQN(16,8).to(device).double()         #and 8 actions (SWAP, 7 holes)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

BATCH_SIZE = 128
GAMMA = 0.999
EPSILON = 0.25      #EPSILON start value
TARGET_UPDATE=20    #TARGET value for update

memory_size = 1000000
optimizer = optim.Adam(policy_net.parameters())
memory = replay_memory.ReplayMemory(memory_size)

def plot_scores():
    plt.figure(2)
    plt.clf
    plt.title("Training")
    plt.xlabel("Episode")
    plt.ylabel("Score Difference")
    plt.plot(scores.numpy())

    plt.pause(0.001)

    if is_python:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model(memory, optimizer, batch_size, gamma):
    #We want a batch
    if len(memory) < batch_size:
        return

    #Need a nice way to sequence the BATCH
    batch = memory.Transition(*zip(*transitions))

    #Sequence batch into reward, state and action
    state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)
    action_batch = torch.cat(batch.action)

    #Compute the state-action pair Q(s',a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    #Changed
    #V(s_t+1)
    #We use the next state as "expected" return
    #hence we need to take the max return of the next_state 
    #ensuring that we do not pass any None next_states
    mask = torch.where(batch.next_state is not [None])
    non_final_states = torch.cat([s for s in batch.next_state if s is not None])

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
    #To select whether we explore or exploit, we will compute 
    #a random value in the range [0,1.0] and if our epsilon
    #is smaller, then we take a random action.
    probability = random.random()
    if probability > epsilon:
        #use our policies from target NN
        with torch.no_grad():
            return env.actionspace[torch.argmax(policy_net(state))]
    else:
        return env.actionspace[random.randrange(n_actions)]

#Player's side
#Let's say NORTH - 0 side SOUTH - 1
def board_view_player(board, curr_turn):
    if curr_turn == Side.SOUTH:
        board[[0,1]] = board[[1,0]]
    print(board)
    board = torch.from_numpy(board.flatten())
    return board


def compete(env):

    player1_score = {'win': 0, 'draw': 0, 'loss': 0}
    player2_score = {'win': 0, 'draw': 0, 'loss': 0}

    for episode in range(0, 10):
        env.reset()
        state = env.board.board.copy()
        done = False
        while(not done):
            # Make move in environment
            state = board_view_player(state, env.player1)
            action = select_action(state, 0)
            next_state, _, done = env.makeMove(action)  #State is 2D here

            while(env.turn == env.player2 and not done):
                #Simulate the turn(s) for the second player
                next_state = board_view_player(next_state, env.player2)
                action_p2 = select_action(next_state, EPSILON)
                next_state_p2, _, done = env.makeMove(action_p2)

                next_state = next_state_p2

            state = next_state

        if env.reward == 1:
            player1_score['win'] += 1
            player2_score['loss'] += 1
        elif env.reward == 0:
            player1_score['draw'] += 1
            player2_score['draw'] += 1
        elif env.reward == -1:
            player1_score['loss'] += 1
            player2_score['win'] += 1

    return player1_score['win'] / (10)

# ## Training ##
# for episode in range(0,10000):
#     env.reset()
#     curr_turn = env.turn
#     state = torch(env.board.board.flatten())
#     done = False

#     while(not done):
#         #Make move in environment
#         state_curr = state.copy()
#         state_curr = board_view_player(state, env.player1)
#         action = select_action(state, EPSILON)
#         next_state, reward, done = env.makeMove(action)  #State is 2D here
        
#         #Simulation of the second player's turn.
#         while(env.turn == env.player2 and not done):
#             #Simulate the turn(s) for the second player
#             next_state = board_view_player(next_state, env.player2)
#             action_p2 = select_action(next_state, EPSILON)
#             next_state_p2, _, done = env.makeMove(action_p2)

#             next_state = next_state_p2

#         #^Next state is always in the view of north player 
#         #Next state should be in the view of the current player for the memory
#         next_state_curr = next_state.copy()
#         next_state_curr = board_view_player(next_state_curr, curr_turn)

#         #Store in memory
#         memory.push(state_curr, action, next_state_curr, reward)

#         state = next_state.copy()

#         #Save target network
#         optimize_model()
#         if done:
#             scores.append(env.score_player1 - env.score_player2)
#             plot_scores()

#     if episode % TARGET_UPDATE:
#         if compete(env) > 0.55:
#             target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plt.ioff()
plt.show()

# torch.save(target_net.state_dict(), 'tutorial_mode.pth')

print(compete(env))