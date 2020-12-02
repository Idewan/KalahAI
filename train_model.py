from __future__ import print_function

#Custom imports
from model import dqn
from model import replay_memory
from python_agent.kalah_train import Kalah
from python_agent.board import Board
from python_agent.side import Side

import matplotlib
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim


board = Board(7,7)
env = Kalah(board)

loop 
env_sim = Kalah(curr_board)
break when turn == curr_player
board_sim 
is_ipython = 'iniline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

device = torch.device('cpu')
scores = []

#ENVIRONMENT SPECIFC
n_actions=env.actionspace_size
policy_net = dqn.DQN(16,8).to(device)             #Board has 16 inputs
target_net = dqn.DQN(16,8).to(device)             #and 8 actions (SWAP, 7 holes)
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

#Player's side
#Let's say NORTH - 0 side SOUTH - 1
def board_view_player(board, curr_turn):
    if curr_turn == Side.SOUTH:
        board[[0,1]] = board[[1,0]]
    return torch(board.flatten())


# ## Training ##
steps_done = 0

for episode in range(0,10000):
    env.reset()
    curr_turn = old_turn = env.turn
    state = torch(board.flatten())
    while(True):
        #Make move in environment
        action = select_action(state, EPSILON)
        next_state, reward, done = env.makeMove(action)  #State is 2D here

        #^Next state is always in the view of north player 
        #Next state should be in the view of the current player for the memory
        next_state_curr = next_state.copy()
        next_state_curr = board_view_player(next_state_curr, curr_turn)
        
        #Store in memory
        memory.push(state, action, next_state_curr, reward)
        
        old_turn = curr_turn
        curr_turn = env.turn
        
        while(True and env.turn == env.player2):

        next_state = #Blah

        #If the turn is the same i.e. old_turn == curr_turn
        #then we do not swap the board we just change state -> next state
        if old_turn != curr_turn:
            state = board_view_player(next_state, curr_turn)
        else:
            state = next_state_curr.copy()

        #Save target network
        optimize_model()
        if done:
            scores.append([env.score_player1, env.score_player2])
            plot_scores()

    if episode % TARGET_UPDATE:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plt.ioff()
plt.show()

torch.save(target_net.state_dict(), 'tutorial_mode.pth')