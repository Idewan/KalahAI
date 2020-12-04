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
import torch.nn.functional as F


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
TARGET_UPDATE=100    #TARGET value for update

memory_size = 1000000
optimizer = optim.Adam(policy_net.parameters())
memory = replay_memory.ReplayMemory(memory_size)

def plot_scores():
    plt.figure(2)
    plt.clf
    plt.title("Training")
    plt.xlabel("Episode")
    plt.ylabel("Score Difference")
    plt.plot(scores)

    plt.pause(0.001)

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    #We want a batch
    if len(memory) < BATCH_SIZE:
        return

    #Need a nice way to sequence the BATCH
    transitions = memory.sample(BATCH_SIZE)
    batch = memory.Transition(*zip(*transitions))

    #Sequence batch into reward, state and action
    state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)
    action_batch = torch.cat(batch.action)

    state_batch = torch.reshape(state_batch, (BATCH_SIZE, 16))

    #Compute the state-action pair Q(s',a)
    state_action_values = policy_net(state_batch).gather(1, action)
    
    #Changed
    #V(s_t+1)
    #We use the next state as "expected" return
    #hence we need to take the max return of the next_state 
    #ensuring that we do not pass any None next_states
    mask = torch.tensor(tuple(map(lambda l: l is not None,
                                            batch.next_state)), device = device,
                                            dtype=torch.bool)
    non_final_states = torch.cat([s for s in batch.next_state if s is not None])
    non_final_states = torch.reshape(non_final_states, (-1, 16))

    expected_state_values = torch.zeros(BATCH_SIZE, device=device)
    expected_state_values= expected_state_values.double()
    expected_state_values[mask] = target_net(non_final_states).max(1)[0].detach()

    #Compute expected Q value
    expected_action_values = (GAMMA * expected_state_values) + reward_batch

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

def select_action_fixed(state, epsilon):
    probability = random.random()
    if probability > epsilon:
        #use our policies from target NN
        with torch.no_grad():
            return env.actionspace[torch.argmax(target_net(state))]
    else:
        return env.actionspace[random.randrange(n_actions)]

#Player's side
#Let's say NORTH - 0 side SOUTH - 1
def board_view_player(board, curr_turn):
    if curr_turn == Side.SOUTH:
        board[[0,1]] = board[[1,0]]
    board = torch.from_numpy(board.flatten())
    return board


def compete(env, n):

    player1_score = {'win': 0, 'draw': 0, 'loss': 0}
    player2_score = {'win': 0, 'draw': 0, 'loss': 0}

    for episode in range(n):
        env.reset()
        state = env.board.board.copy()
        done = False
        while(not done):
            # Make move in environment
            state = board_view_player(state, env.player1)
            action = select_action(state, 0)
            # print("Action Player 1 {}".format(action))
            next_state, _, done = env.makeMove(action)  #State is 2D here
            next_state = next_state if next_state is None else next_state.copy()

            while(env.turn == env.player2 and not done):
                #Simulate the turn(s) for the second player
                next_state = board_view_player(next_state, env.player2)
                # print(next_state)
                action_p2 = select_action_fixed(next_state, 0)
                # print("Action Player 2 {}".format(action_p2))
                next_state_p2, _, done = env.makeMove(action_p2)

                next_state = None if next_state_p2 is None else next_state_p2.copy()
            # print(env.reward)

            state = next_state
        # print(env.reward)
        if env.reward == 1:
            # print("wassap")
            player1_score['win'] += 1
            player2_score['loss'] += 1
        elif env.reward == 0:
            # print("There")
            player1_score['draw'] += 1
            player2_score['draw'] += 1
        elif env.reward == -1:
            # print("Here")
            player1_score['loss'] += 1
            player2_score['win'] += 1
    print(player1_score, player2_score)
    return player1_score['win'] / (n)

# ## Training ##
for episode in range(100000):
    env.reset()
    curr_turn = env.turn
    state= env.board.board.copy()
    done = False

    while(not done):
        #Make move in environment
        state_curr = state.copy()
        state_curr = board_view_player(state_curr, env.player1)
        action = select_action(state_curr, EPSILON)
        next_state, reward, done = env.makeMove(action)  #State is 2D here
        reward = torch.tensor([reward])

        #Transition action to proper format
        action_full = np.zeros((1,8), dtype=int)
        action_full[0][7 if action == -1 else action - 1] = 1
        action = torch.from_numpy(action_full)

        next_state = next_state if next_state is None else next_state.copy()
        
        #Simulation of the second player's turn.
        while(env.turn == env.player2 and not done):
            #Simulate the turn(s) for the second player
            next_state = board_view_player(next_state, env.player2)
            action_p2 = select_action_fixed(next_state, EPSILON)
            next_state_p2, _, done = env.makeMove(action_p2)

            next_state = None if next_state_p2 is None else next_state_p2.copy()

        #^Next state is always in the view of north player 
        #Next state should be in the view of the current player for the memory
        if next_state is not None:
            next_state_curr = next_state.copy() 
            next_state_curr = board_view_player(next_state_curr, curr_turn)
        else:
            next_state_curr = None

        #Store in memory
        memory.push(state_curr, action, next_state_curr, reward)

        state = next_state.copy() if next_state is not None else next_state

        #Save target network
        optimize_model()
        if done:
            scores.append(env.score_player1 - env.score_player2)
            plot_scores()

    if episode % TARGET_UPDATE == 0:
        win_percentage = compete(env, 100) 
        print(win_percentage)
        if win_percentage > 0.55:
            target_net.load_state_dict(policy_net.state_dict())

print('Complete')
#plt.ioff()


# torch.save(policy_net.state_dict(), 'policy_net.pth')
# torch.save(target_net.state_dict(), 'target_net.pth')