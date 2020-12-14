import math
import copy
import logging as log
import numpy as np
import sys

from python_agent.kalah_train import Kalah
from python_agent.board import Board
from python_agent.side import Side

# sys.setrecursionlimit(10000)

EPSILON = 1e-8

class MCTS():

    def __init__(self, game, net):
        self.game = game
        self.net = net
        self.cpuct = 1
        self.no_mcts = 50

        # the Q-values for (steate, action)
        self.Q = {}
        # the number of times the edge (state, action) was visited
        self.N_sa = {}
        # the number of times the state has been visited
        self.N = {}
        # initial policy returned by the net
        self.P = {}

        # stores the score for the end games
        self.end_states = {}
        # stores the valid moves
        self.legal_actions = {}


    def getProbs(self, tau=1):
        
        state = self.game.board

        for i in range(self.no_mcts):
            # print(f'MCTS NUMBER {i}')
            game_copy = copy.deepcopy(self.game)
            self.search(game_copy, self.net)

        state_string_p = state.toString()
        counts = [self.N_sa[(state_string_p, action)] if (state_string_p, action) in self.N_sa else 0 for action in range(self.game.actionspace_size)]
        # print(tau)
        
        if tau == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            return probs
        
        counts = [x ** (1. / tau) for x in counts]
        counts_sum = float(sum(counts))
        legal_actions = np.array(self.game.getLegalMoves())
        probs = [x / counts_sum for x in counts] * legal_actions
        probs /= np.sum(probs)

        return probs


    def search(self, game, net):
        
        # player and board at the root of the tree
        player = game.turn
        state = game.board
        state_string = state.toString()
        gt = game.no_turns

        if state_string not in self.end_states:
            self.end_states[state_string] = game.getGameOver(game.prev_player)
        # if the game has ended (i.e. value associated with the state is non-zero) return the reward, we cannot expand any more
        if self.end_states[state_string] != 0:
            # print('END STATES')
            # print(f'TURN: {game.turn}')
            # print(f'PREV: {game.prev_player}')
            # print(f'VALUE: {self.end_states[state_string]}\n')  # this is the value i get by playing the action
            return self.end_states[state_string] 

        # the state is a leaf node
        # here if we haven't visited the state in the view of a specific player we should also consider that we need 
        # to visit it. i.e. this has not been visited yet.
        if (state_string, game.turn) not in self.P:

            state_np = self.net.board_view_player(state, player)

            # this gives the policy vector and the value for the current player
            self.P[(state_string, game.turn)], value = self.net.predict(state_np)
            legal_actions = game.getLegalMoves()
            # masking out invalid actions
            self.P[(state_string, game.turn)] = self.P[(state_string, game.turn)] * legal_actions

            sum_P_s = np.sum(self.P[(state_string, game.turn)])

            if sum_P_s > 0:
                self.P[(state_string, game.turn)] /= sum_P_s  # re-normalize
            else:
                print("All valid moves were masked, doing a workaround.")
                self.P[(state_string, game.turn)] = self.P[(state_string, game.turn)] + legal_actions
                self.P[(state_string, game.turn)] /= np.sum(self.P[(state_string, game.turn)])

            self.legal_actions[(state_string, game.turn)] = legal_actions
            self.N[state_string] = 0

            # print('EXPAND')
            # print(f'TURN: {game.turn}')
            # print(f'PREV: {game.prev_player}')
            # print(f'VALUE: {value}\n')  # this is the value i get by playing the action

            if gt == 2 and game.swap_occured:
                return -value
            else:
                # was return value if game.prev_player == game.player1 else -value
                return value if game.prev_player == game.turn else -value

        # legal_actions is a list of 0's if illegal and 1's if legal
        legal_actions = self.legal_actions[(state_string, game.turn)]
        current_best = -float('inf')
        best_action = -5

        # UCB
        for action in range(game.actionspace_size):
            if legal_actions[action]:
                if (state_string, action) in self.Q:
                    u = self.Q[(state_string, action)] + self.cpuct * self.P[(state_string, game.turn)][action] * math.sqrt(self.N[state_string]) / (1 + self.N_sa[(state_string, action)])
                else:
                    u = self.cpuct * self.P[(state_string, game.turn)][action] * math.sqrt(self.N[state_string] + EPSILON)

                if u > current_best:
                    current_best = u
                    best_action = action

        if best_action == 0:
            action = -1
        else:
            action = best_action

        if action == -1:
            self.legal_actions[(state_string, game.turn)][0] = 0 
        
        next_state, _, _ = game.makeMove(action)
        next_player = game.turn
        prev_p = game.prev_player
        game.prev_player = player

        value = self.search(game, self.net)

        # print('OUT')
        # print(f'TURN: {player}')
        # print(f'PREV: {prev_p}')
        # print(f'VALUE: {value}\n')  # this is the value i get by playing the action

        # this is to test the swap values
        # if action == -1:
        #     exit()
        
        if (state_string, action) in self.Q:
            self.Q[(state_string, action)] = (self.N_sa[(state_string, action)] * self.Q[(state_string, action)] + value) / (self.N_sa[(state_string, action)] + 1)
            self.N_sa[(state_string, action)] += 1
        else:
            self.Q[(state_string, action)] = value
            self.N_sa[(state_string, action)] = 1
        
        self.N[state_string] += 1

        if gt == 2 and game.swap_occured:
            return -value
        else:
            return value if player == prev_p else -value