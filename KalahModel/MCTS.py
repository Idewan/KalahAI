import math
import copy
import logging as log
import numpy as np
import sys

from python_agent.kalah_train import Kalah
from python_agent.board import Board
from python_agent.side import Side

sys.setrecursionlimit(10000)

EPSILON = 1e-8

class MCTS():

    def __init__(self, game, net):
        self.game = game
        self.net = net
        self.cpuct = 1
        self.no_mcts = 25

        # the Q-values for (steate, action)
        self.Q = {}
        # # the number of times the edge (state, action) was visited
        self.N_sa = {}
        # the number of times the state has been visited
        self.N = {}
        # initial policy returned by the net
        self.P = {}

        # # stores the score for the end games
        self.end_states = {}
        # # stores the valid moves
        self.legal_actions = {}


    def getProbs(self, tau=1):
        state = self.game.board

        print(f'START GET PROBS')

        for i in range(self.no_mcts):
            game_copy = copy.deepcopy(self.game)
            print(f'RUN NUMBER {i}')
            self.search(game_copy, self.net)
        state_string_p = state.toString()
        counts = [self.N_sa[(state_string_p, action)] if (state_string_p, action) in self.N_sa else 0 for action in range(self.game.actionspace_size)]
  
        if tau == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            return probs

        counts = [x ** (1. / tau) for x in counts]
        counts_sum = float(sum(counts))
        legal_actions = np.array(self.game.getLegalMoves())
        # print(f'legal {legal_actions}')
        probs = [x / counts_sum for x in counts] * legal_actions
        probs /= np.sum(probs)

        return probs


    def search(self, game, net):

        # print("***** Start MCTS *****")
        
        # player and board at the root of the tree
        player = game.turn
        state = game.board
        state_string = state.toString()

        # print(f'Player {player}')
        # print(f'State {state}')

        if state_string not in self.end_states:
            self.end_states[state_string] = game.getGameOver(game.prev_player)
        # if the game has ended (i.e. value associated with the state is non-zero) return the reward, we cannot expand any more
        if self.end_states[state_string] != 0:
            return self.end_states[state_string]

        # the state is a leaf node (or has not been expanded yet)
        if state_string not in self.P:
            print('HERE AGAIN')
            state_np = self.net.board_view_player(state, player)
            # print(f'Board in right orientation {state_np}')
            self.P[state_string], value = self.net.predict(state_np)  # this gives the policy vector and the value for the current player
            legal_actions = game.getLegalMoves()
            # masking out invalid actions
            self.P[state_string] = self.P[state_string] * legal_actions

            sum_P_s = np.sum(self.P[state_string])

            if sum_P_s > 0:
                self.P[state_string] /= sum_P_s  # re-normalize
            else:
                print("All valid moves were masked, doing a workaround.")
                self.P[state_string] = self.P[state_string] + legal_actions
                self.P[state_string] /= np.sum(self.P[state_string])
            
            self.legal_actions[(state_string, game.turn)] = legal_actions
            self.N[state_string] = 0
            return value if game.prev_player == game.player1 else -value

        # legal_actions is a list of 0's if illegal and 1's if legal
        legal_actions = self.legal_actions[(state_string, game.turn)]
        # print(f'turn {game.turn}')
        # print(f'state {state_string}')
        # print(f'Legal {legal_actions}')
        # print(game.board.board[0][:])
        # print(game.board.board[1][:])
        current_best = -float('inf')
        # cannot use -1 because that is a possible action
        best_action = -5

        for action in range(game.actionspace_size):
            print(f'Legal Action {legal_actions[action]}')
            if legal_actions[action]:
                if (state_string, action) in self.Q:
                    u = self.Q[(state_string, action)] + self.cpuct * self.P[state_string][action] * math.sqrt(self.N[state_string]) / (1 + self.N_sa[(state_string, action)])
                else:
                    u = self.cpuct * self.P[state_string][action] * math.sqrt(self.N[state_string] + EPSILON)

                if u > current_best:
                    current_best = u
                    best_action = action

        if best_action == 0:
            action = -1
        else:
            action = best_action
        print(action)
        # making move on the copy so that the original game does not change
        next_state, _, _ = game.makeMove(action)
        next_player = game.turn
        prev_p = game.prev_player
        game.prev_player = player

        print("HERE")
        value = self.search(game, self.net)

        print(state_string, action)
        
        if (state_string, action) in self.Q:
            self.Q[(state_string, action)] = (self.N_sa[(state_string, action)] * self.Q[(state_string, action)] + value) / (self.N_sa[(state_string, action)] + 1)
            self.N_sa[(state_string, action)] += 1
        else:
            self.Q[(state_string, action)] = value
            self.N_sa[(state_string, action)] = 1
        
        self.N[state_string] += 1
        return state_string if player == prev_p else -value