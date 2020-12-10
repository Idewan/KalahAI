import math
import copy
import logging as log
import numpy as np

from python_agent.kalah_train import Kalah
from python_agent.board import Board
from python_agent.side import Side

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


    def getProbs(self, game, tau=1):
        state = self.game.board

        for i in range(self.no_mcts):
            self.search(self.game, self.net)

        counts = [self.N_sa[(state, action)] if (state, action) in self.N_sa else 0 for action in range(self.game.actionspace_size)]
        # print(counts)
  
        if tau == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            return probs

        counts = [x ** (1. / tau) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs


    def search(self, game, net):

        # print("***** Start MCTS *****")
        
        # player and board at the root of the tree
        player = game.turn
        state = game.board

        # print(f'Player {player}')
        # print(f'State {state}')

        # create an independent copy of the game, so that when i run the simulation they will not change the original game
        game_copy = copy.deepcopy(game)

        if state not in self.end_states:
            self.end_states[state] = game.getGameOver(game.prev_player)
        # if the game has ended (i.e. value associated with the state is non-zero) return the reward, we cannot expand any more
        if self.end_states[state] != 0:
            return self.end_states[state]

        # the state is a leaf node (or has not been expanded yet)
        if state not in self.P:
            state_np = self.net.board_view_player(state, player)
            self.P[state], value = self.net.predict(state_np)  # this gives the policy vector and the value for the current player
            legal_actions = game.getLegalMoves()
            # masking out invalid actions
            self.P[state] = self.P[state] * legal_actions

            sum_P_s = np.sum(self.P[state])

            if sum_P_s > 0:
                self.P[state] /= sum_P_s  # re-normalize
            else:
                log.error("All valid moves were masked, doing a workaround.")
                self.P[state] = self.P[state] + legal_actions
                self.P[state] /= np.sum(self.P[state])
            
            self.legal_actions[state] = legal_actions
            self.N[state] = 0
            return value if game.prev_player == game.player1 else -value

        # legal_actions is a list of 0's if illegal and 1's if legal
        legal_actions = self.legal_actions[state]
        current_best = -float('inf')
        # cannot use -1 because that is a possible action
        best_action = -5

        for action in range(game.actionspace_size):
            if legal_actions[action]:
                if (state, action) in self.Q:
                    u = self.Q[(state, action)] + self.cpuct * self.P[state][action] * math.sqrt(self.N[state]) / (1 + self.N_sa[(state, action)])
                else:
                    u = self.cpuct * self.P[state][action] * math.sqrt(self.N[state] + EPSILON)

                if u > current_best:
                    current_best = u
                    best_action = action

        if best_action == 0:
            action = -1
        else:
            action = best_action
        # making move on the copy so that the original game does not change
        next_state, _, _ = game_copy.makeMove(action)
        next_player = game_copy.turn
        prev_p = game_copy.prev_player
        game_copy.prev_player = player

        value = self.search(game_copy, self.net)
        
        if (state, action) in self.Q:
            self.Q[(state, action)] = (self.N_sa[(state, action)] * self.Q[(state, action)] + value) / (self.N_sa[(state, action)] + 1)
            self.N_sa[(state, action)] += 1
        else:
            self.Q[(state, action)] = value
            self.N_sa[(state, action)] = 1
        
        self.N[state] += 1
        return value if player == prev_p else -value