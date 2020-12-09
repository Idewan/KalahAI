import math
import logging
import numpy as np

from python_agent.kalah_train import Kalah
from python_agent.board import Board
from python_agent.side import Side

EPSILON = 1e-8

class MCTS():

    def __init__(self, game, net, args):
        self.game = game
        self.net = net
        # args contain the hyperparameter cpuct and the number of MCTS iterations
        self.args = args

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


    def getProbs(self, state, tau=1):
        for i in range(self.args.no_mcts):
            self.search(state)

        counts = [self.N_sa[(state, action)] if (state, action) in self.N_sa else 0 for action in range(self.game.actionspace_size)]

        if tau == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            return probs

        counts_sum = float(sum(counts))
        probs = [x / counts for x in counts]
        return probs


    def search(self, game, net):
        
        # player and board at the root of the tree
        player = self.game.turn
        state = self.game.board

        # create an independent copy of the game, so that when i run the simulation they will not change the original game
        game_copy = copy.deepcopy(game)

        if state not in self.end_states:
            self.end_states[state] = self.game.getGameOver(player)
        if self.end_states[state] != 0:
            return self.end_states[state]

        # the state is a leaf node (or has not been expanded yet)
        if state not in self.P:
            # TODO fix board (call board_view_player?)
            self.P[state], value = self.net.predict(state)  # this gives the policy vector and the value for the current player
            legal_actions = self.game.getLegalMoves()
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
            return value

        # legal_actions is a list of 0's if illegal and 1's if legal
        legal_actions = self.legal_actions[state]
        current_best = -float('inf')
        # cannot use -1 because that is a possible action
        best_action = -5

        for action in range(self.game.actionspace_size):
            if legal_actions[action]:
                if (state, action) in self.Q:
                    u = self.Q[(state, action)] + self.args.cpuct * self.P[state][action] * math.sqrt(self.N[state]) / (1 + self.next_state[(state, action)])
                else:
                    u = self.args.cpuct * self.P[state][action] * math.sqrt(self.N[state] + EPSILON)

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

        if next_player == player:
            value = self.search(next_state)
        else:
            value = -self.search(next_state)

        if (state, action) in self.Q:
            self.Q[(state, action)] = (self.N_sa[(state, action)] * self.Q[(state, action)] + value) / (self.N_sa[(state, action)] + 1)
            self.N_sa[(state, action)] += 1
        else:
            self.Q[(state, action)] = value
            self.N_sa[(state, action)] = 1
        
        self.N[state] += 1
        return value