import math
import copy
import logging as log
import numpy as np
import sys

from python_agent.kalah_train import Kalah
from python_agent.board import Board
from python_agent.side import Side

log.basicConfig(level=log.DEBUG, filename='./logs/MCTS_debug.log', format='%(asctime)s %(message)s')
# sys.setrecursionlimit(10000)

EPSILON = 1e-8

class MCTS():

    def __init__(self, game, net, cpuct, no_mcts):
        self.game = game
        self.net = net

        # to tweak to get the best exploration-exploitation tradeoff
        self.cpuct = cpuct
        # number of simulations
        self.no_mcts = no_mcts

        # the Q-values for (state, action)
        self.Q = {}
        # the number of times the edge (state, action) was visited
        self.N_sa = {}
        # the number of times the state has been visited
        self.N = {}
        # initial policy returned by the net
        self.P = {}

        # stores the score for the end games
        self.end_states = {}

    def getProbs(self, tau=1):

        state = self.game.board
        state_string_p = state.toString()

        for i in range(self.no_mcts):
            game_copy = copy.deepcopy(self.game)
            self.search(game_copy, self.net)

        # count number of visits
        counts = []
        for action in range(self.game.actionspace_size):
            if (state_string_p, action) in self.N_sa:
                counts.append(self.N_sa[(state_string_p, action)])
            else:
                counts.append(0)
        
        # select best action with probability 1 if playing competitively
        if tau == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            return probs
        
        legal_actions = np.array(self.game.getLegalMoves())
        counts = [x ** (1. / tau) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]

        np_probs = np.array(probs)
        return np_probs.copy()


    def search(self, game, net):

        """
        1. Select
        2. Expand
        3. Simulate
        4. Backpropagate
        """
        
        # player and board at the root of the tree
        player = game.turn
        state = game.board
        # string representation of state
        state_string = state.toString()
        # the turn that called search (number of turn)
        gt = game.no_turns

        if state_string not in self.end_states:
            self.end_states[state_string] = game.getGameOver(game.prev_player)
        # if the game has ended (i.e. value associated with the state is non-zero) return the reward, we cannot expand any more
        if self.end_states[state_string] != 0:
            return self.end_states[state_string] 

        # has not been expanded
        if (state_string, game.turn) not in self.P:

            state_np = self.net.board_view_player(state, player)

            # EXPAND and "SIMULATE"
            # this gives the policy vector and the value for the current player
            self.P[(state_string, game.turn)], value = self.net.predict(state_np)
            value = value[0]

            # masking out invalid actions
            legal_actions = game.getLegalMoves()
            self.P[(state_string, game.turn)] = self.P[(state_string, game.turn)] * legal_actions

            sum_P_s = np.sum(self.P[(state_string, game.turn)])

            if sum_P_s > 0:
                self.P[(state_string, game.turn)] /= sum_P_s  # re-normalize
            else:
                log.debug("All valid moves were masked, doing a workaround.")
                self.P[(state_string, game.turn)] = self.P[(state_string, game.turn)] + legal_actions
                self.P[(state_string, game.turn)] /= np.sum(self.P[(state_string, game.turn)])

            self.N[state_string] = 0

            # BACKPROPAGATE
            if gt == 2 and game.swap_occured:
                return -value
            else:
                # was return value if game.prev_player == game.player1 else -value
                return value if game.prev_player == game.turn else -value

        action = self.select_action(game, state_string)
        
        next_state, _, _ = game.makeMove(action)
        next_player = game.turn
        prev_p = game.prev_player
        game.prev_player = player

        # recursive call to go deeper in the tree
        value = self.search(game, self.net)

        # update values
        self.update_values(state_string, action, value)

        if gt == 2 and game.swap_occured:
            return -value
        else:
            return value if player == prev_p else -value


    def select_action(self, game, state_string):
        # legal_actions is a list of 0's if illegal and 1's if legal
        legal_actions = game.getLegalMoves()
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
        return action

    def update_values(self, state_string, action, value):
        if (state_string, action) in self.Q:
            self.Q[(state_string, action)] = (self.N_sa[(state_string, action)] * self.Q[(state_string, action)] + value) / (self.N_sa[(state_string, action)] + 1)
            self.N_sa[(state_string, action)] += 1
        else:
            self.Q[(state_string, action)] = value
            self.N_sa[(state_string, action)] = 1
        
        self.N[state_string] += 1