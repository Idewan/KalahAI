import math
import copy
import logging as log
import numpy as np
import sys

from python_agent.kalah_train import Kalah
from python_agent.board import Board
from python_agent.side import Side

log.basicConfig(level=log.DEBUG, filename='./logs/MCTS_new_debug.log', format='%(asctime)s %(message)s')
# sys.setrecursionlimit(10000)

EPSILON = 1e-8

class MCTS():

    def __init__(self, game, cpuct, no_mcts):
        self.game = game

        # to tweak to get the best exploration-exploitation tradeoff
        self.cpuct = cpuct
        self.no_mcts = no_mcts

        # the Q-values for (state, action)
        self.Q = {}
        # the number of times the edge (state, action) was visited
        self.N_sa = {}
        # the number of times the state has been visited
        self.N = {}
        # visited states
        self.visited = set()

        # stores the score for the end games
        self.end_states = {}
        # stores the valid moves
        self.legal_actions = {}


    def getProbs(self, tau=1):

        state = self.game.board

        for i in range(self.no_mcts):
            # print(f'MCTS NUMBER {i}')
            game_copy = copy.deepcopy(self.game)
            self.search(game_copy)

        state_string_p = state.toString()
        counts = [self.N_sa[(state_string_p, action)] if (state_string_p, action) in self.N_sa else 0 for action in range(self.game.actionspace_size)]

        legal_actions = np.array(self.game.getLegalMoves())
        
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


    def search(self, game):
        
        # player and board at the root of the tree
        player = game.turn
        state = game.board
        state_string = state.toString()
        gt = game.no_turns

        if state_string not in self.end_states:
            self.end_states[state_string] = game.getGameOver(game.prev_player)

        # if the game has ended (i.e. value associated with the state is non-zero) return the reward, we cannot expand any more
        if self.end_states[state_string] != 0:
            # log.debug(f"WINNER {game.prev_player}")
            # log.debug(f"SWAP? {game.swap_occured}")
            # log.debug(f"SCORE = {self.end_states[state_string]}\n")
            return self.end_states[state_string] 

        # the state is a leaf node
        # here if we haven't visited the state in the view of a specific player we should also consider that we need 
        # to visit it. i.e. this has not been visited yet.
        if state_string not in self.visited:

            self.visited.add(state_string)
            the_copy = copy.deepcopy(game)
            value = self.play_out(the_copy)

            self.N[state_string] = 0

            # log.debug('EXPAND')
            # log.debug(f'TURN: {game.turn}')
            # log.debug(f'PREV: {game.prev_player}')
            # log.debug(f'VALUE: {value}\n')  # this is the value i get by playing the action

            if gt == 2 and game.swap_occured:
                return -value
            else:
                # was return value if game.prev_player == game.player1 else -value
                return value if game.prev_player == game.turn else -value

        # legal_actions is a list of 0's if illegal and 1's if legal
        legal_actions = game.getLegalMoves()
        current_best = -float('inf')
        best_action = -5

        # UCB
        for action in range(game.actionspace_size):
            if legal_actions[action]:
                if (state_string, action) in self.Q:
                    u = self.Q[(state_string, action)] + self.cpuct * math.sqrt(self.N[state_string]) / (1 + self.N_sa[(state_string, action)])
                else:
                    u = self.cpuct * math.sqrt(self.N[state_string] + EPSILON)

                if u > current_best:
                    current_best = u
                    best_action = action

        if best_action == 0:
            action = -1
        else:
            action = best_action
        
        next_state, _, _ = game.makeMove(action)
        next_player = game.turn
        prev_p = game.prev_player
        game.prev_player = player

        value = self.search(game)
        
        # log.debug('OUT')
        # log.debug(f'TURN: {player}')
        # log.debug(f'PREV: {prev_p}')
        # log.debug(f'SWAP OCCURED:')
        # log.debug(f'VALUE: {value}\n')  # this is the value i get by playing the action
        
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


    def play_out(self, game):
        starting_turn = game.turn
        num_turn = game.no_turns
        done = game.getGameOver(starting_turn) != 0

        while not done:
            legal_actions = game.getLegalActionState()
            action = np.random.choice(legal_actions)

            _,_,done = game.makeMove(action)
        
        if num_turn <= 1 and game.swap_occured:
            return -game.getGameOver(starting_turn)
        else:
            return game.getGameOver(starting_turn)