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
        # number of simulations
        self.no_mcts = no_mcts

        # the number of times the edge (state, action) was visited
        self.N_sa = {}
        # the number of times the state has been visited
        self.N = {}
        # visited states
        self.visited = set()
        # heuristic value (state_string, game.turn)
        self.v = {}
        # 
        self.v_sa = {}
        #
        self.r = {}
        #
        self.r_sa = {}
        # alpha value for implicit minimax
        self.alpha = 0.4


    def getProbs(self, tau=1):

        state = self.game.board
        state_string_p = state.toString()

        # run simulation
        for i in range(self.no_mcts):
            game_copy = copy.deepcopy(self.game)
            key = (None, None, False, -2)
            self.simulate(game_copy, key)

        key_short = (state_string_p, self.game.turn, self.game.no_turns == 2 and self.game.swap_occured)

        # count number of visits
        counts = []
        for action in range(self.game.actionspace_size):
            key_long = key_short + (action,)
            if key_long in self.N_sa:
                counts.append(self.N_sa[key_long])
            else:
                counts.append(0)

        log.debug(f'COUNTS: {counts}')
        
        # select best action with probability 1 if playing competitively
        if tau == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            log.debug(f'PROBABILITIES: {probs}')
            return probs
        
        counts = [x ** (1. / tau) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs


    def q_im(self, key_long):
        """
            
        """
        return (1 - self.alpha) * (self.r_sa[key_long] / self.N_sa[(key_long)]) + self.alpha * self.v_sa[(key_long)]

    
    def calculate_uct(self, key_long, key_short):
        """

        """
        return self.q_im(key_long) + self.cpuct * math.sqrt(math.log(self.N[key_short]) / self.N_sa[key_long])


    def select(self, game, key_short):
        """

        """
        legal_actions = game.getLegalMoves()
        best_uct = -float('inf')
        actions = {}

        for action in range(game.actionspace_size):
            if legal_actions[action]:
                action = -1 if action == 0 else action
                key_long = key_short + (action,)
                uct_val = self.calculate_uct(key_long, key_short)
                if uct_val not in actions:
                    actions[uct_val] = [action]
                else:
                    actions[uct_val].append(action)

        best_actions = actions[max(actions)]

        return np.random.choice(best_actions)


    def revert_game(self, game, board, r, p1, p2, t, prev_p, s_p1, s_p2, no, s_o):
        """

        """
        game.board, game.reward = copy.deepcopy(board), r
        game.player1, game.player2, game.turn, game.prev_player = p1, p2, t, prev_p
        game.score_player1, game.score_player2, game.no_turns = s_p1, s_p2, no
        game.swap_occured = s_o

        return game


    def update(self, legal_actions, key_short, reward):
        """
        
        """
        self.N[key_short] += 1
        self.r[key_short] += reward

        for action in legal_actions:
            key_long = key_short + (action,)
            if self.v_sa[key_long] > self.v[key_short]:
                self.v[key_short] = self.v_sa[key_long]


    def simulate(self, game, key_long_parent):
        """

        """
        player = game.turn
        state_string = game.board.toString()
        gt = game.no_turns  # the turn that called search (number of turn)
        just_swapped = gt == 2 and game.swap_occured

        key_short = (state_string, player, just_swapped)

        # get game over 
        r_turn = game.getGameOver(game.prev_player)
        if r_turn != 0:
            return r_turn

        # has not been expanded
        if key_short not in self.visited:

            # EXPAND
            self.visited.add(key_short)
            self.N[key_short] = 0
            self.r[key_short] = 0
            self.v[key_short] = -float('inf')

            legal_actions = game.getLegalActionState()

            for action in legal_actions:
                game_c = copy.deepcopy(game)
                game_c.makeMove(action)

                # get the score difference
                score_me = game_c.board.getSeedsInStore(game_c.turn)
                opp_side = game_c.turn.opposite(game_c.turn)
                score_opp = game_c.board.getSeedsInStore(opp_side)

                # get long key 
                key_long = key_short + (action,)
                self.v_sa[key_long] = score_me - score_opp
                
                if self.v_sa[key_long] > self.v[key_short]:
                    self.v[key_short] = self.v_sa[key_long]
                
                # initialize RSA for current state
                self.r_sa[key_long] = 0
                self.N_sa[key_long] = EPSILON
                
            # PLAYOUT
            the_copy = copy.deepcopy(game)
            reward = self.play_out(the_copy)

            # UPDATE
            self.update(legal_actions, key_short, reward)
               
            # BACKPROPAGATE
            if just_swapped:
                reward = -reward
            else:
                reward = reward if game.prev_player == game.turn else -reward
            
            if None not in key_long_parent:
                self.r_sa[key_long_parent] = reward
                self.N_sa[key_long_parent] = 1
            
            return reward

        # SELECT
        action = self.select(game, key_short)

        legal_actions = game.getLegalActionState()

        # play action
        game.makeMove(action)
        prev_p = game.prev_player
        game.prev_player = player

        key_long_curr = key_short + (action,)

        # recursive call to go deeper in the tree
        reward = self.simulate(game, key_long_curr)

        self.v_sa[key_long_curr] = 0

        # UPDATE
        self.update(legal_actions, key_short, reward)

        if gt == 2 and game.swap_occured:
            reward = -reward
        else:
            reward = reward if player == prev_p else -reward

        self.r_sa[key_long_curr] += reward
        self.N_sa[key_long_curr] += 1

        return reward


    def play_out(self, game):
        """

        """
        starting_turn = game.turn
        num_turn = game.no_turns
        done = game.getGameOver(starting_turn) != 0

        while not done:
            legal_actions = game.getLegalActionState()
            action = np.random.choice(legal_actions)

            _, _, done = game.makeMove(action)
        
        if num_turn <= 1 and game.swap_occured:
            return -game.getGameOver(starting_turn)
        else:
            return game.getGameOver(starting_turn)