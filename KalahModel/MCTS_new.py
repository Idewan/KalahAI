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
        
        self.playout_n = 10

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


    def getProbs(self):

        state = self.game.board
        state_string_p = state.toString()

        # run simulation
        for i in range(self.no_mcts):
            # log.debug(f'****************************************************')
            # log.debug(f'****************** SIMULATION {i} ******************')
            # log.debug(f'****************************************************')
            game_copy = copy.deepcopy(self.game)
            key = (None, None, False, -2)
            self.simulate(game_copy, key, game_copy.turn)

        key_short = (state_string_p, self.game.turn, self.game.no_turns == 2 and self.game.swap_occured)

        # count number of visits
        counts = []
        values = []
        for action in range(self.game.actionspace_size):
            if action == 0:
                action = -1 
            key_long = key_short + (action,)
            if key_long in self.N_sa:
                values.append(self.v_sa[key_long])
                counts.append(self.N_sa[key_long])
            else:
                values.append(0)
                counts.append(0)    

        log.debug(f'COUNTS: {counts}')
        log.debug(f'VALUES: {values}')
        return counts


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


    def update(self, legal_actions, key_short, reward, boss_turn, js):
        """
        
        """
        self.N[key_short] += 1
        self.r[key_short] += reward

        for action in legal_actions:
            key_long = key_short + (action,)
            if boss_turn and not js:
                if self.v_sa[key_long] > self.v[key_short]:
                    self.v[key_short] = self.v_sa[key_long]
            else:
                if self.v_sa[key_long] < self.v[key_short]:
                    self.v[key_short] = self.v_sa[key_long]

    def simulate(self, game, key_long_parent, player):
        """

        """
        player_curr = copy.deepcopy(player)
        state_string = game.board.toString()
        gt = game.no_turns  # the turn that called search (number of turn)
        just_swapped = gt == 2 and game.swap_occured

        if just_swapped:
            player_curr = player.opposite(player_curr)

        key_short = (state_string, game.turn, just_swapped)

        # get game over 
        r_turn = game.getGameOver(player_curr)
        if r_turn != 0:
            return r_turn
        # log.debug(f'GAME TURNS: {gt}')

        # has not been expanded
        if key_short not in self.visited:

            # EXPAND
            self.visited.add(key_short)
            self.N[key_short] = 0
            self.r[key_short] = 0
            self.v[key_short] = -float('inf') if player_curr == key_short[1] else float('inf')

            legal_actions = game.getLegalActionState()

            for action in legal_actions:
                game_c = copy.deepcopy(game)
                game_c.makeMove(action)

                # get the score difference
                score_me = game_c.board.getSeedsInStore(player_curr)
                opp_side = copy.deepcopy(player_curr)
                opp_side = player_curr.opposite(opp_side)
                score_opp = game_c.board.getSeedsInStore(opp_side)

                key_long = key_short + (action,)
                self.v_sa[key_long] = score_me - score_opp

                j_s_in = game_c.no_turns == 2 and game_c.swap_occured

                if player_curr == key_short[1] and not j_s_in:
                    if self.v_sa[key_long] > self.v[key_short]:
                        self.v[key_short] = self.v_sa[key_long]
                else:
                    if self.v_sa[key_long] < self.v[key_short]:
                        self.v[key_short] = self.v_sa[key_long]
                
                # initialize RSA for current state
                self.r_sa[key_long] = 0
                self.N_sa[key_long] = EPSILON
            
            # PLAYOUT
            the_copy = copy.deepcopy(game)
            reward = self.play_out(player_curr, the_copy, self.playout_n)

            # UPDATE
            self.update(legal_actions, key_short, reward, player_curr == key_short[1], just_swapped)
               
            # BACKPROPAGATE
            if just_swapped:
                reward = -self.r[key_short]
            else:
                reward = self.r[key_short] if game.prev_player == game.turn else -self.r[key_short]
            
            if None not in key_long_parent:
                self.r_sa[key_long_parent] = reward
                self.N_sa[key_long_parent] = 1
                self.v_sa[key_long_parent] = self.v[key_short]
            
            return reward

        # SELECT
        action = self.select(game, key_short)

        legal_actions = game.getLegalActionState()

        # play action
        game.makeMove(action)
        prev_p = copy.deepcopy(game.prev_player)
        game.prev_player = copy.deepcopy(game.turn)

        key_long_curr = key_short + (action,)

        # recursive call to go deeper in the tree
        reward = self.simulate(game, key_long_curr, player_curr)

        # UPDATE
        self.update(legal_actions, key_short, reward, player_curr == key_short[1], game.no_turns == 2 and game.swap_occured)

        if just_swapped:
            reward = -self.r[key_short]
        else:
            reward = self.r[key_short] if prev_p == game.turn else -self.r[key_short]
            
        if None not in key_long_parent:
            self.r_sa[key_long_parent] += reward
            self.N_sa[key_long_parent] += 1
            self.v_sa[key_long_parent] = self.v[key_short]

        return reward


    def play_out(self, curr_player, game, n):
        """

        """
        starting_turn = curr_player
        num_turn = game.no_turns
        done = game.getGameOver(starting_turn) != 0

        for _ in range(n):
            legal_actions = game.getLegalActionState()
            action = np.random.choice(legal_actions)

            _, _, done = game.makeMove(action)

            if done:
                break
        
        # log.debug(f'FINAL STATE AFTER PLAYOUT: \n{game.board.toString()}\n')

        if num_turn <= 1 and game.swap_occured:
            return -game.getGameOver(starting_turn)
        else:
            return game.getGameOver(starting_turn)