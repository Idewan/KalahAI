import logging
import torch
import random
import numpy as np
import time
import math
from tqdm import tqdm

from KalahModel.KalahNetTrain import KalahNetTrain
from KalahModel.MCTS import MCTS
from KalahModel.Memory import Memory

from python_agent.kalah_train import Kalah
from python_agent.board import Board
import Arena as bitchcage

from pickle import Pickler, Unpickler

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10 
LR = 0.02
DROPOUT = 0.3
NUM_GAMES = 40

class Trainer():
    
    def __init__(self, game):
        self.game = game
        self.iter = 0
        self.net = KalahNetTrain(self.game, BATCH_SIZE, DEVICE, EPOCHS, LR, DROPOUT, self.iter)
        self.memory = Memory(200000)
        self.num_eps = 100
        self.num_iters = 1000
        self.mcts_sims = 100
        self.cpuct_t = 3
        self.cpuct_c = 1
        self.opp_nnet = KalahNetTrain(self.game, BATCH_SIZE, DEVICE, EPOCHS, LR, DROPOUT, self.iter)
        self.og_nnet = KalahNetTrain(self.game, BATCH_SIZE, DEVICE, EPOCHS, LR, DROPOUT, self.iter)
        self.threshold = 0.55

    
    def policyIter(self):

        for i in range(self.num_iters):
            print(f"Executing iteration {i}")
            self.iter+=1
            for _ in tqdm(range(self.num_eps), desc='Self Play'):
                # reset and execute episode, i.e. one game using MCTS
                self.game.reset()
                self.executeEpisode()
            
            # back up of the memory
            # log.info("Saving back-up of the memory")
            # self.save_training_memory(f"checkpoint_{i}")
            
            nnet_name = "model_100_1000_100/checkpoint_{}.pth".format(i)
            temp_name = "model_100_1000_100/temp.pth"
            self.net.save_model_checkpoint(temp_name)
            self.opp_nnet.load_model_checkpoint(temp_name)

            #Save the original network so we can compare further down the
            if i == 0:
                self.og_nnet.load_model_checkpoint(temp_name)
            
            self.game.reset()   #Reset game

            # MCTS for the opponent
            opp_mcts = MCTS(self.game, self.opp_nnet, self.cpuct_c, self.mcts_sims)

            print("Training...")
            self.net.train(self.memory)
            curr_mcts = MCTS(self.game, self.net, self.cpuct_c, self.mcts_sims)

            arena = bitchcage.Arena(lambda x: np.argmax(curr_mcts.getProbs(tau=0)),
                        lambda x: np.argmax(opp_mcts.getProbs(tau=0)), self.game, self.net)

            # ARENA PART
            n_win, n_draw, n_lose = arena.playGames(NUM_GAMES)

            print(f"Wins: {n_win} Draws: {n_draw}, Losses: {n_lose}")
            if float(n_win / (n_draw + n_lose + n_win)) < self.threshold:
                print('Old model still better')
                self.net.load_model_checkpoint(temp_name)
            else:
                print('New model is banging')
                self.net.save_model_checkpoint(nnet_name)
                self.net.save_model_checkpoint("model_100_1000_100/thedestroyerofworlds.pth")

                print("Pitting against the Original Network")
                self.game.reset()
                og_mcts = MCTS(self.game, self.og_nnet, self.cpuct_c, self.mcts_sims)
                arena_og = bitchcage.Arena(lambda x: np.argmax(curr_mcts.getProbs(tau=0)),
                        lambda x: np.argmax(og_mcts.getProbs(tau=0)), self.game, self.net)

                n_win, n_draw, n_lose = arena_og.playGames(NUM_GAMES)


        return self.net
    
    # Has to be edited once the MCTS changes
    def executeEpisode(self):
        # start = time.time()
        examples = []
        mcts = MCTS(self.game, self.net, self.cpuct_t, self.mcts_sims)
        state_np = self.net.board_view_player()

        # print(f'State from view of player: {state_np}')  

        while True:
            pi = mcts.getProbs() # start from the board at the bieginn ofthe game
            pi_torch = torch.from_numpy(pi.astype(np.float64))

            examples.append([state_np, pi_torch, self.game.turn])
            action = np.random.choice(range(len(pi)), p=pi)
            if action == 0:
                action = -1

            _, _, _ = self.game.makeMove(action) # you made one move
            state_np = self.net.board_view_player()

            current_turn = self.game.turn

            reward = self.game.getGameOver(current_turn)
            
            # ACCOUNT FOR SwAPS
            if reward != 0:
                # print(f'Reward: {reward}')
                # print(f'Current turn: {current_turn}')
                # print(f'Swap: {self.game.swap_occured}')
                for i in range(len(examples)):
                    if (i == 0 or i == 1) and self.game.swap_occured:
                        # For the first two turns it changes where 
                        if current_turn == examples[i][2]: 
                          reward_torch = torch.tensor([[float(-reward)]])     
                        else:
                            reward_torch = torch.tensor([float(reward)])
                    elif (current_turn == examples[i][2]):
                        reward_torch = torch.tensor([[float(reward)]])
                    else:
                        reward_torch = torch.tensor([[float(-reward)]])
                    # print(f'Reward: {reward_torch}')
                    # print(f'Current turn: {examples[i][2]}')

                    # print(f'Reard torch: {reward_torch}')
                    self.memory.push(examples[i][0], examples[i][1], reward_torch)
                # print("TIME TAKEN EPSIODE: {:.3f}".format(time.time()-start))
                return
       


    def load_training_memory(self, filename):
        """
            :param filename: Checkpoint file name for memory (includes iter)
        """
        memory_file = f"{filename}.memory"
        with open(memory_file, "rb") as f:
            self.memory.memory = Unpickler(f).load()


    def save_training_memory(self, filename):
        """
            :param filename: Includes the number of iterations
            :return: returns nothing
        """
        memory_file = f"{filename}.memory"
        with open(memory_file, "wb+") as f:
            Pickler(f).dump(self.memory.memory)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename='debug_training.log')
    logging.debug('******** NEW GAME ********')
    # filenames for NeuralNet and Examples
    # filename_nn = str(input())
    # filename_ex = str(input())
 
    # create board and game
    board = Board(7, 7)
    game = Kalah(board)


    # # check if checkpoints can be taken
    # if filename_nn:
    #     kalahnn.nnet.load_model_checkpoint(filename_nn)

    # initialize the trainer
    t = Trainer(game)

    # check whether memory exist to take from
    # if filename_nn:
    #     t.load_training_memory(filename_nn)

    logging.info("Start the training process")
    t.policyIter()