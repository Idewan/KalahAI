import logging as log
import torch
import random
import numpy as np
import time

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
LR = 0.001
DROPOUT = 0.5
NUM_GAMES = 40

class Trainer():
    
    def __init__(self, game, net):
        self.game = game
        self.net = net
        self.memory = Memory(200000)
        self.num_eps = 1
        self.num_iters = 1
        self.mcts_sims = 25
        self.opp_nnet = KalahNetTrain(game, BATCH_SIZE, DEVICE, EPOCHS, LR, DROPOUT)
        self.threshold = 0.6

    
    def policyIter(self):
        log.basicConfig(filename="training.log", level=log.INFO)

        for i in range(self.num_iters):
            for e in range(self.num_eps):
                self.game.reset()
                print(f"Executing episode {e}")
                self.executeEpisode()
            
            # back up of the memory
            log.info("Saving back-up of the memory")
            # self.save_training_memory(f"checkpoint_{i}")
            
            nnet_name = "checkpoints/checkpoint_{}.pth".format(i)
            temp_name = "checkpoints/temp.pth"
            self.net.save_model_checkpoint(temp_name)
            self.opp_nnet.load_model_checkpoint(temp_name)

            opp_mcts = MCTS(self.game, self.opp_nnet)

            log.info("Training ...")
            print("Training...")
            self.net.train(self.memory)
            curr_mcts = MCTS(self.game, self.net)

            arena = bitchcage.Arena(lambda x: np.argmax(curr_mcts.getProbs(tau=0)),
                          lambda x: np.argmax(opp_mcts.getProbs(tau=0)), self.game, self.net)

            # ARENA PART
            n_win, n_draw, n_lose = arena.playGames(NUM_GAMES)

            print(f"Wins: {n_win} Draws: {n_draw}, Losses: {n_lose}")
            if float(n_win / (n_draw + n_lose + n_win)) >= self.threshold:
                print('Old model still better')
                self.net.load_model_checkpoint(temp_name)
            else:
                print('New model is banging')
                self.net.save_model_checkpoint(nnet_name)
                self.net.save_model_checkpoint("checkpoints/thedestroyerofworlds.pth")

        return self.net
    

    def executeEpisode(self):
        start = time.time()
        examples = []
        mcts = MCTS(self.game, self.net)
        state = self.game.board.board
        # print("Execute Episode Part 1")
        state_np = self.net.board_view_player()      

        while True:
            pi = mcts.getProbs() # start from the board at the bieginn ofthe game
            pi_torch = torch.from_numpy(pi.astype(np.float64))
            # print(f'probs: {pi}')
            examples.append([state_np, pi_torch, self.game.turn])
            action = np.random.choice(range(len(pi)), p=pi)
            if action == 0:
                action = -1
            # print(f'action: {action}')
            state, _, _ = self.game.makeMove(action) # you made one move
            state_np = self.net.board_view_player()

            # print("Execute Episode Part 2")
            current_turn = self.game.turn

            reward = self.game.getGameOver(self.game.turn)
            # print(f'reward: {reward}')
            

            if reward != 0:
                for ex in examples:
                    if (current_turn == ex[2]):
                        reward_torch = torch.tensor([[float(reward)]])
                        self.memory.push(ex[0], ex[1], reward_torch)
                    else:
                        reward_torch = torch.tensor([[float(-reward)]])
                        self.memory.push(ex[0], ex[1], reward_torch)
                print("TIME TAKEN EPSIODE: {:.3f}".format(time.time()-start))
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
    filename_nn = str(input())
    filename_ex = str(input())
 
    # create board and game
    board = Board(7, 7)
    game = Kalah(board)

    # create neural network
    kalahnn = KalahNetTrain(game, BATCH_SIZE, DEVICE, EPOCHS, LR, DROPOUT)

    # check if checkpoints can be taken
    if filename_nn:
        kalahnn.nnet.load_model_checkpoint(filename_nn)

    #initialize the trainer
    t = Trainer(game, kalahnn)

    #check whether memory exist to take from
    if filename_nn:
        t.load_training_memory(filename_nn)

    log.info("Start the training process")
    t.policyIter()