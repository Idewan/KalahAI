import sys

import numpy as np

from tqdm import tqdm 
from KalahNet import KalahNet as kalahnet

sys.path.append('../')
from python_agent.side import Side

import torch
import torch.optim as optim


class KalahNetTrain(object):

    def __init__(self, env, batch_size, device, epochs, lr=0.001, dropout=0.3):
        """
            Initialize the training class for KalahNet NN class

            :param env: Kalah_train environment
            :param batch_size: Training batchsize
            :param device: Cuda enabled or not
            :param epochs: Number of epochs to train 
            :param lr: learning rate
            :param dropout: Probability of drop out in dropout layers
        """

        #Initialize neural net
        self.device = device
        self.is_cuda = torch.cuda.is_available()
        self.nnet = kalahnet(7, dropout).to(self.device)
        self.env = env

        #Training parameters
        self.batch_size = batch_size
        self.lr = lr 
        self.epochs = epochs
    
    def train(self, memory):
        """
            Train the neural network according to a combination of 
            sum of mean squared error and cross-entropy loss
            
            l = (z − v)^2 − πTlog p + c||θ||2

            :param memory: memory of examples generated by MCTS
            :return: returns nothing
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(self.epochs):
            print(f'EPOCH: {epoch +1}')

            batch_count = int(len(memory) / self.batch_size)

            t = tqdm(range(batch_count), desc="Training Net")

            for _ in t:
                transitions = memory.sample(self.batch_size)
                batch = memory.Transition(*zip(*transitions))

                #Sequence batch into states, pi, values
                s = torch.cat(batch.state)
                pi = torch.cat(batch.pi)
                v = torch.cat(batch.v)

                if self.is_cuda:
                    s, pi, v = s.contiguous().cuda(), pi.contiguous().cuda(), v.contiguous().cuda()

                #Compute current pi and values given current nn
                output_pi, output_v = self.nnet(s)
                l_pi = self.loss_pi(pi, output_pi)
                l_v = self.loss_v(v, output_v)
                loss = l_v + l_pi

                #Compute gradient and perform single step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, s):
        """
            Predicts what actions should be taken

            :param s: current state (kalah board) - view corrected
        """
        #Preparing input
        self.nnet.eval()
        s = s.contiguous().cuda() if self.is_cuda else s
        with torch.no_grad:
            pi, v = self.nnet(s)
        
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    
    def board_view_player(self, board=None, turn=None):
        """
            The board, [[side.South], [side.North]], is always south facing.
            This methods returns the board so that is facing that the current
            player's turn exists in the first row first column.

            :param board: Kalah board
            :return: returns Board facing correctly for the current player's turn
        """
        s = self.env.board.board.copy() if board is not None else board.copy()
        p = self.env.turn if turn is not None else turn

        if p == Side.NORTH:
            s[[0,1]] = s[[1,0]]
        
        s = s.astype(np.float64)
        s = torch.from_numpy(s.flatten())

        return s

    def loss_pi(self, targets, outputs):
        """
            :param targets: Target prob distribution
            :param outputs: Output prob distribution
        """
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        """
            :param targets: Target values
            :param outputs: Output values
        """
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_model_checkpoint(self, filename, title):
        """
            :param filename: name of the file/dir
            :param title: title of the weights file
        """
        path = f"{filename}/{title}"
        torch.save(self.nnet.state_dict(), path)

    def load_model_checkpoint(self, filename, title):
        """
            :param filename: name of the file/dir
            :param title: title of the weights file
        """
        path = f"{filename}/{title}"
        state_dict = pytorch.load(path, map_location= None if self.is_cuda else 'cpu')
        self.nnet.load_state_dict(state_dict)