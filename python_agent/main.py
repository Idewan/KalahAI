from protocol import Protocol
from board import Board
from kalah_train import Kalah
from side import Side
from msgType import MsgType

import sys
import random
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.DEBUG, filename='debug.log')
logging.debug('NEW GAME')

sys.path.append('../')
from KalahModel.KalahNetTrain import KalahNetTrain
from KalahModel.MCTS import MCTS

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10 
LR = 0.001
DROPOUT = 0.5
NUM_GAMES = 40

def sendMsg(msg):
    print(msg, flush=True)


def recvMsg():
    """
        Receive the message from stdin
    """
    msg = sys.stdin.readline()

    if msg is None:
        raise EOFError('Input ended unexpectedly.')

    return msg


def main():
    
    board = Board(7, 7)
    game = Kalah(board)

    # initialise the agent
    net = KalahNetTrain(game, BATCH_SIZE, DEVICE, EPOCHS, LR, DROPOUT)
    net.load_model_checkpoint("../thedestroyerofworlds.pth")
    mcts = MCTS(game, net)

    just_moved = False
    p = Protocol()
    
    while True:
        
        # read message
        message = recvMsg()
        logging.debug(f'Message: {message}')

        # interprest message with protocol
        message_type = p.getMessageType(message)
        logging.debug(f'Message type: {message_type}')

        # START
        if message_type == MsgType.START:

            logging.debug(f'START')
            
            turn = p.interpretStartMsg(message)

            logging.debug(f'{turn}')

            if turn:
                logging.debug(f'Play turn on start')
                action = np.argmax(mcts.getProbs(tau=0))
                logging.debug(f'Action: {action}')
                msg = p.createMoveMsg(action)
                logging.debug(f'I am sending this message {msg}')
                sendMsg(msg)
                game.makeMove(action)
                logging.debug(f'board after i played my starting move:\n {game.board.toString()}')
            else:
                continue

        # CHANGE
        elif message_type == MsgType.STATE:

            # this is the update when opponent moves
            if not just_moved:
                logging.debug(f'Not my turn')
                action = p.get_action(message)
                logging.debug(f'other player takes action {action}')
                game.makeMove(action)
                just_moved = False
                logging.debug(f'game board after other player moved: \n {game.board.toString()}')

            if p.get_again(message):            
                logging.debug(f'now I make a move')
                action = np.argmax(mcts.getProbs(tau=0))
                if action == 0:
                    action = -1
                logging.debug(f'my action is {action}')
                        
                if action == -1:
                    msg = p.createSwapMsg()
                else:
                    msg = p.createMoveMsg(action)
                logging.debug(f'I am sending this message {msg}')
                sendMsg(msg)
                game.makeMove(action)
                just_moved = True
                logging.debug(f'game board after I moved: \n {game.board.toString()}')
            
            if not p.get_again(message):
                just_moved = False

        # END
        elif message_type == MsgType.END:
            logging.debug(f'END')
            break


if __name__ == "__main__":
    main()