from python_agent.protocol import Protocol
from python_agent.board import Board
from python_agent.kalah_train import Kalah
from python_agent.side import Side
from python_agent.msgType import MsgType

import sys
import random
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.DEBUG, filename='debug.log')
logging.debug('******** NEW GAME ********')

from KalahModel.KalahNetTrain import KalahNetTrain
from KalahModel.MCTS import MCTS

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10 
LR = 0.001
DROPOUT = 0.3
NUM_GAMES = 40

def sendMsg(msg):
    print(msg)
    sys.stdout.flush()


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
    net.load_model_checkpoint("thedestroyerofworlds.pth")
    mcts = MCTS(game, net, 3, 25)

    just_moved = False
    p = Protocol()
    
    while True:
        
        # read message
        message = recvMsg()
        logging.debug(f'MSG: {message}')

        # interprest message with protocol
        message_type = p.getMessageType(message)
        logging.debug(f'MSG TYPE: {message_type}')

        # START
        if message_type == MsgType.START:
            
            turn = p.interpretStartMsg(message)

            if turn:
                logging.debug(f'I AM PLAYING ON START :)')

                action = np.argmax(mcts.getProbs(tau=0))
                msg = p.createMoveMsg(action)
                sendMsg(msg)
                game.makeMove(action)
                just_moved = True

                logging.debug(f'ACTION: {action}')
                logging.debug(f'I am sending this message: {msg}')
                logging.debug(f'Board after I moved: \n {game.board.toString()} \n')
            else:
                logging.debug(f'I AM NOT PLAYING ON START :(\n')
                continue

        # CHANGE
        elif message_type == MsgType.STATE:

            # update board when I do not move
            if not just_moved:
                action = p.get_action(message)
                game.makeMove(action)
                just_moved = False

                logging.debug(f'I HAVE NOT JUST MOVED, SO I NEED TO UPDATE THE BOARD WITH THE OPPONENT\'S MOVE')
                logging.debug(f'ACTION: {action}')
                logging.debug(f'Board after other player moved: \n {game.board.toString()} \n')

            if p.get_again(message):            
                action = np.argmax(mcts.getProbs(tau=0))
                if action == 0:
                    action = -1
                    msg = p.createSwapMsg()                   
                else:
                    msg = p.createMoveMsg(action)

                sendMsg(msg)
                game.makeMove(action)
                just_moved = True

                logging.debug(f'MY TURN TO MOVE')
                logging.debug(f'ACTION: {action}')
                logging.debug(f'I am sending this message: {msg}')
                logging.debug(f'Board after I moved: \n {game.board.toString()}')
            
            if not p.get_again(message):
                just_moved = False
                logging.debug(f'Setting just_moved to FALSE')

        # END
        elif message_type == MsgType.END:
            logging.debug(f'THIS IS THE END :(')
            break


if __name__ == "__main__":
    main()