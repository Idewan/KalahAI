import logging as log

from KalahModel import KalahNetTrain

class Arena():

    def __init__(self, p1, p2, game, net):
        self.player1 = p1
        self.player2 = p2
        self.game = game
        self.net = net

    
    def playGame(self):
        """
            Let's explain what is going on. Player1 and Player2 are functions which take
            the max action from a MCTS run. 
        """
        players = [self.player1, self.player2]

        while self.game.getGameOver() == 0:
            if self.game.player1 == self.game.turn:
                curPlayer = 0
            else:
                curPlayer = 1

            action = players[curPlayer](game)

            valids = self.game.getLegalMoves()

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            next_state, _, _ = self.game.makeMove(action)
       
        return self.game.getGameOver(self.game.turn)

    def playGames(self, num):

        num = num // 2
        oneWon = 0
        twoWon = 0
        draws = 0

        for _ in range(num):
            gameResult = self.playGame()
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws