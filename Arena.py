import logging as log

from KalahModel import KalahNetTrain

class Arena():

    def __init__(self, p1, p2, game, net):
        self.player1 = p1
        self.player2 = p2
        self.game = game
        self.net = net

    
    def playGame(self, player):
        """
            Let's explain what is going on. Player1 and Player2 are functions which take
            the max action from a MCTS run. 
        """
        self.game.reset()

        players = [self.player1, self.player2]

        while self.game.getGameOver(self.game.turn) == 0:
            if self.game.player1 == self.game.turn:
                curPlayer = 0
            else:
                curPlayer = 1
            print(f"Game turn {self.game.turn}")
            print(f"Game board {self.game.board.toString()}")
            action = players[curPlayer](1)
            print(f"Game action {action}")

            valids = self.game.getLegalMoves()

            if valids[action] == 0:
                print(f'Action {action} is not valid!')
                print(f'valids = {valids}')
                assert valids[action] > 0
            next_state, _, _ = self.game.makeMove(action)

        if "p1" == player:
            result = self.game.getGameOver(self.game.player1)
        else:
            result = self.game.getGameOver(self.game.player2)

        return result

    def playGames(self, num):

        num = num // 2
        oneWon = 0
        twoWon = 0
        draws = 0

        for _ in range(num):
            gameResult = self.playGame("p1")
            print(f'Playing as player 1 {gameResult}')
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1
        print(f"PLAYER 1 WIN: {oneWon} DRAWS: {draws} LOSSES: {twoWon}")

        for _ in range(num):
            gameResult = self.playGame("p2")
            print(f'Playing as player 2 {gameResult}')
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        print(f"WIN: {oneWon} DRAWS: {draws} LOSSES: {twoWon}")
        return oneWon, draws, twoWon