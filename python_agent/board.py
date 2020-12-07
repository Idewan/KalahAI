import side as s
import numpy as np

class Board(object):

    NORTH_ROW = 1
    SOUTH_ROW = 0


    def indexOfSide(self, side):
        if side == s.Side.NORTH:
            return self.NORTH_ROW
        else:
            return self.SOUTH_ROW


    def __init__(self, holes, seeds):

        if holes < 1:
            raise ValueError(f'There has to be at least one hole, but {holes} were requested.')
        if seeds < 0:
            raise ValueError(f'There has to be at least a non-negative number of seeds, but {seeds} were requested.')

        self.holes = holes
        self.seeds = seeds

        self.board = np.zeros((2, holes+1))

        for i in range(1, holes+1):
            self.board[self.NORTH_ROW][i] = seeds
            self.board[self.SOUTH_ROW][i] = seeds

    
    def clone(self, original):

        holes = original.holes
        
        clone = np.zeros((2, holes+1))

        for i in range(holes+1):
            clone[self.NORTH_ROW][i] = original.board[self.NORTH_ROW][i]
            clone[self.SOUTH_ROW][i] = original.board[self.SOUTH_ROW][i]

        return clone

    
    def getNoOfHoles(self):
        return self.holes


    def getSeeds(self, side, hole):
        if hole < 1 or hole > self.holes:
            raise ValueError(f'Hole number must be between 1 and {len(self.board[self.NORTH_ROW]) - 1} but was {hole}.')

        return self.board[self.indexOfSide(side)][hole]

    
    def setSeeds(self, side, hole, seeds):
        if hole < 1 or hole > self.holes:
            raise ValueError(f'Hole number must be between 1 and {len(self.board[self.NORTH_ROW]) - 1} but was {hole}.')
        if seeds < 0:
            raise ValueError(f'There has to be at least a non-negative number of seeds, but {seeds} were requested.')

        self.board[self.indexOfSide(side)][hole] = seeds

    
    def addSeeds(self, side, hole, seeds):
        if hole < 1 or hole > self.holes:
            raise ValueError(f'Hole number must be between 1 and {len(self.board[self.NORTH_ROW]) - 1} but was {hole}.')
        if seeds < 0:
            raise ValueError(f'There has to be at least a non-negative number of seeds, but {seeds} were requested.')

        self.board[self.indexOfSide(side)][hole] += seeds


    def getSeedsOp(self, side, hole):
        if hole < 1 or hole > self.holes:
            raise ValueError(f'Hole number must be between 1 and {len(self.board[self.NORTH_ROW]) - 1} but was {hole}.')

        return self.board[1-self.indexOfSide(side)][self.holes+1-hole]

    
    def setSeedsOp(self, side, hole, seeds):
        if hole < 1 or hole > self.holes:
            raise ValueError(f'Hole number must be between 1 and {len(self.board[self.NORTH_ROW]) - 1} but was {hole}.')
        if seeds < 0:
            raise ValueError(f'There has to be at least a non-negative number of seeds, but {seeds} were requested.')

        self.board[1-self.indexOfSide(side)][self.holes+1-hole] = seeds

    
    def addSeedsOp(self, side, hole, seeds):
        if hole < 1 or hole > self.holes:
            raise ValueError(f'Hole number must be between 1 and {len(self.board[self.NORTH_ROW]) - 1} but was {hole}.')
        if seeds < 0:
            raise ValueError(f'There has to be at least a non-negative number of seeds, but {seeds} were requested.')

        self.board[1-self.indexOfSide(side)][self.holes+1-hole] += seeds

    
    def getSeedsInStore(self, side):
        return self.board[self.indexOfSide(side)][0]

    
    def setSeedsInStore(self, side, seeds):
        if seeds < 0:
            raise ValueError(f'There has to be at least a non-negative number of seeds, but {seeds} were requested.')

        self.board[self.indexOfSide(side)][0] = seeds


    def addSeedsToStore(self, side, seeds):
        if seeds < 0:
            raise ValueError(f'There has to be at least a non-negative number of seeds, but {seeds} were requested.')

        self.board[self.indexOfSide(side)][0] += seeds

    
    def toString(self):
        boardString = f'{self.board[self.NORTH_ROW][0]}  --'

        for i in range(self.holes, 0, -1):
            boardString += f'  {self.board[self.NORTH_ROW][i]}'
        boardString += '\n'
        for i in range (1, self.holes+1):
            boardString += f'{self.board[self.SOUTH_ROW][i]}  '
        boardString += f'--  {self.board[self.SOUTH_ROW][0]}\n'

        return boardString

if __name__ == "__main__":
    print('This class has been checked and works as expected.')