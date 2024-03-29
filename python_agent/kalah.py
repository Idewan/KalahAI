import board as b
import side as s
import move as m

# TODO CHECK AND DEBUG THIS CLASS

class Kalah(object):

    def __init__(self, board):
        if board is None:
            raise TypeError

        self.board = board


    def getBoard(self):
        return self.board

    
    def isLegalMove(self, move):
        return (move.getHole() <= self.board.getNoOfHoles()) and \
               (self.board.getSeeds(move.getSide(), move.getHole()) != 0)


    def makeMove(self, move):
        seedsToSow = board.getSeeds(move.getSide(), move.getHole())
        self.board.setSeeds(move.getSide(), move.getHole, 0)

        holes = self.board.getNoOfHoles()
        receivingPits = 2 * holes + 1
        rounds = seedsToSow / receivingPits
        extra = seedsToSow % receivingPits

        if rounds != 0:
            for i in range(1, holes+1):
                self.board.addSeeds(s.Side.NORTH, hole, rounds)
                self.board.addSeeds(s.Side.SOUTH, hole, rounds)
            self.board.addSeedsToStore(move.getSide(), rounds)

        sowSide = move.getSide()
        sowHole = move.getHole()

        for i in range(extra, 0, -1):
            sowHole += 1
            if sowHole == 1:
                sowSide = sowSide.opposite()
            if sowHole > holes:
                if sowSide == move.getSide():
                    sowHole = 0
                    self.board.addSeedsToStore(sowSide, 1)
                    continue
                else:
                    sowSide = sowSide.opposite()
                    sowHole = 1
            self.board.addSeeds(sowSide, sowHole, 1)
        
        if sowSide == move.getSide() and \
           sowHole > 0 and \
           self.board.getSeeds(sowSide, sowHole) == 1 and \
           self.board.getSeedsOp(sowSide, sowHole) > 0:

            self.board.addSeedstoStore(move.getSide(),
                                       1 + self.board.getSeedsOp(move.getSide(), sowHole))
            self.board.setSeeds(move.getSide(), sowHole, 0)
            self.board.setSeedsOp(move.getSide(), sowHole, 0)

        finishedSide = None
        if self.holesEmpty(self.board, move.getSide()):
            finishedSide = move.getSide()
        elif self.holesEmpty(self.board, move.getSide().opposite()):
            finishedSide = move.getSide().opposite()
        
        if finishedSide is not None:
            seeds = 0
            collectingSide = finishedSide.opposite()
            for i in range(1, holes+1):
                seeds += self.board.getSeeds(collectingSide, hole)
                self.board.setSeeds(collectingSide, hole, 0)
            self.board.addSeedsToStore(collectingSide, seeds)

        if sowHole == 0:
            return move.getSide()
        else:
            return move.getSide().opposite()


    def holesEmpty(self, board, side):
        for i in range(1, board.getNoOfHoles()+1):
            if board.getSeeds(side, hole) != 0:
                return False
            return True

    
    def gameOver(self):
        return self.holesEmpty(self.board, s.Side.NORTH) or holesEmpty(self.board, s.Side.SOUTH)