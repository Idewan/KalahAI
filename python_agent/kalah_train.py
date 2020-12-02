from . import board as b
from . import side as s
from . import move as m

class Kalah(object):

    def __init__(self, board):
        if board is None:
            raise TypeError

        self.board = board

        self.actionspace = [1,2,3,4,5,6,7,-1]
        
        # Reward
        self.reward_player1 = 0
        self.reward_player2 = 0
        self.reward = 0

        # Players
        self.player1 = s.Side.NORTH
        self.player2 = s.Side.SOUTH
        self.turn = self.player1

        self.score_player1 = 0
        self.score_player2 = 0
        self.no_turns = 0


    def getBoard(self):
        return self.board

    
    def isLegalMove(self, move):
        return (move.getHole() <= self.board.getNoOfHoles()) and \
               (self.board.getSeeds(move.getSide(), move.getHole()) != 0)


    # return next_state (i.e. the board after the move, None if game ended), reward (0 if game not ended), if end of game
    def makeMove(self, action):

        print("make move stated")

        self.no_turns += 1

        move = None
        if action == -1:
            print("try to swap")
            if self.no_turns != 2:
                print("illegal swap")
                return None, -1, True
            print("swapped")
            self.swap()
            return self.board, self.reward, False
        else:
            print("create move")
            move = m.Move(self.turn, action)

        # if illegal move, then lose
        if not self.isLegalMove(move):
            print("illegal move")
            return None, -1, True

        # pick seeds
        seedsToSow = int(board.getSeeds(move.getSide(), move.getHole()))
        self.board.setSeeds(move.getSide(), move.getHole(), 0)
        print(f'seeds to sow: {seedsToSow}')

        holes = self.board.getNoOfHoles()
        receivingPits = 2 * holes + 1  # sow into: all holes + 1 store       
        rounds = int(seedsToSow // receivingPits)  # sowing rounds
        extra = int(seedsToSow % receivingPits)  # seeds for the last partial round
        # the first "extra" number of holes get "rounds"+1 seeds, the remaining ones get "rounds" seeds
        print(f'holes: {holes}')
        print(f'receivingPits: {receivingPits}')
        print(f'rounds: {rounds}')
        print(f'extra: {extra}')

        # sow the seeds of the full rounds (if any)
        if rounds != 0:
            print("more than one round")
            for i in range(1, holes+1):
                self.board.addSeeds(s.Side.NORTH, holes, rounds)
                self.board.addSeeds(s.Side.SOUTH, holes, rounds)
            self.board.addSeedsToStore(move.getSide(), rounds)

        # sow the extra seeds (last round)
        sowSide = move.getSide()
        sowHole = move.getHole()  # 0 means store
        print(f'sowside: {sowSide}')
        print(f'sowhole: {sowHole}')

        for i in range(extra, 0, -1):
            print(f'extra {i}')
            # go to next pit
            sowHole += 1
            print(f'sowhole: {sowHole}')
            if sowHole == 1:  # last pit was a store
                print("sowhole = 1")
                sowSide = sowSide.opposite(sowSide)
            if sowHole > holes:
                print("sowhole > holes")
                if sowSide == move.getSide():
                    print("sow side = move side")
                    sowHole = 0  # sow to the store now
                    self.board.addSeedsToStore(sowSide, 1)
                    print(self.board.toString())
                    continue
                else:
                    print("go to other side")
                    sowSide = sowSide.opposite(sowSide)
                    sowHole = 1
            # sow to hole
            self.board.addSeeds(sowSide, sowHole, 1)
            print(self.board.toString())
        
        # capture
        # last seed was sown on the moving player's side ...
        # ... not into the store ...
        # ... but into an empty hole (so there's 1 seed) ...
        # ... and the opposite hole is non-empty
        if sowSide == move.getSide() and \
           sowHole > 0 and \
           self.board.getSeeds(sowSide, sowHole) == 1 and \
           self.board.getSeedsOp(sowSide, sowHole) > 0:

            print("capture")
            self.board.addSeedsToStore(move.getSide(),
                                       1 + self.board.getSeedsOp(move.getSide(), sowHole))
            self.board.setSeeds(move.getSide(), sowHole, 0)
            self.board.setSeedsOp(move.getSide(), sowHole, 0)

        self.score_player1 = self.get_score(self.player1)
        self.score_player2 = self.get_score(self.player2)
        print(f'score 1: {self.score_player1}')
        print(f'score 2: {self.score_player2}')

        # game over?
        finishedSide = None
        if self.holesEmpty(self.board, move.getSide()):
            finishedSide = move.getSide()
        elif self.holesEmpty(self.board, move.getSide().opposite(move.getSide())):
            finishedSide = move.getSide().opposite(move.getSide())

        print(f'finished side: {finishedSide}')
        
        # game ended
        if finishedSide is not None:
            print("collecting remaining seeds")
            # collect the remaining seeds
            seeds = 0
            collectingSide = finishedSide.opposite(finishedSide)
            for i in range(1, holes+1):
                seeds += self.board.getSeeds(collectingSide, holes)
                self.board.setSeeds(collectingSide, holes, 0)
            self.board.addSeedsToStore(collectingSide, seeds)

            self.score_player1 = self.get_score(self.player1)
            self.score_player2 = self.get_score(self.player2)

            reward = 0
            if self.score_player1 > self.score_player2 and move.getSide() == self.player1:
                reward = 1
            elif self.score_player1 < self.score_player2 and move.getSide() == self.player1:
                reward = -1
            elif self.score_player2 > self.score_player1 and move.getSide() == self.player2:
                reward = 1
            elif self.score_player2 < self.score_player1 and move.getSide() == self.player2:
                reward = -1

            return None, reward, True

        # whose turn is it?
        if sowHole == 0:  # the store implies (sowSide == move.getSide())
            print("sowhole = 0")
            self.turn = move.getSide()  # move again
        else:
            print("sowhole != 0")
            self.turn = move.getSide().opposite(move.getSide())
        
        # game has not ended
        return self.board, 0, False


    def holesEmpty(self, board, side):
        for hole in range(1, board.getNoOfHoles()+1):
            if board.getSeeds(side, hole) != 0:
                return False
        return True

    
    def gameOver(self):
        return self.holesEmpty(self.board, s.Side.NORTH) or holesEmpty(self.board, s.Side.SOUTH)


    def reset(self):
        new_board = b.Board(self.board.holes, self.board.seeds)
        self.__init__(new_board)


    def swap(self):
        s = self.player1
        self.player1 = self.player2
        self.player2 = s


    def get_score(self, side):
        return self.board.getSeedsInStore(side)

if __name__ == "__main__":
    board = b.Board(7, 7)

    game = Kalah(board)
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(2))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(-1))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(2))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(1))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(3))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(3))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(3))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(6))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(6))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(6))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(5))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(2))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(6))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(4))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(3))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(3))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(6))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(5))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(6))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(2))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(6))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(1))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(1))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")
    
    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(5))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(3))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(6))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(4))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(4))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(7))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(1))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(5))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(4))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(3))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")

    print(game.makeMove(3))
    print(game.__dict__)
    print(game.board.toString())
    print("------------------------------------------------")
