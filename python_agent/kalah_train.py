from . import board as b
from . import side as s
from . import move as m

class Kalah(object):

    def __init__(self, board):
        if board is None:
            raise TypeError

        self.board = board

        self.actionspace = [1,2,3,4,5,6,7,-1]
        self.actionspace_size = len(self.actionspace)
        self.actionspace_legal = []
        
        # Reward
        self.reward = 0

        # Players
        self.player1 = s.Side.SOUTH
        self.player2 = s.Side.NORTH
        self.turn = self.player1
        self.prev_player = self.player1

        self.score_player1 = 0
        self.score_player2 = 0
        self.no_turns = 0

        self.swap_occured =  False


    def getLegalMoves(self):
        legal_moves = self.getLegalActionState()
        legal_moves_bin = [0] * 8
        for m in legal_moves:
            if m != -1:
                legal_moves_bin[m] = 1
            elif m == -1:
                legal_moves_bin[0] = 1
        return legal_moves_bin


    def getLegalActionState(self):
        legal_moves_in_board = []
        for i in range(1,8):
            if self.board.getSeeds(self.turn, i) != 0:
                legal_moves_in_board.append(i)

        if self.no_turns == 1:
            legal_moves_in_board.append(-1)
        
        return legal_moves_in_board


    def getBoard(self):
        return self.board

    
    def isLegalMove(self, move):
        return (move.getHole() <= self.board.getNoOfHoles()) and \
               (self.board.getSeeds(move.getSide(), move.getHole()) != 0)


    # return next_state (i.e. the board after the move, None if game ended), reward (0 if game not ended), if end of game
    def makeMove(self, action):

        self.no_turns += 1

        move = None
        if action == -1:
            # illegal swap
            if self.no_turns != 2:
                if self.turn == self.player1:
                    self.reward = -1
                else:
                    self.reward = 1
                return None, self.reward, True
            # legal swap
            self.swap()
            return self.board.board, 0, False
        else:
            # create a move
            move = m.Move(self.turn, action)

        # print(f'Move: {move.getSide()}, {move.getHole()}')

        # if illegal move, then lose
        if not self.isLegalMove(move):
            # print("ILLEGAL MOVE")
            if self.turn == self.player1:
                self.reward = -1
            else:
                self.reward = 1
            return None, self.reward, True

        # pick seeds
        seedsToSow = int(self.board.getSeeds(move.getSide(), move.getHole()))
        self.board.setSeeds(move.getSide(), move.getHole(), 0)
        # print(f'Seeds to sow: {seedsToSow}')

        holes = self.board.getNoOfHoles()
        receivingPits = 2 * holes + 1  # sow into: all holes + 1 store       
        rounds = int(seedsToSow // receivingPits)  # sowing rounds
        extra = int(seedsToSow % receivingPits)  # seeds for the last partial round
        # the first "extra" number of holes get "rounds"+1 seeds, the remaining ones get "rounds" seeds
        # print(f'Holes: {holes}')
        # print(f'Receiving Pits: {receivingPits}')
        # print(f'Rounds: {rounds}')
        # print(f'Extra: {extra}')

        # sow the seeds of the full rounds (if any)
        if rounds != 0:
            for hole in range(1, holes+1):
                self.board.addSeeds(s.Side.NORTH, hole, rounds)
                self.board.addSeeds(s.Side.SOUTH, hole, rounds)
            self.board.addSeedsToStore(move.getSide(), rounds)

        # sow the extra seeds (last round)
        sowSide = move.getSide()
        sowHole = move.getHole()  # 0 means store

        for i in range(extra, 0, -1):
            # go to next pit
            sowHole += 1
            if sowHole == 1:  # last pit was a store
                sowSide = sowSide.opposite(sowSide)
            if sowHole > holes:
                if sowSide == move.getSide():
                    sowHole = 0  # sow to the store now
                    self.board.addSeedsToStore(sowSide, 1)
                    continue
                else:
                    sowSide = sowSide.opposite(sowSide)
                    sowHole = 1
            # sow to hole
            self.board.addSeeds(sowSide, sowHole, 1)
        
        # capture
        # last seed was sown on the moving player's side ...
        # ... not into the store ...
        # ... but into an empty hole (so there's 1 seed) ...
        # ... and the opposite hole is non-empty
        if sowSide == move.getSide() and \
           sowHole > 0 and \
           self.board.getSeeds(sowSide, sowHole) == 1 and \
           self.board.getSeedsOp(sowSide, sowHole) > 0:
            self.board.addSeedsToStore(move.getSide(),
                                       1 + self.board.getSeedsOp(move.getSide(), sowHole))
            self.board.setSeeds(move.getSide(), sowHole, 0)
            self.board.setSeedsOp(move.getSide(), sowHole, 0)

        self.score_player1 = self.get_score(self.player1)
        self.score_player2 = self.get_score(self.player2)

        # game over?
        finishedSide = None
        if self.holesEmpty(self.board, move.getSide()):
            finishedSide = move.getSide()
        elif self.holesEmpty(self.board, move.getSide().opposite(move.getSide())):
            finishedSide = move.getSide().opposite(move.getSide())
        
        # game ended
        if finishedSide is not None:
            # collect the remaining seeds
            seeds = 0
            collectingSide = finishedSide.opposite(finishedSide)
            for hole in range(1, holes+1):
                seeds += self.board.getSeeds(collectingSide, hole)
                self.board.setSeeds(collectingSide, hole, 0)
            self.board.addSeedsToStore(collectingSide, seeds)

            self.score_player1 = self.get_score(self.player1)
            self.score_player2 = self.get_score(self.player2)

            self.reward = 0
            if self.score_player1 > self.score_player2:
                self.reward = 1
            elif self.score_player1 < self.score_player2:
                self.reward = -1

            return self.board.board, self.reward, True

        # whose turn is it?
        if self.no_turns == 1:
            self.turn = move.getSide().opposite(move.getSide())  # pie rule, player 1 cannot move again on first turn
        elif sowHole == 0:  # the store implies (sowSide == move.getSide())
            self.turn = move.getSide()  # move again
        else:
            self.turn = move.getSide().opposite(move.getSide())
        
        # game has not ended
        self.reward = 0
        return self.board.board, self.reward, False


    def holesEmpty(self, board, side):
        for hole in range(1, board.getNoOfHoles()+1):
            if board.getSeeds(side, hole) != 0:
                return False
        return True

    
    def gameOver(self):
        return self.holesEmpty(self.board, s.Side.NORTH) or self.holesEmpty(self.board, s.Side.SOUTH)
    
    
    def getGameOver(self, player):
        if self.gameOver():
            if player == self.player1:
                if self.get_score(self.player1) > self.get_score(self.player2):
                    return 1
                elif self.get_score(self.player1) < self.get_score(self.player2):
                    return -1
                else:
                    return 1e-4
            elif player == self.player2:
                if self.get_score(self.player1) < self.get_score(self.player2):
                    return 1
                elif self.get_score(self.player1) > self.get_score(self.player2):
                    return -1
                else:
                    return 1e-4
        else:
            return 0


    def reset(self):
        new_board = b.Board(self.board.holes, self.board.seeds)
        self.__init__(new_board)


    def swap(self):
        s = self.player1
        self.player1 = self.player2
        self.player2 = s
        self.swap_occured = True


    def get_score(self, side):
        return self.board.getSeedsInStore(side)

if __name__ == "__main__":
    print('This class has been checked and works as expected.')
    print("***** START GAME ******")
    board = b.Board(7, 7)
    game = Kalah(board)
    print()
    print(game.__dict__)
    print(game.board.toString())

    while True:
        print(f'Player {game.turn}')
        action = int(input("Action to take: "))
        state, reward, done = game.makeMove(action)

        if done:
            print()
            print("DONE")
            print(f'Reward: {reward}')
            break

        print()
        print(game.__dict__)
        print(game.board.toString())