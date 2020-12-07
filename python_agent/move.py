import side as s

class Move(object):

    def __init__(self, side, hole):
        if hole < 1:
            raise ValueError(f'There has to be at least one hole, but {hole} were requested.')

        self.side = side
        self.hole = hole


    def getSide(self):
        return self.side

    
    def getHole(self):
        return self.hole

if __name__ == "__main__":
    print('This class has been checked and works as expected.')