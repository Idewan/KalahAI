import enum

class Side(enum.Enum):
    NORTH = 0
    SOUTH = 1

    def opposite(self, side):
        if side == self.NORTH:
            return self.SOUTH
        else:
            return self.NORTH