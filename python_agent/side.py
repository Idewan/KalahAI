import enum

class Side(enum.Enum):
    NORTH = 1
    SOUTH = 0

    def opposite(self, side):
        if side == self.NORTH:
            return self.SOUTH
        else:
            return self.NORTH