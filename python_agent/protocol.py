import moveTurn as mt
import msgType as msgt
import side as s
import invalidMessageException as ime

# TODO: CHECK AND DEBUG THIS CLASS

class Protocol(object):

    def createMoveMsg(self, hole):
        return f'MOVE;{hole}'


    def createSwapMsg(self):
        return f'SWAP'
    

    def getMessageType(self, msg):
        if msg.startswith('START;'):
            return msgt.MsgType.START
        elif msg.startswith('CHANGE;'):
            return msgt.MsgType.STATE
        elif msg == 'END\n':
            return msgt.MsgType.END
        else:
            raise ime.InvalidMessageException('Could not determine message type.')

    
    def interpretStartMsg(self, msg):
        if msg[-1] != '\n':
            raise ime.InvalidMessageException('Message not terminated with 0x0A character.')

        position = msg[6:len(msg)-1:1]
        if position == 'South':
            return True
        elif position == 'North':
            return False
        else:
            raise ime.InvalidMessageException(f'Illegal position parameter: {position}')


    def interpretStateMsg(self, msg, board):
        moveTurn = mt.MoveTurn()

        if msg[len(msg)-1] != '\n':
            raise ime.InvalidMessageException('Message not terminated with 0x0A character.')

        msgParts = msg.split(';', 4)
        if len(msgParts) != 4:
            raise ime.InvalidMessageException('Missing arguments.')

        if msgParts[1] == 'SWAP':
            moveTurn.move = -1
        else:
            try:
                moveTurn.move = int(msgParts[1])
            except ValueError as e:
                raise ime.InvalidMessageException(f'Illegal value for move parameter: {str(e)}')
        
        boardParts = msgParts[2].split(',', -1)

        if 2 * (board.getNoOfHoles()+1) != len(boardParts):
            raise ime.InvalidMessageException(f'Board dimensions in message ({len(boardParts)} entries) are not as expected ({2*board.getNoOfHoles()+1}) entries).')

        try:
            for i in range(board.getNoOfHoles()):
                board.setSeeds(s.Side.NORTH, i+1, int(boardParts[1]))
                board.setSeedsInStore(s.Side.NORTH, int(boardParts[board.getNoOfHoles()]))
            for i in range(board.getNoOfHoles()):
                board.setSeeds(s.Side.SOUTH, i+1, int(boardParts[i+board.getNoOfHoles()+1]))
                board.setSeedsInStore(s.Side.NORTH, int(2*board.getNoOfHoles()+1))
        except ValueError as e:
            raise ime.InvalidMessageException(f'Illegal value for seed count {str(e)}')
        
        moveTurn.end = False
        if msgParts[3] == 'YOU\n':
            moveTurn.again = True
        elif msgParts[3] == 'OPP\n':
            moveTurn.again = False
        elif msgParts[3] == 'END\n':
            moveTurn.end = True
            moveTurn.again = False
        else:
            raise ime.InvalidMessageException(f'Illegal valule for turn parameter: {msgParts[3]}')

        return moveTurn


    def get_action(self, msg):

        if msg[len(msg)-1] != '\n':
            raise ime.InvalidMessageException('Message not terminated with 0x0A character.')

        msgParts = msg.split(';', 4)
        if len(msgParts) != 4:
            raise ime.InvalidMessageException('Missing arguments.')

        if msgParts[1] == 'SWAP':
            action = -1
        else:
            try:
                action = int(msgParts[1])
            except ValueError as e:
                raise ime.InvalidMessageException(f'Illegal value for move parameter: {str(e)}')

        return action


    def get_again(self, msg):

        if msg[len(msg)-1] != '\n':
            raise ime.InvalidMessageException('Message not terminated with 0x0A character.')

        msgParts = msg.split(';', 4)
        if len(msgParts) != 4:
            raise ime.InvalidMessageException('Missing arguments.')

        if msgParts[3] == 'YOU\n':
            again = True
        elif msgParts[3] == 'OPP\n':
            again = False
        elif msgParts[3] == 'END\n':
            again = False
        else:
            raise ime.InvalidMessageException(f'Illegal valule for turn parameter: {msgParts[3]}')

        return again