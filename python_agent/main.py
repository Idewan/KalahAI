import protocol as p
import board as b
import kalah as k
import side as s

def sendMsg( msg):
    print(msg, flush=True)


def recvMsg():
    msg = input()
    msg += '\n'

    if msg is None:
        raise EOFError('Input ended unexpectedly.')

    return msg


def main():
    print("Hello World!")

if __name__ == "__main__":
    main()