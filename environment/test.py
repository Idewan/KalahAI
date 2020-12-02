import numpy as np

def board_view_player(board, curr_turn):
    if curr_turn == "SOUTH":
        board[[0,1]] = board[[1,0]]
        print("SWAPPED")
    return board

if __name__ == "__main__":
    player1="NORTH"
    player2="SOUTH"

    old_turn = player1
    curr_turn = player2
    next_turn = player1

    state = np.random.randint(8,size=(2,8))
    next_state = state.copy() + 8
    print("OUTPUT OF ACTION NEXTSTATE NORTH BIAS")
    print(next_state)
    state[[0,1]] = state[[1,0]]
    print("PLAYER 2 CURRENT STATE")
    print(state)
    
    
    #^Next state is always in the view of north player 
    #Next state should be in the view of the current player for the memory
    next_state_curr = next_state.copy()
    next_state_curr = board_view_player(next_state_curr, curr_turn)
    print("PLAYER 2 NEXT STATE")
    print(next_state_curr)
        
    #Store in memory
    print(state, next_state_curr)
        
    old_turn = curr_turn
    curr_turn = next_turn

    #If the turn is the same i.e. old_turn == curr_turn
    #then we do not swap the board we just change state -> next state
    if old_turn != curr_turn:
        state = board_view_player(next_state, curr_turn)
        print("PLayer 1 NEXT STATE")
        print(state)
    else:
        state = next_state_curr
