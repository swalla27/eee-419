# NOTE: on Macs, there is a pesky warning that pops up with some GUI programs:
# "2019-02-02 20:12:10.627 Python[64172:11906952] ApplePersistenceIgnoreState:
# Existing state will not be touched.
# New state will be written to (null)"
# To get rid of this, execute this command on a terminal line:
# defaults write org.python.python ApplePersistenceIgnoreState NO

# Based on an example from John Keyser, updates by sdm

# NOTE: this program is meant to illustrate Python usage and is not
# necessarily the most efficient implementation.

from numpy.random import choice                  # to randomly pick replacements
import pyglet                                    # the GUI environment
import numpy as np                               # needed for randomization

BOARD_SIZE = 8                                   # board dimensions
BOARD_MAX  = BOARD_SIZE - 1                      # the maximum index
PIECE_SIZE = 50                                  # the size of each piece
NUM_PIECES = 5                                   # how many unique pieces
CHOICES = np.arange(NUM_PIECES)                  # numbers from which to choose
REPLACE = -1                                     # need to replace this entry

RND_LIMIT   = 20                                 # maximum rounds in a game
SCORE_LIMIT = 100                                # score required to win

################################################################################
# Function to both initialize the board and create replacements                #
# inputs:  board: the game board                                               #
# changes: board: updates in place                                             #
# returns: nothing                                                             #
################################################################################

def initialize(board):
    # generate as many choices as needed to replace all required new entries
    # and then replace them

    # On the right side of the equation,
    # board[board==REPLACE] returns a 1D array with n rows and 1 column
    # and len(array) returns the number of rows in the array
    # On the left side of the equation,
    # board[board==REPLACE] selects which entries should be overwritten
    # in order across rows then down columns
    board[board == REPLACE] = choice(CHOICES,len(board[board == REPLACE]))

################################################################################
# Function to remove pieces from the board if >= 3 in a row.                   #
# Note that we need to analyze the entire board before removing anything since #
# we need to check both vertical and horizontal directions as well as check    #
# for more than three in a row.                                                #
# inputs:  board: the game board                                               #
# changes: board: updates in place                                             #
# returns: whether any pieces were removed                                     #
################################################################################

def remove_pieces(board):

    # create an array in which to track items which should be removed
    remove = np.zeros([BOARD_SIZE,BOARD_SIZE],int)

    removed = False     # track whether we successfully removed something

    for row in range(0,BOARD_SIZE,1):           # start at top
        for col in range(BOARD_MAX,-1,-1):      # start at right
            # since starting at right boundary, only need to look to the left
            # but don't look if there aren't at least 2 to the left

            if ( col > 1 ):
                if ( np.array_equal(board[row,col-2:col+1],
                                    np.full(3,board[row,col])) ):
                    remove[row,col-2:col+1] = 1 # flag the entries to remove
                    removed = True              # and set the overall flag

            # since started at top, only need to look down
            # but stop if there aren't at least 2 below
            if row < BOARD_MAX-1:
                if ( np.array_equal(board[row:row+3,col],
                                    np.full(3,board[row,col])) ):
                    remove[row:row+3,col] = 1   # flag the entries to remove
                    removed = True              # and the overall flag

    # if the corresponding entry in remove is 1, replace the entry in board
    board[remove == 1] = REPLACE

    return removed    # return the flag

################################################################################
# Function to lower the pieces in each column so that the REPLACE spots are    #
# all at the top so that the initialize function can randomly replace them     #
# inputs:  board: the game board                                               #
# changes: board: updates in place                                             #
# returns: total number of pieces dropped                                      #
################################################################################

def drop_pieces(board):
    total_dropped = len(board[board == REPLACE]) # track the number dropped

    for col in range(BOARD_SIZE):                         # for each column...
        a_column = board[:,col]                           # extract a column
        keepers = a_column[a_column != REPLACE]           # the pieces to keep
        a_column[BOARD_SIZE - keepers.size:] = keepers    # slide to bottom
        a_column[0:BOARD_SIZE - keepers.size] = REPLACE   # mark rest to replace
        board[:,col] = a_column                           # put colum back in

    return total_dropped                                  # add to the score

################################################################################
# Function to update the board with a given move. First the move is done, then #
# the pieces which match are removed and, if any are, drop the pieces and then #
# replace those with random new pieces                                         #
# inputs:  board: the game board                                               #
#          move:  a move: [row1, col1, row2, col2]                             #
# changes: board: updates in place                                             #
# returns: new running score                                                   #
################################################################################

def update_board(board,move):
    score_update = 0

    # swap the two pieces
    board[move[0],move[1]], board[move[2],move[3]] = \
                                 board[move[2],move[3]], board[move[0],move[1]]

    pieces_eliminated = True
    while ( pieces_eliminated ):
        pieces_eliminated = remove_pieces(board)    # can we remove more pieces?
        if pieces_eliminated:
            score_update += drop_pieces(board)      # lower pieces to fill gaps
            initialize(board)                       # insert new pieces from top
    return score_update

################################################################################
# Function to print the game board for debug purposes                          #
# inputs:  board: the board (can be the game board or the remove board)        #
# changes: nothing                                                             #
# returns: nothing                                                             #
################################################################################

#def print_board(board):
#    for i in range(0,BOARD_SIZE):
#        myst = "row " + str(i) + ": "
#        for j in range(0,BOARD_SIZE):
#            myst += str(board[i,j])+" "
#        print(myst)

################################################################################
# Functions called when the GUI wants to draw the board.                       #
# This includes on initial startup!                                            #
################################################################################

# Initialize the widget to hold the game
window = pyglet.window.Window(width=BOARD_SIZE*PIECE_SIZE,
                              height=BOARD_SIZE*PIECE_SIZE+PIECE_SIZE,
                              caption="Matching Game")

@window.event                                   # whenever there's a draw event
def on_draw():                                  # no args passed in!
    window.clear()                              # clear the board

    for col in range(BOARD_SIZE):               # start at left
        x = PIECE_SIZE * col                    #   horizontal axis...

        for row in range(BOARD_SIZE):           # start at top
            # vertical axis: leave room for the label and make row 0 be on top!
            y = PIECE_SIZE + (PIECE_SIZE * ( BOARD_MAX - row ) )
            image_list[board[row,col]].blit(x,y)           # insert the photo

    # update the label with the latest score
    new_label = "current score is "+str(score)+" after round "+str(round)
    label = pyglet.text.Label(new_label,
                              font_name='Times New Roman',
                              font_size=12,
                              x=10, y=10)
    label.draw()

@window.event                                   # when the mouse is pressed down
def on_mouse_press(x, y, symbol, modifier):     # x,y plus unused parameters
    # unfortunately these need to be global so on_mouse_release can see them
    global start_x                              # remember these coordinates
    global start_y
    start_x = x
    start_y = y

@window.event                                   # when the mouse is released
def on_mouse_release(x, y, symbol, modifier):   # x,y plus unused parameters
    # unfortunately these need to be global so they can be maintained
    global score                                # maintain score
    global round                                # maintain round number

    # Note that we do integer math since we need the index values to access
    # the board array. The board is exactly BOARD_SIZE pieces wide so finding
    # the columns is easy.
    start_col = start_x//PIECE_SIZE
    end_col = x//PIECE_SIZE

    # The board is ( BOARD_SIZE + 1 ) pieces tall, allowing space at the
    # bottom for the label. Plus, to make things easy, row 0 is at the top!
    start_row = BOARD_MAX - (start_y - PIECE_SIZE)//PIECE_SIZE
    end_row = BOARD_MAX - (y-PIECE_SIZE)//PIECE_SIZE

    # check for legal move - don't allow moves from below the grid where the
    # label is found
    if ( ( start_row < 0 ) or ( end_row < 0 ) ):
        print("Illegal move! Please stay within the grid!")

    # check for legal move - only vertically or horizontally adjacent moves
    # are allowed
    elif ( ( ( start_col == end_col ) and ( ( start_row == end_row + 1 ) or
                                            ( start_row == end_row - 1 ) ) ) or
           ( ( start_row == end_row ) and ( ( start_col == end_col + 1 ) or
                                            ( start_col == end_col - 1 ) ) ) ):

        # put into a list to make passing easier
        move = [start_row,start_col,end_row,end_col]
        score += update_board(board,move)            # make the requested change
        round += 1                                   # track turns

        if ( score > SCORE_LIMIT ):                  # have we won?
            print("Victory: score was "+str(score)+" after round "+str(round))
            pyglet.app.exit()                        # graceful shutdown

        if ( round > RND_LIMIT ):                    # have we lost?
            print("Too many rounds ("+str(RND_LIMIT)+"): score was "+str(score))
            pyglet.app.exit()                        # graceful shutdown
    else:
        print("Illegal move! Only vertically or horizontally adjacent moves!")

################################################################################
# Start the main program                                                       #
# create the main playing board...                                             #
################################################################################

global round                                     # global so events can update
global score                                     # global so events can update

# initialize a board of all REPLACE so we can use the initial function as the
# function which also does replacements.

# board is not global, but visible to events
board = np.full([BOARD_SIZE,BOARD_SIZE],REPLACE,int)

# read the images and place them into a list.
# image files must be in the executing directory and be called piece_#.jpg,
# where the number starts from 0 and increments from there

image_list = []                             # not global, but visible to on_draw
for image in range(NUM_PIECES):
    image_list.append(pyglet.image.load("piece_"+str(image)+".jpg"))

round = 0                                   # initialize
initialize(board)                           # randomize the initial board

# a move to clear any lucky matches on startup
fake_move = [0,0,0,0]                       
score = update_board(board,fake_move)       # call to clear the lucky matches

pyglet.app.run()                            # start the game
