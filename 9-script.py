### Minimax
from tic_tac_toe import *

my_board = [
	["1", "2", "X"],
	["4", "5", "6"],
	["7", "8", "9"]
]
print_board(my_board)
select_space(my_board, 5, "O")
print_board(my_board)
select_space(my_board, 7, "X")
select_space(my_board, 8, "O")
print_board(my_board)
print(available_moves(my_board))
select_space(my_board, 1, "X")
select_space(my_board, 2, "X")
print(has_won(my_board, "X"))
print(has_won(my_board, "O"))

## Detecting Tic-Tac-Toe Leaves
from tic_tac_toe import *
start_board = [
	["1", "2", "3"],
	["4", "5", "6"],
	["7", "8", "9"]
]

x_won = [
	["X", "O", "3"],
	["4", "X", "O"],
	["7", "8", "X"]
]

o_won = [
	["O", "X", "3"],
	["O", "X", "X"],
	["O", "8", "9"]
]

tie = [
	["X", "X", "O"],
	["O", "O", "X"],
	["X", "O", "X"]
]
def game_is_over(board):
  if has_won(board,"X") or has_won(board,"O") or len(available_moves(board))==0:
    return True
  return False
print(game_is_over(start_board))
print(game_is_over(x_won))
print(game_is_over(o_won))
print(game_is_over(tie))

def evaluate_board(board):
  if game_is_over(board):
    if has_won(board, "X"):
      return 1
    elif has_won(board, "O"):
      return -1
    else:
      return 0
print(evaluate_board(start_board))
print(evaluate_board(x_won))
print(evaluate_board(o_won))
print(evaluate_board(tie))

## Copying Boards
from tic_tac_toe import *
from copy import deepcopy

my_board = [
	["1", "2", "X"],
	["4", "5", "6"],
	["7", "8", "9"]
]
# new_board = my_board
# select_space(new_board, 5, "O")
# print_board(new_board)
# print_board(my_board) #same

new_board = deepcopy(my_board)
select_space(new_board, 5, "O")
print_board(new_board)
print_board(my_board) #diff

## The Minimax function
from tic_tac_toe import *
from copy import deepcopy

x_winning = [
	["X", "2", "O"],
	["4", "O", "6"],
	["7", "8", "X"]
]

def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def evaluate_board(board):
  if has_won(board, "X"):
    return 1
  elif has_won(board, "O"):
    return -1
  else:
    return 0

def minimax(input_board, is_maximizing):
  # Base case - the game is over, so we return the value of the board
  if game_is_over(input_board):
    return evaluate_board(input_board)
  if is_maximizing:
    best_value = -float("Inf")   #Right now, we haven’t looked at any moves, so we should start best_value at something lower than the lowest possible value — -float("Inf").
  else:
    best_value = float("Inf")
  for move in available_moves(input_board):
    #print(move)
    new_board = deepcopy(input_board)
    symbol = "X" if is_maximizing else "O"
    select_space(new_board, move, symbol)
  return new_board
  #return best_value
print(minimax(x_winning, True))

## Recursion In Minimax
from tic_tac_toe import *
from copy import deepcopy

def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def evaluate_board(board):
  if has_won(board, "X"):
    return 1
  elif has_won(board, "O"):
    return -1
  else:
    return 0

new_game = [
	["1", "2", "3"],
	["4", "5", "6"],
	["7", "8", "9"]
]

x_winning = [
	["X", "2", "O"],
	["4", "O", "6"],
	["7", "8", "X"]
]

o_winning = [
	["X", "X", "O"],
	["4", "X", "6"],
	["7", "O", "O"]
]

def minimax(input_board, is_maximizing):
  # Base case - the game is over, so we return the value of the board
  if game_is_over(input_board):
    return evaluate_board(input_board)
  if is_maximizing == True:
    best_value = -float("Inf")
    symbol = "X"
  else:
    best_value = float("Inf")
    symbol = "O"
  for move in available_moves(input_board):
    new_board = deepcopy(input_board)
    select_space(new_board, move, symbol)
    # make recursive call
    hypothetical_value = minimax(new_board, not is_maximizing)
    if is_maximizing and hypothetical_value > best_value:
      best_value = hypothetical_value 
    elif not is_maximizing and hypothetical_value < best_value:
      best_value = hypothetical_value 
  return best_value

print(minimax(x_winning, True))
print(minimax(o_winning, True))
print(minimax(new_game, True))
  
## Which Move?
# keeping track of what move will cause that
from tic_tac_toe import *
from copy import deepcopy

def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def evaluate_board(board):
  if has_won(board, "X"):
    return 1
  elif has_won(board, "O"):
    return -1
  else:
    return 0

new_game = [
	["1", "2", "3"],
	["4", "5", "6"],
	["7", "8", "9"]
]

x_winning = [
	["X", "2", "O"],
	["4", "O", "6"],
	["7", "8", "X"]
]

o_winning = [
	["X", "X", "O"],
	["4", "X", "6"],
	["7", "O", "O"]
]

def minimax(input_board, is_maximizing):
  # Base case - the game is over, so we return the value of the board
  if game_is_over(input_board):
    return [evaluate_board(input_board), ""]
  best_move = ""
  if is_maximizing == True:
    best_value = -float("Inf")
    symbol = "X"
  else:
    best_value = float("Inf")
    symbol = "O"
  for move in available_moves(input_board):
    new_board = deepcopy(input_board)
    select_space(new_board, move, symbol)
    hypothetical_value = minimax(new_board, not is_maximizing)[0]
    if is_maximizing == True and hypothetical_value > best_value:
      best_value = hypothetical_value
      best_move = move
    if is_maximizing == False and hypothetical_value < best_value:
      best_value = hypothetical_value
      best_move = move
  return [best_value, best_move]

# This should return [1, 7]. This means that "X" should be able to win the game if they select move 7.
print(minimax(x_winning, True))
# This should return [-1, 4]. This means that no matter what "X" does, "O" will win. "X" might as well select move 4.
print(minimax(o_winning, True))

## Play a Game
# This line of code instructs the AI to make a move as the "X" player:
# select_space(my_board, minimax(my_board, True)[1], "X")
from tic_tac_toe import *

my_board = [
	["1", "2", "3"],
	["4", "5", "6"],
	["7", "8", "9"]
]

#while not game_is_over(my_board):
#  select_space(my_board, minimax(my_board, True)[1], "X")
#  print_board(my_board)
#  if not game_is_over(my_board):
#    select_space(my_board, minimax(my_board, False)[1], "O")
#    print_board(my_board)  

# Take some time to really understand all of the parameters. Why do we pass True to minimax()? Why do we use [1] at the end of minimax()?