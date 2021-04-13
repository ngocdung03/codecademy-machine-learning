### Build Your Own Connect Four AI
from connect_four import *
def random_eval(board):
  return random.randint(-100, 100)

def my_evaluate_board(board):
  if has_won(board, "X"):
    return float("Inf")
  else:
    return -float("Inf")
  # evaluate a board if neither player has won: have more streaks of two or streaks of three than your opponent
  x_two_streak = 0
  o_two_streak = 0
  #  see if there’s the same symbol to the right
  for col in range(len(board)-1):
    for row in range(len(board[0])):
      if board[col][row]=="X" and board[col+1][row]=="X":
        x_two_streak += 1
      if board[col][row]=="O" and board[col+1][row]=="O":
        o_two_streak += 1
  return x_two_streak - o_two_streak

def two_ai_game():
    my_board = make_board()
    while not game_is_over(my_board):
      #The "X" player finds their best move.
      result = minimax(my_board, True, 4, -float("Inf"), float("Inf"), my_evaluate_board)
      print( "X Turn\nX selected ", result[1])
      print(result[1])
      select_space(my_board, result[1], "X")
      print_board(my_board)

      if not game_is_over(my_board):
        #The "O" player finds their best move
        result = minimax(my_board, False, 4, -float("Inf"), float("Inf"), random_eval)
        print( "O Turn\nO selected ", result[1])
        print(result[1])
        select_space(my_board, result[1], "O")
        print_board(my_board)
    if has_won(my_board, "X"):
        print("X won!")
    elif has_won(my_board, "O"):
        print("O won!")
    else:
        print("It's a tie!")
# new_board = make_board()
# select_space(new_board, 6, "X")
# select_space(new_board, 7, "X")
# select_space(new_board, 1, "O")
# select_space(new_board, 2, "O")
# select_space(new_board, 6, "O")
# print_board(new_board)
# print(my_evaluate_board(new_board))
two_ai_game()

# Check for streaks in all eight directions.
# Weight streaks of three higher than streaks of two.
# Only count streaks if your opponent hasn’t blocked the possibility of a Connect Four. For example, if there’s an "X" streak of two to the right, but the next column over has an "O", then that streak is useless.
# Only count streaks if there’s enough board space to make a Connect Four. For example, there’s no need to check for left streaks of two in the second column from the left. Even if there is a streak of two, you can’t make a Connect Four going to the left, so the streak is useless.