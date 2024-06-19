import numpy as np

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(i + 1, end=" ")
        print()
        while True:
            input_move = input("Enter your move (1-9): ")
            if input_move.isdigit():
                a = int(input_move) - 1
                if 0 <= a < self.game.n ** 2 and valid[a]:
                    break
            print('Invalid move')
        return a


class GreedyPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = np.where(valids == 1)[0]

        # Check for a winning move
        for a in candidates:
            next_board, _ = self.game.getNextState(board, 1, a)
            if self.game.getGameEnded(next_board, 1):
                return a

        # Check for a blocking move
        for a in candidates:
            next_board, _ = self.game.getNextState(board, -1, a)
            if self.game.getGameEnded(next_board, -1):
                return a

        # Fallback to a random move
        return np.random.choice(candidates)