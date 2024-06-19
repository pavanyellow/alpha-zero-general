import sys
sys.path.append('..')
import Arena
from MCTS import MCTS


from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.TicTacToePlayers import  RandomPlayer, GreedyPlayer, HumanTicTacToePlayer
from tictactoe.pytorch.NNet import NNetWrapper as NNet
import numpy as np
from utils import dotdict

folder = 'temp/'
best_filename = 'best.pth.tar'
second_best_filename = 'temp.pth.tar'

game = TicTacToeGame(3)

def get_player(folder, filename , temp = 1):
    n = NNet(game)
    n.load_checkpoint(folder, filename)
    args = dotdict({'numMCTSSims': 32, 'cpuct': 1.0})
    mcts = MCTS(game, n, args)
    return lambda x: np.argmax(mcts.getActionProb(x, temp=1))

# all players
random_player = RandomPlayer(game).play
greedy_player = GreedyPlayer(game).play
human_player = HumanTicTacToePlayer(game).play
best_player = get_player(folder, best_filename)
second_best_player = get_player(folder, second_best_filename)


arena = Arena.Arena(best_player, second_best_player, game, display=TicTacToeGame.display)

print(arena.playGames(100, verbose=False))