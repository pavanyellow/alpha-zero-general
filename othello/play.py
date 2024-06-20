import sys
sys.path.append('..')
import Arena
from MCTS import MCTS

from othello.OthelloGame import OthelloGame as Game
from othello.OthelloPlayers import  RandomPlayer, GreedyOthelloPlayer as GreedyPlayer, HumanOthelloPlayer as HumanPlayer
from othello.pytorch.NNet import NNetWrapper as NNet
import numpy as np
from utils import dotdict

folder = 'temp/'
best_filename = 'best.pth.tar'
second_best_filename = 'temp.pth.tar'

game = Game(6)

def get_player(folder, filename , temp = 1):
    n = NNet(game)
    n.load_checkpoint(folder, filename)
    args = dotdict({'numMCTSSims': 32, 'cpuct': 1.0})
    mcts = MCTS(game, n, args)
    return lambda x: np.argmax(mcts.getActionProb(x, temp=temp))

# all players
random_player = RandomPlayer(game).play
greedy_player = GreedyPlayer(game).play
human_player = HumanPlayer(game).play
best_player = get_player(folder, best_filename)
second_best_player = get_player(folder, second_best_filename)


arena = Arena.Arena(best_player, random_player, game, display=Game.display)

print(arena.playGames(100, verbose=False, print_final_board=True))