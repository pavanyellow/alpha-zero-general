import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet

from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.TicTacToePlayers import *
from tictactoe.keras.NNet import NNetWrapper as NNet
import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = False

g = TicTacToeGame(3)

# all players
rp = RandomPlayer(g).play
hp = HumanTicTacToePlayer(g).play



# nnet players
n1 = NNet(g)
import os
folder = './pretrained_models/tictactoe/keras/'
filename = 'best-25eps-25sim-10epch.h5'
filepath = os.path.join(folder, filename)
if not os.path.exists(filepath):
    print("Checkpoint Directory does not exist! Making directory {}".format(folder))
n1.nnet.model.load_weights(filepath)
n1.load_checkpoint('./pretrained_models/tictactoe/keras/', 'best-25eps-25sim-10epch.h5')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=1))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./pretrained_models/tictactoe/keras/', 'best-25eps-25sim-10epch.h5')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=1))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=TicTacToeGame.display)

print(arena.playGames(2, verbose=True))