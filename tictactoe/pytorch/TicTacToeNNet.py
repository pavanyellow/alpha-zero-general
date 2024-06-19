import sys
sys.path.append('..')
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..TicTacToeGame import TicTacToeGame

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn(self.fc(x)))
        return out + residual

class TicTacToeNNet(nn.Module):
    def __init__(self, game : TicTacToeGame, args, residual_blocks=4, hidden_dim=64):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(TicTacToeNNet, self).__init__()
        
        # Input layer
        self.fc1 = nn.Linear(self.board_x * self.board_y, hidden_dim)
        self.fc_bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        self.res_block3 = ResidualBlock(hidden_dim)
        self.res_block4 = ResidualBlock(hidden_dim)
        
        # Output layers
        self.policy_head = nn.Linear(hidden_dim, self.action_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, s: torch.Tensor):
        #print(f"early shape: {s.shape}")  
        s = s.view(-1, self.board_x * self.board_y)
        #print(f"Input shape: {s.shape}")  
        
        s = F.relu(self.fc_bn1(self.fc1(s)))
        #print(f"FC1 shape: {s.shape}")         
        
        s = self.res_block1(s)                  
        s = self.res_block2(s)
        s = self.res_block3(s)
        s = self.res_block4(s)

        policy = self.policy_head(s)

        value = self.value_head(s)                              

        return F.log_softmax(policy, dim=1), torch.tanh(value)

