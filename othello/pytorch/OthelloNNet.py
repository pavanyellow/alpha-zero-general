import sys
sys.path.append('..')
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..OthelloGame import OthelloGame

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn(self.fc(x)))
        return out + residual

class OthelloNNet(nn.Module):
    def __init__(self, game: OthelloGame, args, num_layers=4, hidden_dim=256):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(OthelloNNet, self).__init__()

        self.fc1 = nn.Linear(self.board_x * self.board_y, hidden_dim)
        self.fc_bn1 = nn.BatchNorm1d(hidden_dim)

        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_layers)])
        
        self.policy_head = nn.Linear(hidden_dim, self.action_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, s: torch.Tensor):
        s = s.view(-1, self.board_x * self.board_y)
        s = F.relu(self.fc_bn1(self.fc1(s)))

        for res_block in self.res_blocks:
            s = res_block(s)

        policy = self.policy_head(s)
        value = self.value_head(s)

        return F.log_softmax(policy, dim=1), torch.tanh(value)
