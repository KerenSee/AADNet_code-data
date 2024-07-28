

import torch
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pdb
import math

import torch
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pdb
import math
from transformer import Transformer


from configParameter import Config
from sklearn.neural_network import MLPClassifier
class BFE(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes
        self.indim = (self.nodes*(self.nodes-1))//2
        self.mlp = nn.Sequential(
            nn.Linear(self.indim,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512 ,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128 ,32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.config = Config()
        self.transformer = Transformer(self.config)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        # self.conv1 = nn.Conv1d()
        self.linear1 = nn.Linear(500 ,128)
        self.linear2 = nn.Linear(128 ,64)
        self.linear3 = nn.Linear(96 ,2)

    def forward(self, x ,adj):
        bz, roi, ts = x.size()

        features = self.mlp(adj.to(torch.float32))

        x = x.to(torch.float32)

        x = self.transformer(x)
        # print(x.shape)
        x = self.avgpool(x.permute(0, 2, 1))  # 1,120
        # print(x.shape)
        x = self.linear1(x.permute(0, 2, 1))
        x = torch.squeeze(self.linear2(x))
        # print(x.size(),features.size())
        x0 = self.linear3(torch.cat([x, features], dim=1))
        return torch.cat([x, features], dim=1), x0
