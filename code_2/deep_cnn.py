# 导入工具包
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader,Dataset

class DeepCNN(nn.Module): # 3 layers
    def __init__(self, classes_num):
        super(DeepCNN, self).__init__()
        self.drop_out = 0.25

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=32,  # num_filters
                kernel_size=(3, 3),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(32),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 2)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )
        self.block_11 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=32,  # input shape (1, C, T)
                out_channels=32,  # num_filters
                kernel_size=(3, 3),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(32),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 2)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )
        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,  # input shape (8, C, T)
                out_channels=64,  # num_filters
                kernel_size=(3, 3),  # filter size 有问题  (32, 1)
                # groups=32,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(64),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 2)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )
        self.block_22 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,  # input shape (8, C, T)
                out_channels=64,  # num_filters
                kernel_size=(3, 3),  # filter size 有问题  (32, 1)
                # groups=32,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(64),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 2)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=64,  # input shape (16, 1, T//4)
                out_channels=64,  # num_filters
                kernel_size=(3, 3),  # filter size
                # groups=64,
                bias=False
            ),  # output shape (16, 1, T//
            nn.BatchNorm2d(64),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.Conv2d(
                in_channels=64,  # input shape (16, 1, T//4)
                out_channels=64,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(64),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear(64*40, 64)   #0.1 64*40   0.5  64*60  1.0  64*80  1.5 64*100
        self.out2 = nn.Linear(64, classes_num)
        # self.out = nn.Linear(1232, classes_num)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.block_1(x)
        x = self.block_11(x)
        # x = self.block_11(x)
        # x = self.block_11(x)
        # print("block1", x.shape)
        x = self.block_2(x)
        x = self.block_22(x)
        x = self.block_22(x)
        # x = self.block_22(x)
        # print("block2", x.shape)
        x = self.block_3(x)
        # print("block3", x.shape)
        x = x.view(x.size(0), -1)
        #x0 = x
        # print('jjjjjjj')
        # print(x.shape)
        x = self.out(x)
        x0=x
        x = self.out2(x)
        # return F.softmax(x, dim=1), x  # return x for visualization
        return x0, x
