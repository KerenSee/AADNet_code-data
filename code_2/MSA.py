import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn.functional as F
import torch.nn as nn

class mydataset(Dataset):
    def __init__(self,x,y):
        super(mydataset).__init__()
        self.x = x
        self.y = y
    def __len__(self):
        return len( self.x)
    def __getitem__(self, item):
        return self.x[item],self.y[item]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.head_size = input_size // num_heads

        self.query_linear = nn.Linear(input_size, input_size)
        self.key_linear = nn.Linear(input_size, input_size)
        self.value_linear = nn.Linear(input_size, input_size)

        self.output_linear = nn.Linear(input_size, input_size)
        self.fc1 = nn.Linear(32*500,1024)
        self.fc2 = nn.Linear(1024,64) #self.classs_num
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.to(torch.float32)
        batch_size, seq_len, _ = x.size()

        # Linear transformations for queries, keys, and values
        query = self.query_linear(x).view(batch_size, seq_len, self.num_heads, self.head_size)
        key = self.key_linear(x).view(batch_size, seq_len, self.num_heads, self.head_size)
        value = self.value_linear(x).view(batch_size, seq_len, self.num_heads, self.head_size)

        # Transpose to prepare for scaled dot-product attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_size ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)

        # Transpose and reshape to get the final output
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output_linear(attended_values)
        out1 = self.fc2(self.fc1(output.view(batch_size,-1)))
        out = self.fc3(out1)
        return out1, out
        # return F.softmax(out,dim=1)

# N*N 注意矩阵（0，1表示注意程度 Q*K 内积--A-（0.1，0.9）*V--注意机制--对yuan'shi'shu'ju --过滤/特征提取'滤波器/特征提取器
# QKV B--Linear--投影--QKV--增强表达能力--能表示（-----）
# FC
# DL --特征提取，FC/Linear
# （特征提取）+下游任务（分类/分割/。。。。）
# MSA 输出前后 size不变 B*32*500