import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_processing import Data_Processing
from EEGNet_recode import EEGNet
from torch.utils.data import DataLoader,Dataset
from EEGNet_recode import mydataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from train_eegnet import train_test
import os

path = 'D:/WorkSpace/实验数据/被试3/data/预处理后/'
if not os.path.exists(path + 'ProcessedData_New/'):
    os.mkdir(path + 'ProcessedData_New/')
Data_Processing(True, False, PathName = path+'trials/',output_directory = path+'ProcessedData_New/',num_segments = 69).Data_processing()