#处理数据的脚本，整合了数据marker识别，create_label，数据切分等
import os
import mne
import torch.nn as nn
import numpy as np
from data_processing import Data_Processing

class data_process(nn.Module):
    def __init__(self, lr_flag, fm_flag, PathName, output_directory, num_segments):
        super(data_process, self).__init__()
        self.PathName = PathName
        self.output_directory = output_directory
        self.num_segments = num_segments
        self.lr_flag = lr_flag
        self.fm_flag = fm_flag
