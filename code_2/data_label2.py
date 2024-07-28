import os
import mne
import torch.nn as nn
import numpy as np
import torch
class Data_Label(nn.Module):
    def __init__(self, output_directory, lr_flag, fm_flag):
        super(Data_Label, self).__init__()
        self.output_directory = output_directory
        self.lr_flag = lr_flag
        self.fm_flag = fm_flag

    def data_label(self):
        dataset_list = []
        label_list = []
        file_paths = [os.path.join(self.output_directory, file) for file in os.listdir(self.output_directory) if
                      file.endswith('.fif')]
        channels_list = ['P8', 'T8', 'CP6', 'FC6', 'F8', 'F4', 'C4', 'P4', 'AF4', 'Fp2', 'Fp1', 'AF3', 'Fz', 'FC2',
                         'Cz', 'CP2', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'Pz', 'CP1', 'FC1', 'P3', 'C3', 'F3', 'F7', 'FC5',
                         'CP5', 'T7', 'P7']
        # self.data_processing()
        if self.fm_flag:
            flag = 'female'
        elif self.lr_flag:
            flag = 'right'
        for file_path in file_paths:
            raw = mne.io.read_raw_fif(file_path, preload=True)
            raw.pick_channels(ch_names=channels_list)
            data = raw.get_data()[:32, :]
            dataset_list.append(data)
            if flag in file_path:
                label_list.append(0)
            else:
                label_list.append(1)
        print(label_list)

        return np.array(dataset_list), np.array(label_list), file_paths