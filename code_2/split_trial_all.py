import mne
import numpy as np
import os
import torch.nn as nn


# 将16*32*34500的矩阵分成16个32*34500的矩阵，然后取每个矩阵的前69秒数据
class Split_Trial_All(nn.Module):
    def __init__(self, original_path, original_file_path, trial_path):
        super(Split_Trial_All, self).__init__()
        self.original_path = original_path
        self.original_file_path = original_file_path
        self.trial_path = trial_path

    def split_trial_all(self):
        raw = mne.io.read_raw_eeglab(self.original_file_path, preload=True)
        channels_list = ['P8', 'T8', 'CP6', 'FC6', 'F8', 'F4', 'C4', 'P4', 'AF4', 'Fp2', 'Fp1', 'AF3', 'Fz', 'FC2',
                         'Cz', 'CP2', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'Pz', 'CP1', 'FC1', 'P3', 'C3', 'F3', 'F7', 'FC5',
                         'CP5', 'T7', 'P7']
        channels = 32
        num_segments = 16
        freq = 500
        second_segments = 69
        raw.pick_channels(ch_names=channels_list)
        data = raw.get_data()[:channels, :]
        print(data.shape)

        if not os.path.exists(self.trial_path):
            os.makedirs(self.trial_path)

        info = mne.create_info(ch_names=channels_list, sfreq=500, ch_types='eeg')
        for i in range(num_segments):

            trial_file_path = os.path.join(self.trial_path, f'trial{i + 1}.fif')


            start_idx = i * data.shape[1] // num_segments
            end_idx = (i + 1) * data.shape[1] // num_segments
            eeg_data_segment = data[:, start_idx:end_idx]
            eeg_data = eeg_data_segment[:, :second_segments * freq]


            if os.path.exists(trial_file_path):
                os.remove(trial_file_path)
            mne.io.RawArray(eeg_data, info).save(trial_file_path, overwrite=True)

        return num_segments

