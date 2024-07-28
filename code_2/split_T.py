import mne
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
#1、将marker对应trial分开,分成16份
class Split_Trial(nn.Module):
    def __init__(self, original_path, original_file_path, trial_path):
        super(Split_Trial,self).__init__()
        self.original_path = original_path
        self.original_file_path = original_file_path
        self.trial_path = trial_path

    def split_trial(self, flag=None): # flag None original flag 'after' after

        channels_list = ['P8', 'T8', 'CP6', 'FC6', 'F8', 'F4', 'C4', 'P4', 'AF4', 'Fp2', 'Fp1', 'AF3', 'Fz', 'FC2',
                         'Cz', 'CP2', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'Pz', 'CP1', 'FC1', 'P3', 'C3', 'F3', 'F7', 'FC5',
                         'CP5', 'T7', 'P7']
        channels = 32
        num_segments = 69
        freq = 500

        if flag:
            # split 16 files
            # segment
            raw = mne.io.read_epochs_eeglab(self.original_file_path)
            # print(raw.get_data().shape)
            raw.pick_channels(ch_names=channels_list)
            eeg_data = raw.get_data()[:, :channels, :freq*num_segments]  # N,C,T
        else:
            raw = mne.io.read_raw_eeglab(self.original_file_path, preload=True)
            raw.plot(start=5, duration=5)
            plt.show()
            raw.pick_channels(ch_names=channels_list)
            data = raw.get_data()[:channels, :]
            events = mne.events_from_annotations(raw)[0]
            time_marker = events[:,0]
            # data_list = np.zeros((trails,channels,freq*num_segments))
            # for i,start in zip(np.arange(0,trails),time_marker):
            #     if i%2==0:
            #         data_list[i] = data[:,start:start+freq*num_segments]
            mask = [True if i%2==0 else False for i in range(time_marker.shape[0])]
            eeg_data = np.array([data[:,start:start+num_segments*freq] for start in time_marker[mask]])
            print(eeg_data.shape) # N,C,T

        if not os.path.exists(self.trial_path):
            os.mkdir(self.trial_path)
        else:
            print('trails exist')
        info = mne.create_info(ch_names=channels_list, sfreq=500, ch_types='eeg')
        for i in range(eeg_data.shape[0]):
            out_path = self.trial_path + 'trial'+str(i+1)+'.fif'
            if os.path.exists(out_path):
                os.remove(out_path)
            mne.io.RawArray(eeg_data[i], info).save(out_path, overwrite=True)
            # raw = mne.io.read_raw_fif(path+'trials/output_trial'+str(i)+'.fif',preload=True)
        return eeg_data.shape[0]

