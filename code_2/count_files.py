import os
import torch.nn as nn
class Count_Files(nn.Module):
    def __init__(self, trial_path):
        super(Count_Files, self).__init__()
        self.trial_path = trial_path

    def count_files(self):
        trial_count = 0
        for files in os.walk(self.trial_path):
            trial_count += len(files)
        return trial_count