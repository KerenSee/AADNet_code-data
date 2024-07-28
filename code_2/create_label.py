import os
import numpy as np
import torch.nn as nn
#2、生成文件名,69份分割之前
class Create_Label(nn.Module):
    def __init__(self, record_path, trail_path):
        super(Create_Label, self).__init__()
        self.record_path = record_path
        self.trial_path = trail_path

    def create_label(self):

        mask = [True if 'trial' in i else False for i in os.listdir(self.record_path)]
        txt_list = np.array(os.listdir(self.record_path))[mask]
        for txt in txt_list:
            with open(os.path.join(self.record_path,txt),encoding='utf-8') as lines:
                trail = txt.split('_')[-1].split('.')[0]
                flag = ''
                for line in lines:
                    if line.startswith('The target is'):
                        flag += '_'
                        flag += line.strip().split('is')[-1].strip()
                os.rename(self.trial_path+'trial'+trail+'.fif',self.trial_path+'trial'+trail+flag+'.fif')