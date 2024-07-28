import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

path = 'D:/WorkSpace/批处理/OURS/'
lr_path = path + '结果 - 左右16 - 0.5.xlsx'
fm_path = path + '结果 - 男女16 - 0.5.xlsx'
lr_data = pd.DataFrame(pd.read_excel(lr_path))
fm_data = pd.DataFrame(pd.read_excel(fm_path))
for sub_num in range(1,17):
    plt.subplot(2,8, sub_num)
    lr = lr_data.iloc[5*(sub_num-1)+1:5*(sub_num-1)+6,7].to_numpy()
    fm = fm_data.iloc[5*(sub_num-1)+1:5*(sub_num-1)+6,7].to_numpy()
    print(lr)
    n = np.concatenate([lr,fm])

    d = pd.DataFrame()
    d['acc'] = n
    mix = np.concatenate([np.zeros((lr.shape[0])),np.ones((lr.shape[0]))])

    d['label'] = mix.astype(np.uint8).astype(np.str_)
    d['label'] = d['label'].apply(lambda x: x.replace(r'0', 'Orientational'))
    d['label'] = d['label'].apply(lambda x: x.replace(r'1', 'Timbre'))

    j = sns.boxplot(x='label',y= 'acc',data = d,fliersize=0,saturation=15,linewidth=2)


    j.set(ylim=(0.8,1))
    # j.set(yticks=[0,20,40])
    if sub_num%8 == 1:
        plt.xticks([])
        plt.xlabel('')
    else:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')

        # plt.gca().xaxis.set_major_formatter(plt.NullFormatter())
# plt.tight_layout()
plt.show()