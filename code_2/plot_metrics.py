import os.path

import sklearn.metrics as metrics
from PIL import Image
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.manifold import TSNE
import mne
import numpy as np
from scipy.signal import butter, filtfilt
import io
from datetime import datetime
from scipy.stats import zscore


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    fa = 0.5 * fs
    low = lowcut / fa
    high = highcut / fa
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def get_freq_band(data): # trials, C, T
    # print('hhhhhh',data.shape)
    alpha = np.average(butter_bandpass_filter(data, 8, 13, 500, order=4), axis=2)
    beta = np.average(butter_bandpass_filter(data,14, 30, 500, order=4), axis=2)
    sigma = np.average(butter_bandpass_filter(data,0.5, 3, 500, order=4), axis=2)
    theta = np.average(butter_bandpass_filter(data,4, 7, 500, order=4), axis=2)
    # print('hhhhhhhhhhhhhhh',alpha.shape)  # 32,34570
    return np.array([alpha, beta, sigma, theta])

def show_topo(data , i, j, axs, avg_flag = False, vmin = 0 , vmax = 1 , contours = 6 , cmap = 'bwr'):
    channels_list = ['P8', 'T8', 'CP6', 'FC6', 'F8', 'F4', 'C4', 'P4', 'AF4', 'Fp2', 'Fp1', 'AF3', 'Fz', 'FC2', 'Cz', 'CP2', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'Pz', 'CP1', 'FC1', 'P3', 'C3', 'F3', 'F7', 'FC5', 'CP5', 'T7', 'P7']
    data = data.reshape(len(channels_list), 1)
    info = mne.create_info(ch_names=channels_list, sfreq=500, ch_types='eeg')
    evoked = mne.EvokedArray(data, info)
    evoked.set_montage('biosemi32')
    # ax, countour =
    if avg_flag:
        ax = axs[i,j]
    else:
        ax = axs[j]
    im, _ = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, show=False, cmap=cmap, contours=contours, axes=ax)
    return im


def plot_brain_grandplot2(d, label, plot_flag, original_path, trials_n=10):
    if not os.path.exists(original_path + 'result/'):
        os.mkdir(original_path + 'result/')
    c = get_freq_band(d)  # (4, 10, 34750)
    print('hhhhhhhhhhhhhhhhhhhhh', c.shape)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    band_name = ['alpha', 'beta', 'sigma', 'theta']
    if plot_flag == 'female':
        events_name = ["female", "male"]
        num_list = label[np.array(['female' in i or 'Female' in i for i in label])]
    else:
        events_name = ["right", "left"]
        num_list = label[np.array(['right' in i or 'Right' in i for i in label])]
    for event in events_name:
        # fig, ax = plt.subplots()
        # plt.figure(figsize=(10, 20), dpi=300)

        if event == events_name[0]:
            trials_num = len(num_list)
        else:
            trials_num = trials_n - len(num_list)
        print(trials_num)

        fig, axs = plt.subplots(trials_num, len(band_name), figsize=(9, 12))

        for j in range(trials_num):  # trials num
            for i in range(len(band_name)):  # freq band num
                data = c[i, j, :]
                # plt.subplot(trials_num, len(band_name), j * len(band_name) + i + 1)
                show_topo(data, j, i, axs)
                if j == 0:
                    axs[j, i].set_title(band_name[i], fontsize=15)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        plt.suptitle(event + " subject freq-band grangplot", fontsize="x-large")
        cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])  # 调整颜色条位置
        fig.colorbar(axs[0, 0].collections[0], cax=cbar_ax)
        # plt.savefig(original_path + 'result/' + event + "_subject_freq-band_grangplot.png", dpi=300)
        # plt.show()


def plot_brain_grandplot(d, label, plot_flag, original_path, trials_n=10):
    if not os.path.exists(original_path + 'result/'):
        os.mkdir(original_path + 'result/')
        # (4, 10, 32)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    band_name = ['alpha', 'beta', 'sigma', 'theta']
    if plot_flag == 'female':
        events_name = ["female", "male"]
        num_list = label[np.array(['female' in i or 'Female' in i for i in label])]
        num_list2 = label[~np.array(['female' in i or 'Female' in i for i in label])]
        trials_list = np.array(['female' in i or 'Female' in i for i in label])
        index = 1
    else:
        events_name = ["right", "left"]
        num_list = label[np.array(['right' in i or 'Right' in i for i in label])]
        num_list2 = label[~np.array(['right' in i or 'Right' in i for i in label])]
        trials_list = np.array(['right' in i or 'Right' in i for i in label])
        index = -1

    for event in events_name:
        if event == events_name[0]:
            trials_num = len(num_list)
            c = get_freq_band(d[trials_list])
        else:
            trials_num = trials_n - len(num_list)
            c = get_freq_band(d[~trials_list])
            num_list = num_list2
        print(trials_num)
        fig, axs = plt.subplots(trials_num, len(band_name), figsize=(9, 12))
        min, max = [], []
        for j in range(trials_num):  # trials num
            for i in range(len(band_name)):  # freq band num
                data = c[i, j, :]
                # plt.subplot(trials_num, len(band_name), j * len(band_name) + i + 1)
                im = show_topo(data, j, i, axs, avg_flag=True)
                if j == 0:
                    axs[j, i].set_title(band_name[i], fontsize=15)
                if i == 0:
                    axs[j, i].set_ylabel(
                        "%s" % ('trials' + num_list[j].split('-')[0] + '-' + num_list[j].split('-')[index]),
                        rotation=90, size='large')
                vmin, vmax = im.get_clim()
                min.append(vmin)
                max.append(vmax)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        plt.suptitle(event + " subject freq-band grangplot", fontsize="x-large")
        cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])  # 调整颜色条位置
        # cbar =fig.colorbar(axs[0, 0].collections[0], cax=cbar_ax)

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=im.get_cmap())
        sm.set_array([])
        plt.colorbar(sm, cax=cbar_ax)
        plt.suptitle(event + " subject freq-band grangplot", fontsize="x-large")
        plt.savefig(original_path + 'result/' + event + "_subject_freq-band_grangplot.png")
        # cbar.set_clim(np.array(min).min(), np.array(max).max())
        # plt.show()

    # average
    j = 0
    trials_num = 2
    fig, axs = plt.subplots(trials_num, len(band_name), figsize=(9, 6))
    min, max = [], []
    for event in events_name:
        if event == events_name[0]:
            trials_num = len(num_list)
            c = get_freq_band(d[trials_list])
        else:
            trials_num = trials_n - len(num_list)
            c = get_freq_band(d[~trials_list])
        # print(trials_num)
        for i in range(len(band_name)):  # freq band num
            data = np.average(c[i, :, :], axis=0)
            # print(data.shape)
            # plt.subplot(trials_num, len(band_name), j * len(band_name) + i + 1)
            im = show_topo(data, j, i, axs, avg_flag=True)
            if j == 0:
                axs[j, i].set_title(band_name[i], fontsize=15)
            if i == 0:
                axs[j, i].set_ylabel("%s" % (event), rotation=90, size='large')
            vmin, vmax = im.get_clim()
            min.append(vmin)
            max.append(vmax)
        j += 1
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.suptitle( " subject average freq-band grangplot", fontsize="x-large")
    cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=im.get_cmap())
    sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax)
    # cbar.set_clim(np.array(min).min(), np.array(max).max())

    # plt.savefig(original_path + 'result/' + "average_subject_freq-band_grangplot.png", dpi=300)
    # plt.show()

    j = 0
    trials_num = 1
    fig, axs = plt.subplots(trials_num, len(band_name), figsize=(9, 6))
    min, max = [], []
    for event in events_name:
        if event == events_name[0]:
            trials_num = len(num_list)
            c1 = get_freq_band(d[trials_list])
        else:
            trials_num = trials_n - len(num_list)
            c2 = get_freq_band(d[~trials_list])
    axs[0].set_ylabel("%s" % (events_name[0] + '-' + events_name[1]), rotation=90, size='large')
    for i in range(len(band_name)):  # freq band num
        data = np.average(c1[i, :, :], axis=0) - np.average(c2[i, :, :], axis=0)
        # print(data.shape)
        # plt.subplot(trials_num, len(band_name), j * len(band_name) + i + 1)
        im = show_topo(data, j, i, axs)
        axs[i].set_title(band_name[i], fontsize=15)
        vmin, vmax = im.get_clim()
        min.append(vmin)
        max.append(vmax)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.suptitle(" subject sub freq-band grangplot", fontsize="x-large")
    cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=im.get_cmap())
    sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax)
    # cbar.set_clim(np.array(min).min(), np.array(max).max())

    # plt.savefig(original_path + 'result/' + "sub_subject_freq-band_grangplot.png", dpi=300)
    # plt.show()



# def plot_brain_grandplot(d, label, plot_flag, original_path, trials_n=10):
#     if not os.path.exists(original_path + 'result/'):
#         os.mkdir(original_path + 'result/')
#     c = get_freq_band(d)  # (4, 10, 34750)
#     print('hhhhhhhhhhhhhhhhhhhhh', c.shape)
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
#     band_name = ['alpha', 'beta', 'sigma', 'theta']
#     if plot_flag == 'female':
#         events_name = ["female", "male"]
#         num_list = label[np.array(['female' in i for i in label])]
#         index = 1
#     else:
#         events_name = ["right", "left"]
#         num_list = label[np.array(['right' in i for i in label])]
#         index = -1
#
#     for event in events_name:
#         if event == events_name[0]:
#             trials_num = len(num_list)
#         else:
#             trials_num = trials_n - len(num_list)
#         print(trials_num)
#
#         fig, axs = plt.subplots(trials_num, len(band_name), figsize=(9, 12))
#         min, max = [], []
#         for j in range(trials_num):  # trials num
#             for i in range(len(band_name)):  # freq band num
#                 data = c[i, j, :]
#                 # plt.subplot(trials_num, len(band_name), j * len(band_name) + i + 1)
#                 im = show_topo(data, j, i, axs)
#                 if j == 0:
#                     axs[j, i].set_title(band_name[i], fontsize=15)
#                 if i == 0:
#                     axs[j, i].set_ylabel(
#                         "%s" % ('trial-' + num_list[j].split('-')[0] + '-' + num_list[j].split('-')[index]),
#                         rotation=90, size='large')
#                 vmin, vmax = im.get_clim()
#                 min.append(vmin)
#                 max.append(vmax)
#         fig.subplots_adjust(hspace=0.1, wspace=0.1)
#         plt.suptitle(event + " subject freq-band grangplot", fontsize="x-large")
#         cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])
#         norm = plt.Normalize(vmin=vmin, vmax=vmax)
#         sm = plt.cm.ScalarMappable(norm=norm, cmap=im.get_cmap())
#         sm.set_array([])
#         plt.colorbar(sm, cax=cbar_ax)
#         # cbar.set_clim(np.array(min).min(), np.array(max).max())
#         plt.savefig(original_path + 'result/' + event + "_subject_freq-band_grangplot.png", dpi=300)
#         plt.show()
#
#         # plt.suptitle(event + " subject freq-band grangplot", fontsize="x-large")
#         # plt.savefig(original_path+'result/' + event + "_subject_freq-band_grangplot.png")
#
#     # average
#     for event in events_name:
#         trials_num = 1
#         fig, axs = plt.subplots(trials_num, len(band_name), figsize=(9, 12))
#         min, max = [], []
#         for j in range(trials_num):  # trials num
#             for i in range(len(band_name)):  # freq band num
#                 data = np.average(c[i, :, :],axis=0)
#                 print(data.shape)
#                 # plt.subplot(trials_num, len(band_name), j * len(band_name) + i + 1)
#                 im = show_topo(data, j, i, axs)
#                 if j == 0:
#                     axs[j, i].set_title(band_name[i], fontsize=15)
#                 if i == 0:
#                     axs[j, i].set_ylabel(
#                         "%s" % ('trial-' + num_list[j].split('-')[0] + '-' + num_list[j].split('-')[index]),
#                         rotation=90, size='large')
#                 vmin, vmax = im.get_clim()
#                 min.append(vmin)
#                 max.append(vmax)
#         fig.subplots_adjust(hspace=0.1, wspace=0.1)
#         plt.suptitle(event + " subject freq-band grangplot", fontsize="x-large")
#         cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])
#         norm = plt.Normalize(vmin=vmin, vmax=vmax)
#         sm = plt.cm.ScalarMappable(norm=norm, cmap=im.get_cmap())
#         sm.set_array([])
#         plt.colorbar(sm, cax=cbar_ax)
#         # cbar.set_clim(np.array(min).min(), np.array(max).max())
#         plt.savefig(original_path + 'result/' + event + "-average_subject_freq-band_grangplot.png", dpi=300)
#         plt.show()
#
#         # plt.suptitle(event + " subject freq-band grangplot", fontsize="x-large")
#         # plt.savefig(original_path+'result/' + event + "_subject_freq-band_grangplot.png")


def plot_point_figure2(y_test, pp, embeddings, flag='female', sci=False, ori=False):
    pca = PCA(n_components=2)
    df = pd.DataFrame(pca.fit_transform(embeddings))
    # tsne = TSNE(n_components=2)
    # df = pd.DataFrame(tsne.fit(embeddings).embedding_)
    label = []
    if flag=='female':
        ss_l = ['female' if i==0 else 'male' for i in y_test[:, 1]]
        # for i, j in zip(y_test[:, 1], pp):
        #     if i == 0 and j == 0:
        #         label.append('female-right')
        #     elif i==0 and j==1:
        #         label.append('female-error')
        #     elif i==1 and j==1:
        #         label.append('male-right')
        #     else:
        #         label.append('male-error')
    else:
        ss_l = ['right' if i == 0 else 'left' for i in y_test[:, 1]]
        # for i, j in zip(y_test[:, 1], pp):
        #     if i == 0 and j == 0:
        #         label.append('right-right')
        #     elif i == 0 and j == 1:
        #         label.append('right-error')
        #     elif i == 1 and j == 1:
        #         label.append('left-right')
        #     else:
        #         label.append('left-error')
    if ori == True:
        oo = 'ori'
    else:
        oo = 'after'
    if ori == True:
        Q1 = np.percentile(df, 25)
        Q3 = np.percentile(df, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df >= lower_bound) & (df <= upper_bound)]
    dfk = pd.DataFrame()
    dfk['Component -1'] = df[0]
    dfk['Component -2'] = df[1]
    dfk['class'] = ss_l #四类 = label  二类 = ss_l
    # dfk['error'] = ['error' if 'error' in i else 'right' for i in label]
    dfk.to_csv('feature'+ oo +'.csv', index=False)
    # sns.set_palette(['#B12D34', '#060605'])
    # dis.set(ylim=(-6, 6))
    # dis.set(xlim=(-40, 40))
    sns.set(font_scale=1.6, font='Arial',style='white')
    sns.set_palette(['#F76F88', '#34ADA3'])
    dis = sns.lmplot(x='Component -1', y='Component -2', data=dfk, hue='class')
    # ax = dis.ax
    # 设置图例在图外
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # 调整字体
    # legend = ax.get_legend()
    # plt.setp(legend.get_texts(), fontsize='12', fontfamily='Arial')  # 设置图例字体
    # plt.setp(legend.get_title(), fontsize='14', fontfamily='Arial')  # 设置图例标题字体
    # ax.set_xlabel("Component -1", fontsize=14, fontfamily='Arial', fontweight='bold')  # 设置 x 轴标签字体
    # ax.set_ylabel("Component -2", fontsize=14, fontfamily='Arial', fontweight='bold')  # 设置 y 轴标签字体
    # ax.tick_params(axis='x', labelsize=12, labelrotation=45, labelcolor='black', labelbottom=True, which='both', length=5, width=5, colors='black', grid_color='r', grid_alpha=0.5)  # 设置 x 轴刻度标签字体
    # ax.tick_params(axis='y', labelsize=12, labelrotation=0, labelcolor='black', labelleft=True, which='both', length=5, width=5, colors='black', grid_color='r', grid_alpha=0.5)  # 设置 y 轴刻度标签字体
    # dis.set(ylim=(-6, 6))
    # dis.set(xlim=(-40, 40))
    if sci:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tight_layout()
    # if ori == True:
    #     plt.savefig('E:/Privacy/材料【这个】/我的论文/AAD/论文图片/代码结果/feature_ori_' +flag+'.tif', dpi=300)
    #     plt.show()
    # elif ori == False:
    #     plt.savefig('E:/Privacy/材料【这个】/我的论文/AAD/论文图片/代码结果/feature_pro_' + flag + '.tif', dpi=300)
    #     plt.show()

    # plt.tight_layout()
    # png1 = io.BytesIO()
    # plt.savefig(png1, dpi=600)
    # # Load this image into PIL
    # png2 = Image.open(png1)
    # png2.save("E:/Privacy/材料【这个】/我的论文/AAD/论文图片/代码结果/feature"+ flag +".tiff")
    # png1.close()
    # plt.show()


    # dis.set(ylim=(-6, 6))
    # dis.set(xlim=(-6, 6))
    # dis._legend.set_bbox_to_anchor((1.05, 0.5))
    # dis._legend.set_title("Class")
    #
    # plt.setp(dis.ax.get_legend().get_texts(), fontsize='12', fontfamily='Arial')  # 设置图例字体
    # plt.setp(dis.ax.get_legend().get_title(), fontsize='14', fontfamily='Arial')  # 设置图例标题字体
    # dis.set_axis_labels("Component -1", "Component -2", fontsize=14, fontfamily='Arial')  # 设置轴标签字体
    # dis.set_xticklabels(fontsize=12, fontfamily='Arial')  # 设置 x 轴刻度标签字体
    # dis.set_yticklabels(fontsize=12, fontfamily='Arial')  # 设置 y 轴刻度标签字体
    #
    # plt.tight_layout()
    # # dis.subplots_adjust(hspace=40)
    # plt.savefig('feature_4_die.png', dpi=300)
    # # sns.lmplot(x='Compoent -1',y='Compoent -2',data=dfk2)
    # sns.lmplot(x='Compoent -1',y='Compoent -2',data=dfk3)


# def plot_point_figure(y_test, embeddings, flag='female'):
#     # pca = PCA(n_components=2)
#     # df = pd.DataFrame(pca.fit_transform(embeddings))
#     tsne = TSNE(n_components=2)
#     df = pd.DataFrame(tsne.fit(embeddings).embedding_)
#     if flag=='female':
#         class_name = ['female' if i == 0 else 'male' for i in y_test[:,1]]
#     else:
#         class_name = ['right' if i == 0 else 'left' for i in y_test[:, 1]]
#     dfk = pd.DataFrame()
#     dfk['Compoent -1'] = df[0]
#     dfk['Compoent -2'] = df[1]
#     dfk['class'] = class_name


    # dfk1 = dfk[df['class'] == 'mdd']
    # dfk3 = dfk[df['class'] == 'hc']

    # # sns.set_palette(['#B12D34', '#060605'])
    # sns.set_palette(['#AF6550', '#59694F'])
    # dis = sns.lmplot(x='Compoent -1', y='Compoent -2', data=dfk, hue='class')
    # # dis.set(ylim=(-6, 6))
    # # dis.set(xlim=(-6, 6))
    # plt.tight_layout()
    # # dis.subplots_adjust(hspace=40)
    # plt.savefig('feature_2_die.png', dpi=300)
    # # sns.lmplot(x='Compoent -1',y='Compoent -2',data=dfk2)
    # # sns.lmplot(x='Compoent -1',y='Compoent -2',data=dfk3)


def plot_roc_sen_spe(y_test, ppp, pp, flag):
    import numpy as np
    n_classes = 2
    y_score = ppp
    target = y_test.numpy()

    #print('f1-score:' + str(metrics.f1_score(y_test[:,1], pp)))
    print(str(y_test[:,1].sum()) + '/' + str(y_test.shape[0]))
    # print(pp)
    print(pp.sum())

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.rcParams['font.size'] = '12'
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    plt.rcParams['image.interpolation'] = 'nearest'

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(9, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.05f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle='-', linewidth=5)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.05f})'
                   ''.format(roc_auc["macro"]),
             color='y', linestyle='-', linewidth=2)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.02f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.1, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18, fontfamily='Arial', labelpad=15)
    plt.ylabel('True Positive Rate', fontsize=18, fontfamily='Arial', labelpad=15)
    # plt.title('Some extension of Receiver operating characteristic to multi-class', fontsize=12, fontfamily='Arial', pad=20)
    legend = plt.legend(loc="lower right", fontsize=18, prop={'family': 'Arial', 'size': 18})
    # png1 = io.BytesIO()
    # plt.savefig(png1, dpi=600)
    # Load this image into PIL
    # png2 = Image.open(png1)
    plt.savefig("E:/Privacy/材料【这个】/我的论文/AAD/论文图片/代码结果/ROC_"+flag+".tif",dpi=300)
    # png1.close()
    # plt.show()



    # y_ = F.onetest_y
    import numpy as np
    from sklearn.metrics import confusion_matrix

    y_probs = ppp
    positive_class_index = 1 # male positive

    y_probs_positive = y_probs[:, positive_class_index]
    threshold = 0.5
    y_pred_binary = (y_probs_positive >= threshold).astype(int)
    y_true = np.argmax(target, axis=1)
    cm = confusion_matrix(y_true, y_pred_binary)
    TN, FP, FN, TP = cm.ravel()

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    Precision = TP / (TP + FP)
    f1 = 2 * (Precision * sensitivity) / (Precision + sensitivity)
    print('precision',Precision)
    print('f1-score',f1)
    print('acc',(TP+TN)/y_test.size(0))
    print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")


    n_classes = 2
    y_score = ppp

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # fpr, tpr, thresholds = roc_curve(test_y, y_pred_binary)
    # roc_auc = auc(fpr, tpr)
    print("ROC AUC:", roc_auc["micro"])
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(10, 10))

    if flag == 'female':
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                         xticklabels=["female", "male"],
                         yticklabels=["female", "male"],
                         annot_kws={"size": 35, "fontfamily": "Arial"})
    else:
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                         xticklabels=["right", "left"],
                         yticklabels=["right", "left"],
                         annot_kws={"size": 35, "fontfamily": "Arial"})  # 调整数字字体大小和字体

    # 调整坐标轴标签的字体大小和字体
    # plt.xlabel("Predicted Label", fontsize=35, fontweight='bold', fontfamily='Arial', labelpad=8)
    # plt.ylabel("True Label", fontsize=35, fontweight='bold', fontfamily='Arial', labelpad=8)
    plt.xlabel("Predicted Label", fontsize=35, fontfamily='Arial', labelpad=8)
    plt.ylabel("True Label", fontsize=35, fontfamily='Arial', labelpad=8)
    # 调整刻度标签的字体大小和字体
    # plt.xticks(fontsize=35, fontfamily='Arial', fontweight='bold')
    # plt.yticks(fontsize=35, fontfamily='Arial', fontweight='bold')
    plt.xticks(fontsize=35, fontfamily='Arial')
    plt.yticks(fontsize=35, fontfamily='Arial')
    cbar_kws = {"size": 35, "fontfamily": "Arial"}  # 字体大小为 15，字体为 Times New Roman
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=35)

    # 调整标题的字体大小和字体
    # plt.title("Confusion Matrix", fontsize=12, fontweight='bold', fontfamily='Arial', pad=20)
    # png1 = io.BytesIO()
    # plt.savefig(png1, dpi=600)
    # # Load this image into PIL
    # png2 = Image.open(png1)

    # plt.savefig("E:/Privacy/材料【这个】/我的论文/AAD/论文图片/代码结果/Confusion_"+flag+".tif",dpi=300)
    # print('hhhhhhhh12345',fpr["micro"], 'ghishgishgish', tpr["micro"])
    # d = pd.DataFrame(dict({'fpr': fpr["micro"], 'tpr': tpr["micro"]}))
    # d.to_csv('./the_data_to_draw/auc.csv', index=False)
    # png1.close()

    # plt.show()
    return roc_auc["micro"], f1, Precision, sensitivity, specificity

