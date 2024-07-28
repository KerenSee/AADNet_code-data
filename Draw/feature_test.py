import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_point_figure2(y_test, pp, embeddings, flag='female'):
    pca = PCA(n_components=2)
    df = pd.DataFrame(pca.fit_transform(embeddings))

    label = []
    if flag=='female':
        ss_l = ['female' if i==0 else 'male' for i in y_test[:, 1]]
        for i, j in zip(y_test[:, 1], pp):
            if i == 0 and j == 0:
                label.append('female-right')
            elif i==0 and j==1:
                label.append('female-error')
            elif i==1 and j==1:
                label.append('male-right')
            else:
                label.append('male-error')
    else:
        ss_l = ['right' if i == 0 else 'left' for i in y_test[:, 1]]
        for i, j in zip(y_test[:, 1], pp):
            if i == 0 and j == 0:
                label.append('right-right')
            elif i == 0 and j == 1:
                label.append('right-error')
            elif i == 1 and j == 1:
                label.append('left-right')
            else:
                label.append('left-error')
    dfk = pd.DataFrame()
    dfk['Compoent -1'] = df[0]
    dfk['Compoent -2'] = df[1]
    dfk['class'] = ss_l #四类 = label  二类 = ss_l
    # dfk['error'] = ['error' if 'error' in i else 'right' for i in label]

    # sns.set_palette(['#B12D34', '#060605'])
    sns.set_palette(['#F76F88', '#34ADA3'])
    # sns.set(font_scale=1.2, font='Arial')
    dis = sns.lmplot(x='Compoent -1', y='Compoent -2', data=dfk, hue='class')
    ax = dis.ax
    # 设置图例在图外
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # 调整字体
    # legend = ax.get_legend()
    # plt.setp(legend.get_texts(), fontsize='12', fontfamily='Arial')  # 设置图例字体
    # plt.setp(legend.get_title(), fontsize='14', fontfamily='Arial')  # 设置图例标题字体
    ax.set_xlabel("Component -1", fontsize=14, fontfamily='Arial', fontweight='bold')  # 设置 x 轴标签字体
    ax.set_ylabel("Component -2", fontsize=14, fontfamily='Arial', fontweight='bold')  # 设置 y 轴标签字体
    # ax.legend(loc='upper right', bbox_to_anchor=(1, 0.5), prop={'family': 'Arial', 'size': 12})
    # ax.tick_params(axis='x', labelsize=12, labelrotation=45, labelcolor='black', labelbottom=True, which='both', length=5, width=5, colors='black', grid_color='r', grid_alpha=0.5)  # 设置 x 轴刻度标签字体
    # ax.tick_params(axis='y', labelsize=12, labelrotation=0, labelcolor='black', labelleft=True, which='both', length=5, width=5, colors='black', grid_color='r', grid_alpha=0.5)  # 设置 y 轴刻度标签字体

    plt.tight_layout()
    # plt.savefig('E:/Privacy/材料【这个】/我的论文/AAD/论文图片/代码结果/feature_'+flag+'.png', dpi=300)
    plt.show()

# 示例调用
# y_test 和 pp 是示例数据
y_test = np.array([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]])
pp = np.array([1, 0, 0, 1, 0])
embeddings = np.random.rand(5, 5)  # 示例嵌入

plot_point_figure2(y_test, pp, embeddings, flag='female')
