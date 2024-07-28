import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 示例数据
y_true = [0, 1, 1, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 0, 1, 1]

# 生成混淆矩阵
cm = confusion_matrix(y_true, y_pred)

flag = 'female'

plt.figure(figsize=(10, 10))

if flag == 'female':
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     xticklabels=["female", "male"],
                     yticklabels=["female", "male"],
                     annot_kws={"size": 48, "fontfamily": "Arial"})
else:
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     xticklabels=["right", "left"],
                     yticklabels=["right", "left"],
                     annot_kws={"size": 48, "fontfamily": "Arial"})  # 调整数字字体大小和字体

# 调整坐标轴标签的字体大小和字体
plt.xlabel("Predicted Label", fontsize=35, fontweight='bold', fontfamily='Arial', labelpad=8)
plt.ylabel("True Label", fontsize=35, fontweight='bold', fontfamily='Arial', labelpad=8)

# 调整刻度标签的字体大小和字体
plt.xticks(fontsize=35, fontfamily='Arial', fontweight='bold')
plt.yticks(fontsize=35, fontfamily='Arial',  fontweight='bold')
cbar_kws = {"size": 35, "fontfamily": "Times New Roman"}  # 字体大小为 15，字体为 Times New Roman
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=35)

# 调整 color bar 刻度的字体大小
# cbar.set_label('Count', rotation=270, labelpad=20, **cbar_kws)
# 调整标题的字体大小和字体
# plt.title("Confusion Matrix", fontsize=30, fontweight='bold', fontfamily='Arial')

plt.savefig('E:/Privacy/材料【这个】/我的论文/AAD/论文图片/代码结果/title_confusion.png', dpi=300)
plt.show()
