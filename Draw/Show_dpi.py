from PIL import Image
import os

path = 'E:/Privacy/材料【这个】/我的论文/AAD/论文图片/所有图/png/'
for fileName in os.listdir(path):
    path_new = path + fileName
    img = Image.open(path_new)

    dpi = img.info['dpi']
    print('图片：',fileName)
    print("水平DPI:",dpi[0])
    print("垂直DPI:",dpi[1])