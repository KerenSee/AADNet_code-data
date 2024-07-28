from osgeo import gdal
import os


def TIFToPNG(tifDir_path, pngDir_path):
    for fileName in os.listdir(tifDir_path):
        if fileName[-4:] == ".tif":
            ds = gdal.Open(tifDir_path + fileName)
            driver = gdal.GetDriverByName('PNG')
            driver.CreateCopy(pngDir_path + fileName[:-4] + ".png", ds)
            print("已生成：", pngDir_path + fileName[:-4] + ".png")


if __name__ == '__main__':
    tifDir_path = "E:/Privacy/材料【这个】/我的论文/AAD/论文图片/所有图/tiff/"
    pngDir_path = "E:/Privacy/材料【这个】/我的论文/AAD/论文图片/所有图/png/"
    TIFToPNG(tifDir_path, pngDir_path)