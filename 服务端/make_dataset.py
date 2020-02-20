# -*- coding: <encoding name> -*-
"""
数据预处理
"""
from __future__ import print_function, division
import os
import pandas as pd
from sklearn.model_selection import  train_test_split
import numpy as np
import glob
from PIL import Image

################################################################################
# 划分数据集为训练集和测试集，互不交叉，同时保证划分后的数据集类别分布和原始数据集一致
################################################################################
def divide_dataset(path):
    """
    划分数据集
    param:
        path--存放图片和csv文件的目录地址(str)
    return:
        无返回值，以文本形式存储
    """
    print("start preprocess")
    # 判断目录地址是否有效
    if (not os.path.isdir(path)):
        print("%s is not a directory path!!!" % path)
        return

    # 读取csv文件
    label_path = os.path.join(path, "ISIC2018_Task3_Training_GroundTruth.csv")
    labels = pd.read_csv(label_path)
    names = ('MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC')
    # 图片地址
    data_x = []
    # 图片类别
    data_y = []
    for i in range(len(labels)):
        # 存储图片地址
        image_path = 'images/' + labels.at[i, 'image'] + '.jpg'
        image_path = os.path.join(path, image_path)
        if (not os.path.isfile(image_path)):
            print("%s is not a file path!!!" % image_path)
        data_x.append(image_path)

        # 存储图片类别
        tag = []
        for name in names:
            tag.append(labels.at[i, name])
        tag = np.array(tag)
        # 直接存储图片类别，不使用one-hot方式
        data_y.append(np.argmax(tag))

    # 划分数据集
    print('data_x:', len(data_x), ' ', "data_y:", len(data_y))
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.15, stratify = data_y)
    print('x_train:', len(x_train), ' ', "y_train:", len(y_train))
    print('x_test:', len(x_test), ' ', "y_test:", len(y_test))

    # 存储训练集、测试集
    with open(os.path.join(path, 'train_data.txt'), 'wt') as f:
        f.write(str(x_train))
        f.write('\n')
        f.write(str(y_train))
    with open(os.path.join(path, "test_data.txt"), 'wt') as f:
        f.write(str(x_test))
        f.write('\n')
        f.write(str(y_test))
    print("stop preprocess")

################################################################################
# 显示训练集、测试集各类别分布情况
################################################################################
def show_dateset_info():
    """
    显示训练集、测试集各类别分布情况

    :return
        无返回值，直接打印结果
    """
    # 各类别名称
    names = ('MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC')
    # 路径字典
    datas = {'traindataset' : 'Data/train_data.txt', 'testdataset' : 'Data/test_data.txt'}

    # 打印训练集、测试集各类别分布情况
    for k, v in datas.items():
        print(k)
        with open(v, 'rt') as f:
            # 统计每个类别的个数
            pairs = {}

            # 读取数据
            data = f.readlines()
            data = data[1].strip()
            data = data[1: len(data) - 1]
            data = data.split(',')
            for i in range(len(data)):
                data[i] = int(data[i])
            data = np.array(data)

            # 打印各类别分布情况
            # 总样本个数
            total = data.shape[0]
            for val in data:
                if val in pairs.keys():
                    pairs[val] += 1
                else:
                    pairs[val] = 1
            for i in range(7):
                key = i
                val = pairs[key]
                print("%-8s%d/%d%10.2f%%" % (names[key], val, total, (val / total) * 100))
            print("%-8s%d" % ("total", total))


################################################################################
# 裁剪图片
################################################################################
def crop_image():
    PATH = 'Data/images' # 原始图片存储目录
    H, W = (224, 224) # 图片裁剪高、宽

    # 创建存储裁剪图片的目录
    path = 'Data/crop_images_{}_{}'.format(H, W)
    if not os.path.isdir(path):
        os.mkdir(path)

    # 获取图片路径
    img_path_s = glob.glob(os.path.join(PATH, "*.jpg"))
    # 裁剪图片并保存
    for img_path in img_path_s:
        # 文件名
        file_name = img_path.split('/')[-1]
        # 读取图片
        img = Image.open(img_path)
        # 裁剪图片
        img = img.resize((W, H))  # w, h
        # 保存图片
        img.save(os.path.join(path, file_name))

    print("Over")


################################################################################
# 函数入口
################################################################################
if __name__ == '__main__':
    # # 图像、标签csv文件存储目录地址
    # path = 'Data'
    # # 划分数据集
    # divide_dataset(path)

    # # 显示训练集、测试集各类别分布情况
    # show_dateset_info()

    # 裁剪图片
    crop_image()