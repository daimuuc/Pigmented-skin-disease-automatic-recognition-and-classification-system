# -*- coding: <encoding name> -*-
"""
SkinDiseaseDataset
    -- 色素性皮肤病分类数据集
"""
from __future__ import print_function, division
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
import torch
from PIL import Image
import random
from skimage import exposure, img_as_float

################################################################################
# 色素性皮肤病分类数据集
################################################################################
class SkinDiseaseDataset(Dataset):
    """
    色素性皮肤病分类数据集
    """
    def __init__(self, path, transforms, aug=True):
        """
        初始化函数

        :param
            path -- 训练集或测试集存储地址(str)
            transforms -- 数据增强操作(torchvision.transforms)
            aug -- 额外数据增强操作
        """
        self.transforms = transforms
        self.aug = aug
        # 读取数据集
        with open(path, "rt") as f:
            data = f.readlines()

            imgs = data[0].strip()
            imgs = imgs[1: len(imgs) - 1]
            imgs = imgs.split(',')
            for i in range(len(imgs)):
                imgs[i] = imgs[i].strip()
                imgs[i] = imgs[i].strip('\'')
            # 图片路径
            self.imgs = imgs

            labels = data[1].strip()
            labels = labels[1: len(labels) - 1]
            labels = labels.split(',')
            for i in range(len(labels)):
                labels[i] = int(labels[i])
            # 图片标签
            self.labels = np.array(labels)

    def __getitem__(self, index):
        """
        获取样本

        :param
            index -- 索引
        :return
            返回(图片数据，图片标签)
        """
        # 图片路径
        image_path = self.imgs[index]
        # 读取图片
        img = Image.open(image_path)
        # # 裁剪图片
        # img = img.resize((224, 224)) # w, h

        # 数据增强
        if self.aug:
            if random.random() > 0.5: # flip
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.7: # gamma transform
                image = img_as_float(img)
                # gamma_img: np.array(dtype=float64) ranging [0,1]
                if random.random() > 0.5:
                    gamma_img = exposure.adjust_gamma(image, 1.5)
                else:
                    gamma_img = exposure.adjust_gamma(image, 0.5)
                gamma_img = gamma_img * 255
                gamma_img = np.uint8(gamma_img)
                img = Image.fromarray(gamma_img)

        return self.transforms(img), self.labels[index]

    def __len__(self):
        """
        获取数据集大小
        :return
            数据集大小
        """
        return len(self.imgs)

    def counter(self):
        return Counter(self.labels)