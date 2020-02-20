# -*- coding: <encoding name> -*-
"""
GANDataset
    -- 色素性皮肤病分类数据集
"""
from __future__ import print_function, division
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random
from skimage import exposure, img_as_float
import glob
import  os

################################################################################
# 色素性皮肤病分类数据集
################################################################################
class GANDataset(Dataset):
    """
    色素性皮肤病分类数据集
    """
    def __init__(self, path, transforms, aug=True):
        """
        初始化函数

        :param
            path -- 图片存储目录(str)
            transforms -- 数据增强操作(torchvision.transforms)
            aug -- 额外数据增强操作
        """
        self.transforms = transforms
        self.aug = aug

        # 获取图片路径
        self.imgs = glob.glob(os.path.join(path, "*.jpg"))

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
        # img = img.resize((64, 64)) # w, h

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

        return self.transforms(img)

    def __len__(self):
        """
        获取数据集大小
        :return
            数据集大小
        """
        return len(self.imgs)