# -*- coding: <encoding name> -*-
"""
常见工具函数.
"""

from matplotlib import pyplot as plt
import os
import torch
import numpy as np
from PIL import Image
import random
import itertools

################################################################################
# 计算平均值(mean)和标准差(std).
# 正则化,即(0,255)-->(0,1)
# cv2.imread()-->BGR-->0~255-->(W,H,C)
# Image.open()-->RGB-->0~255-->(H,W,C) ==> img.size-->(W,H) img.resize-->(W,H)
################################################################################
def compute_mean_std():
    # 训练集地址
    TRAIN_PATH = 'Data/train_data.txt'
    # 挑选多少图片进行计算
    CNUM = 2000

    img_w, img_h = 224, 224
    imgs = np.zeros([img_h, img_w, 3, 0])
    means, stdevs = [], []

    with open(TRAIN_PATH, 'rt') as f:
        # 读取数据
        data = f.readlines()
        lines = data[0].strip()
        lines = lines[1: len(lines) - 1]
        lines = lines.split(',')
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
            lines[i] = lines[i].strip('\'')
        # shuffle , 随机挑选图片
        random.shuffle(lines)

        for i in range(CNUM):
            img_path = lines[i]

            # # cv2.imread读取
            # img = cv2.imread(img_path)
            # img = cv2.resize(img, (img_h, img_w))

            # PIL Image.open读取
            img = Image.open(img_path)
            img = img.resize((img_w, img_h))
            img = np.array(img)

            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis = 3)
            print(i)

    imgs = imgs.astype(np.float32)/255.


    for i in range(3):
        # 拉成一行
        pixels = imgs[:,:,i,:].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # # cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
    # means.reverse() # BGR --> RGB
    # stdevs.reverse()

    # 打印平均值和标准差
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))


################################################################################
# EarlyStopping机制.
# 如果给定patience后，验证集损失还没有改善，则停止训练.
################################################################################
class EarlyStopping:
    """
    EarlyStopping机制.
    """
    def __init__(self, patience = 7, verbose = False):
        """
        初始化函数
        param:
            patience(int) -- 在上一次验证集损失改善后等待多少epoch
            verbose(bool) -- 是否打印信息
        :param verbose:
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score <= self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        """
        当验证集损失下降时，存储模型
            param:
             val_loss -- 验证集损失
             model -- 模型
             epoch -- 轮数
        """
        if self.verbose:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))
        state = {
            'net': model.state_dict(),
            'epoch': epoch
        }
        torch.save(state, 'Checkpoint/models/best_ckpt.pth')
        self.val_loss_min = val_loss


################################################################################
# 绘制混淆矩阵.
################################################################################
def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):
    """
    绘制混淆矩阵
    :param
        cm(numpy) -- 混淆矩阵
        classes(list) --类别名称
        normalize(bool) -- 是否归一化
        title(str) -- 标题
        cmap -- 颜色
    """
    # 是否归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    # 绘制混淆矩阵
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # 显示混淆矩阵
    plt.show()
    # 保存混淆矩阵图
    fig.savefig('Model/confusion_matrix.png', bbox_inches='tight')

################################################################################
# 函数入口
################################################################################
if __name__ == '__main__':
    # 计算平均值(mean)和标准差(std)
    compute_mean_std()