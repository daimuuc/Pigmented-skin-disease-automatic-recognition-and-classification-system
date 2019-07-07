# -*- coding: <encoding name> -*-
"""
常见工具函数.
balanced_multiclass_accuracy
    --BMC(Balanced Multiclass Accuracy)评价指标.
ImbalancedDatasetSampler
    --概率抽样.
EarlyStopping
    --EarlyStopping机制.
plot_confusion_matrix
    -- 绘制混淆矩阵.
compute_mean_std
    -- 计算平均值(mean)和标准差(std).
plot_loss
    -- 绘制损失图
plot_image
    -- 绘制图片
"""
from __future__ import print_function, division
import torch
from data import SkinDiseaseDataset
import numpy as np
import CONFIG
import matplotlib.pyplot as plt
import itertools
import random
from PIL import Image
import math

################################################################################
# BMC(Balanced Multiclass Accuracy)评价指标.
# 即混淆矩阵各类召回率（Recall）的和的平均
################################################################################
def balanced_multiclass_accuracy(cm):
    """
    BMC(Balanced Multiclass Accuracy)评价指标.
    param：
        cm(numpy) -- 混淆矩阵
    return:
        无返回值，直接输出结果
    """
    # 类别个数
    n = len(cm)
    # 各类召回率（Recall）的和
    recalls = 0.
    # 打印每个类别的精确率（Precision）和召回率（Recall）
    for i in range(len(cm[0])):
        rowsum, colsum = sum(cm[i]), sum(cm[r][i] for r in range(n))
        try:
            print('%d ' %i, 'precision: %.2f' % (cm[i][i] / float(colsum)), 'recall: %.2f' % (cm[i][i] / float(rowsum)))
            recalls += (cm[i][i] / float(rowsum))
        except ZeroDivisionError:
            print('%d ' %i, 'precision: %s' % '0', 'recall: %s' % '0')
    # 打印BMC值
    print('balanced_multiclass_accuracy: %.2f' % (recalls / n))

################################################################################
# 概率抽样.
# 即从给定的不平衡数据集索引列表中随机抽样元素
################################################################################
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    概率抽样.
    """
    def __init__(self, dataset, indices=None, num_samples=None):
        """
        初始化函数
        param:
            dataset -- 数据集(torch.utils.data.Dataset)
            indices -- 索引列表(list,optional)
            num_samples -- 取样数量(int,optional)
        """
        # 若索引列表(indices)未提供，则考虑所有元素
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # 若取样数量(num_samples)未提供，则每次迭代取样数量为总样本数量
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # 数据集中各类分布
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # 每个样本的权重
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        """
        返回指定索引的标签

        :param
            dataset --  数据集(torch.utils.data.Dataset)
            idx -- 索引(int)
        :return
            返回指定索引的标签
        """
        dataset_type = type(dataset)
        if dataset_type is SkinDiseaseDataset:
            return dataset[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        """
        返回概率抽样后索引列表
        """
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement = True))

    def __len__(self):
        """
        获取取样数量
        """
        return self.num_samples

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

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        当验证集损失下降时，存储模型
            param:
             val_loss -- 验证集损失
             model -- 模型
        """
        if self.verbose:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), CONFIG.best_model)
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
    fig.savefig(CONFIG.confusion_matrix_image, bbox_inches='tight')

################################################################################
# 计算平均值(mean)和标准差(std).
# 正则化,即(0,255)-->(0,1)
# cv2.imread()-->BGR-->0~255-->(H,W,C)
# Image.open()-->RGB-->0~255-->(H,W,C)
################################################################################
def compute_mean_std():

    # 训练集地址
    train_txt_path = CONFIG.traindataset_path

    # 挑选多少图片进行计算
    CNum = CONFIG.cnum

    img_h, img_w = 300, 300
    imgs = np.zeros([img_h, img_w, 3, 0])
    means, stdevs = [], []

    with open(train_txt_path, 'rt') as f:
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

        for i in range(CNum):
            img_path = lines[i]

            # # cv2.imread读取
            # img = cv2.imread(img_path)
            # img = cv2.resize(img, (img_w, img_h))

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
# 绘制损失图.
################################################################################
def plot_loss(train_loss, val_loss):
    """
    绘制损失图

    :param
        train_loss(list) -- 训练集损失
        val_loss(list) -- 验证集损失
    """
    # 可视化训练集、验证集损失
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')

    # 寻找验证集损失最小的索引
    minposs = val_loss.index(min(val_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    # 寻找最大损失值
    max_val = max(max(train_loss), max(val_loss))
    max_val = math.ceil(max_val)
    # 寻找最小损失值
    min_val = min(min(train_loss), min(val_loss))
    min_val = math.floor(min_val)

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(min_val, max_val)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # 显示图片
    plt.show()
    # 存储图片
    fig.savefig(CONFIG.loss_image, bbox_inches='tight')

################################################################################
# 绘制图片.
################################################################################
def plot_image(ims, figsize = (12,6), rows = 5, interp = False, titles = None):
    """
    绘制图片

    :param
        ims(numpy) -- 图片数据
        figsize -- 画布大小
        rows(int) -- 行数
        interp -- 填充图片方式
        titles(list) -- 图片标题
    """
    # 转换图片
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))

    # 设置画布大小
    f = plt.figure(figsize = figsize)

    # 获取列数
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1

    # 绘制图片
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

