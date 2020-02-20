# -*- coding: <encoding name> -*-
"""
训练分类模型
    模型 -- Resnet152、InceptionV3、EfficientNet
    评价指标 -- ACC、F1、G-mean、AUC
验证模型
    集成学习策略 --  硬投票策略、软投票策略
"""

from __future__ import print_function, division
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from dataset import SkinDiseaseDataset
from torchvision import transforms
from model import resnet, inception, efficient
import torch.nn as nn
import torch.optim as optim
from utils import EarlyStopping, plot_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix

################################################################################
# 相关配置
################################################################################
# 设置随机种子
np.random.seed(1)
torch.manual_seed(1)
# 选择在cpu或cuda运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建存储模型的目录
if not os.path.isdir('Checkpoint'):
    os.mkdir('Checkpoint')
    os.mkdir('Checkpoint/models')
    os.mkdir('Checkpoint/images')

################################################################################
# 训练模型
################################################################################
def train():
    """
    :return:
    """
    # 设置超参数
    TRAIN_PATH = 'Data/train_data.txt' # 训练集文件地址
    VAL_PATH = 'Data/test_data.txt' # 验证集文件地址
    LR = 1e-4  # 学习率
    EPOCH = 100  # 训练轮数
    BATCH_SIZE = 32  # Batch size
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    resume = False  # 是否断点训练
    N_CLASS = 7  # 类别个数

    # 数据处理
    normalize = transforms.Normalize( # 224x224
        mean=[0.7634611, 0.54736304, 0.5729477],
        std=[0.1432169, 0.1519472, 0.16928367]
    )
    # normalize = transforms.Normalize( # 299x299
    #     mean=[0.76359415, 0.5453203, 0.5692775],
    #     std=[0.14003323, 0.15183851, 0.1698507]
    # )
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=180),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize]
    )
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize]
    )

    # 加载数据
    train_dataset = SkinDiseaseDataset(TRAIN_PATH, transforms=train_transform, aug=True) # 定义trainloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True)  # sampler = ImbalancedDatasetSampler(train_dataset)
    val_dataset = SkinDiseaseDataset(VAL_PATH, transforms=val_transform, aug=False)  # 定义valloader
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 定义模型
    model = resnet(pretrained=False) # Resnet152模型, 输入224x224
    # model = inception(pretrained=True) # InceptionV3, 输入299x299
    # model = efficient(pretrained=True) # EfficientNet, 输入224x224
    model = model.double()
    model = model.to(device)
    # 断点训练，加载模型权重
    if resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('Checkpoint'), 'Error: no Checkpoint directory found!'
        state = torch.load('Checkpoint/models/ckpt.pth')
        model.load_state_dict(state['net'])
        start_epoch = state['epoch']

    # 损失函数
    val = dict(train_dataset.counter())  # N / n，权重为各类别频率的倒数
    total = sum(val.values())
    val = sorted(val.items(), key=lambda item: item[0])
    val = [total / v for k, v in val]
    weight = torch.Tensor(val)  # [ 8.9979,  1.4936, 19.4783, 30.6187,  9.1135, 86.8571, 70.9333]
    weight = weight.double()
    weight = weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # 定义优化器
    # optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.9), weight_decay=1e-5)

    # 定义学习率调度策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, verbose=True)
    # EarlyStopping机制
    early_stopping = EarlyStopping(patience=12, verbose=True)

    # 记录训练过程中训练集、验证集损失
    T_losses = []
    V_losses = []
    # 记录最佳acc、f1、g_mean、auc
    best_acc = 0
    best_f1 = 0
    best_g_mean = 0
    best_auc = 0

    # 训练模型
    for epoch in range(start_epoch, EPOCH):
        start_time = time.time()
        ############################
        # 训练
        ############################
        # 训练模式
        model.train()
        # 迭代次数
        cnt = 0
        # 损失
        sum_loss = 0.0
        for data in train_loader:
            cnt += 1

            # 加载数据
            x, y = data
            x, y = x.double(), y.long()
            x, y = x.to(device), y.to(device)

            # 梯度置零
            optimizer.zero_grad()

            # 前向传播
            output = model(x)
            # Resnet152、EfficientNet
            loss = criterion(output, y)
            # # InceptionV3
            # output, aux = model(x)
            # loss = criterion(output, y) + 0.4 * criterion(aux, y)
            # 后向传播
            loss.backward()
            # 梯度更新
            optimizer.step()

            sum_loss += loss.item()
        t_loss = sum_loss / cnt
        # 保存损失
        T_losses.append(t_loss)
        # 打印日志
        print('[%d/%d]\tLoss_T: %.4f'
              % (epoch, EPOCH, t_loss), end='')

        ############################
        # 验证
        ############################
        # 训练模式
        model.eval()
        # 迭代次数
        cnt = 0
        # 损失
        sum_loss = 0.0
        # 真实标签 N
        total_y = None
        # 预测标签 N
        total_a = None
        # 预测概率 N x M
        total_b = None
        with torch.no_grad():
            for data in val_loader:
                cnt += 1

                # 加载数据
                x, y = data
                x, y = x.double(), y.long()
                x, y = x.to(device), y.to(device)

                # 前向传播
                output = model(x)

                # loss
                loss = criterion(output, y)
                sum_loss += loss.item()

                # 预测和真实标签
                output = nn.Softmax(dim=1)(output)
                _, a = torch.max(output.detach(), 1)
                y = y.cpu().detach().numpy()
                a = a.cpu().detach().numpy()
                y = y.astype(int)
                a = a.astype(int)
                if total_y is None:
                    total_y = y
                else:
                    total_y = np.append(total_y, y)
                if total_a is None:
                    total_a = a
                else:
                    total_a = np.append(total_a, a)
                # 预测概率
                b = output.cpu().detach().numpy()
                if total_b is None:
                    total_b = b
                else:
                    total_b = np.concatenate((total_b, b), axis=0)
        v_loss = sum_loss / cnt
        # 保存损失
        V_losses.append(v_loss)

        # 计算acc
        v_acc = accuracy_score(total_y, total_a)
        if best_acc < v_acc:
            best_acc = v_acc
        # 计算f1
        v_f1 = f1_score(total_y, total_a, average='macro')
        if best_f1 < v_f1:
            best_f1 = v_f1
        # 计算G-mean
        v_g_mean = geometric_mean_score(total_y, total_a, average='macro')
        if best_g_mean < v_g_mean:
            best_g_mean = v_g_mean
        # 计算auc
        v_auc = roc_auc_score(label_binarize(total_y, np.arange(N_CLASS)), total_b, average='macro')
        if best_auc < v_auc:
            best_auc = v_auc

        # 打印日志
        print('\tLoss_V: %.4f\tAcc_V: %.4f\tF1_V: %.4f\tG_mean_V: %.4f\tAuc_V: %.4f\n[====]Time: %.4f[minute]'
              % (v_loss, v_acc, v_f1, v_g_mean, v_auc, (time.time() - start_time) / 60))
        # earlyStopping机制
        early_stopping(v_loss, model, epoch)
        # 学习率调度机制
        scheduler.step(v_loss)
        # 保存模型
        state = {
            'net': model.state_dict(),
            'epoch': epoch
        }
        torch.save(state, 'Checkpoint/models/ckpt.pth')
        # 判断是否停止训练
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 打印最佳acc、f1
    print('Best_ACC: %.4f\t Best_F1: %.4f\t Best_G_mean: %.4f\t Best_AUC: %.4f' % (
    best_acc, best_f1, best_g_mean, best_auc))

    # 绘制损失图
    plt.figure(figsize=(10, 5))
    plt.title("Train and Val Loss During Training")
    plt.plot(T_losses, label="T")
    plt.plot(V_losses, label="V")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.legend()
    # 保存损失图
    plt.savefig('Checkpoint/images/loss_curve.png')
    # 显示损失图
    plt.show()


################################################################################
# 验证模型
################################################################################
def val():
    # 设置超参数
    VAL_PATH = 'Data/test_data.txt'  # 验证集文件地址
    BATCH_SIZE = 32  # Batch size
    VOTE = 'soft'  # 投票策略
    N_CLASS = 7  # 类别数量

    # 数据处理
    normalize = transforms.Normalize(  # 224x224
        mean=[0.7634611, 0.54736304, 0.5729477],
        std=[0.1432169, 0.1519472, 0.16928367]
    )
    # normalize = transforms.Normalize( # 299x299
    #     mean=[0.76359415, 0.5453203, 0.5692775],
    #     std=[0.14003323, 0.15183851, 0.1698507]
    # )
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize]
    )

    # 加载数据
    val_dataset = SkinDiseaseDataset(VAL_PATH, transforms=val_transform, aug=False)  # 定义valloader
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 定义模型
    model_list = []
    assert os.path.isdir('Model'), 'Error: no Model directory found!'
    model = resnet(pretrained=False)  # Resnet152模型, 输入224x224
    model.load_state_dict(torch.load('Model/resnet_ckpt.pt'))  # 加载模型权重
    model_list.append(model.double().to(device))
    # model = inception(pretrained=False) # InceptionV3, 输入299x299
    # model.load_state_dict(torch.load('Model/inception_ckpt.pt'))  # 加载模型权重
    # model_list.append(model.double().to(device))
    # model = efficient(pretrained=False) # EfficientNet, 输入224x224
    # model.load_state_dict(torch.load('Model/efficient_ckpt.pt'))  # 加载模型权重
    # model_list.append(model.double().to(device))

    ############################
    # 验证
    ############################
    # 测试模式
    model.eval()
    with torch.no_grad():
        # 真实标签 N
        total_y = None
        # 预测标签 N
        total_a = None
        # 预测概率 N x M
        total_b = None
        for data in val_loader:
            # 加载数据
            x, y = data
            x, y = x.double(), y.long()
            x, y = x.to(device), y.to(device)
            y = y.cpu().detach().numpy()
            y = y.astype(int)
            if total_y is None:
                total_y = y
            else:
                total_y = np.append(total_y, y)

            # 前向传播
            if VOTE == 'soft':
                # 软投票策略
                result = None
                for model in model_list:
                    output = model(x)
                    output = nn.Softmax(dim=1)(output)
                    output = output.detach()
                    if result is None:
                        result = output
                    else:
                        result += output
                result = result / len(model_list)
                # 预测标签
                _, a = torch.max(result.detach(), 1)
                a = a.cpu().detach().numpy()
                a = a.astype(int)
                if total_a is None:
                    total_a = a
                else:
                    total_a = np.append(total_a, a)
                # 预测概率
                b = result.cpu().detach().numpy()
                if total_b is None:
                    total_b = b
                else:
                    total_b = np.concatenate((total_b, b), axis=0)

            else:
                # 硬投票策略
                result = None
                for model in model_list:
                    output = model(x)
                    output = nn.Softmax(dim=1)(output)
                    _, a = torch.max(output.detach(), 1)
                    a = a.cpu().detach().numpy()
                    a = a.astype(int)
                    if result is None:
                        result = a[np.newaxis, :]
                    else:
                        result = np.concatenate((result, a[np.newaxis, :]), axis=0)
                # 根据少数服从多数原则确定每个样本所属类别
                val = []
                for i in range(result.shape[1]):
                    val.append(np.argmax(np.bincount(result[:, i])))
                val = np.asarray(val)
                if total_a is None:
                    total_a = val
                else:
                    total_a = np.append(total_a, val)

        # 计算acc
        acc = accuracy_score(total_y, total_a)
        # 计算f1
        f1 = f1_score(total_y, total_a, average='macro')
        # 计算G-mean
        g_mean = geometric_mean_score(total_y, total_a, average='macro')
        # 计算auc
        auc = 0.
        if VOTE == 'soft':
            auc = roc_auc_score(label_binarize(total_y, np.arange(N_CLASS)), total_b, average='macro')

    # 打印acc、f1
    print('ACC: %.4f\t F1: %.4f\n'
          'G-mean: %.4f\t AUC: %.4f' % (acc, f1, g_mean, auc))

    # 计算混淆矩阵
    cm = confusion_matrix(total_y, total_a)
    # 可视化混淆矩阵
    cm_plot_labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


################################################################################
# 函数入口
################################################################################
if __name__ == '__main__':
    # # 训练分类模型
    # train()

    # 验证模型
    val()