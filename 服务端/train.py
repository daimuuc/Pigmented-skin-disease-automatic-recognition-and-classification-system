# -*- coding: <encoding name> -*-
"""
训练模型.
"""
from __future__ import print_function, division
from torchsummary import summary
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from data import SkinDiseaseDataset
from utils import ImbalancedDatasetSampler, EarlyStopping, plot_loss
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from models import inception, resnet, densenet, senet
import CONFIG
from model import MFFNet

################################################################################
# 训练模型
################################################################################
def train(train_path, val_path, **kwargs):
    """
    训练模型
    :param
        train_path(str) -- 训练集地址
        val_path(str) -- 验证集地址
    : kwargs
        model(int) -- 训练模型
        epoch(int) -- 训练轮数
        batch_size(int) -- 训练批次大小
        learn_rate(int) -- 学习率
    :return
        返回训练集损失(list)、验证集损失(list)
    """
    # 设置超参数
    lrs = [1e-3, 1e-4, 1e-5]
    # 学习率
    LR = lrs[kwargs['learn_rate'] - 1]
    # 训练轮数
    EPOCH = kwargs['epoch']
    # 批次大小
    BATCH_SIZE = kwargs['batch_size']

    # 数据处理
    normalize = transforms.Normalize(
        mean=[0.76209545, 0.54330575, 0.5679443],
        std=[0.14312604, 0.154518, 0.17225058]
    )
    train_transform = transforms.Compose([
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(degrees = 180),
                                    transforms.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1),
                                    transforms.ToTensor(),
                                    normalize]
                                    )
    val_transform = transforms.Compose([
                                    transforms.RandomCrop(224),
                                    transforms.ToTensor(),
                                    normalize]
                                    )

    # 加载数据
    #定义trainloader
    train_dataset = SkinDiseaseDataset(train_path, transforms = train_transform, agumentation = False)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True) # sampler = ImbalancedDatasetSampler(train_dataset)

    #定义valloader
    val_dataset = SkinDiseaseDataset(val_path, transforms = val_transform, agumentation = False)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

    # 选择运行的cpu或gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    if kwargs['model'] == 1:
        model = MFFNet()
    elif kwargs['model'] == 2:
        # 299 x 299
        model = inception()
    elif kwargs['model'] == 3:
        model = resnet()
    elif kwargs['model'] == 4:
        model = densenet()
    elif kwargs['model'] == 5:
        model = senet()
    # # 断点训练，加载模型权重
    # model.load_state_dict(torch.load(CONFIG.best_model))
    model = model.to(device)

    # 定义损失函数
    # N / n，权重为各类别频率的倒数
    weight = torch.Tensor([9., 1.5, 19.48, 30.62, 9.11, 86.86, 71.])
    weight = weight.to(device)
    criterion = nn.CrossEntropyLoss(weight = weight)

    # 定义优化器
    # optimizer = optim.SGD(model.parameters(), lr = LR, weight_decay = 1e-5)
    optimizer = optim.Adam(model.parameters(), lr = LR, weight_decay = 1e-5)

    # 定义学习率调度策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 7, verbose = True)

    # 可视化模型
    if kwargs['model'] == 2:
        summary(model, (3, 299, 299))
    else:
        summary(model, (3, 224, 224))
    print(model)

    # 在模型训练过程中，跟踪每一轮平均训练集损失
    avg_train_losses = []
    # 在模型训练过程中，跟踪每一轮平均验证集损失
    avg_valid_losses = []

    # EarlyStopping机制
    early_stopping = EarlyStopping(patience = 12, verbose = True)

    # 训练模型
    for epoch in range(EPOCH):
        # 训练模式
        model.train()
        # 损失
        sum_loss = 0.0
        # 预测正确样本数
        correct = 0
        # 总样本数
        total = 0
        # 迭代次数
        cnt = 0
        for data in train_loader:
            cnt += 1

            # 加载数据
            image, label = data
            image, label = image.to(device), label.to(device)

            # 梯度置零
            optimizer.zero_grad()

            # 前向传播、后向传播
            output = model(image)
            loss = criterion(output, label)
            # inceptionV3
            # loss = criterion(output, label) + 0.4 * criterion(aux, label)
            loss.backward()
            optimizer.step()

            # 计算loss and acc
            sum_loss += loss.item()
            _, a = torch.max(output.detach(), 1)
            b = label.detach()
            total += label.size(0)
            correct += (a == b).sum()
        # 打印loss和acc
        print('[ %d/%d ] train_loss:%.2f train_acc:%.2f%%' % (epoch + 1, EPOCH, sum_loss / cnt, 100 * correct / total))
        avg_train_losses.append(sum_loss / cnt)

        # 验证模式
        model.eval()
        # 损失
        sum_loss = 0.0
        # 预测正确样本数
        correct = 0
        # 总样本数
        total = 0
        # 迭代次数
        cnt = 0
        for data in val_loader:
            cnt += 1

            # 加载数据
            image, label = data
            image, label = image.to(device), label.to(device)

            # 前向传播
            output = model(image)
            loss = criterion(output, label)

            # 计算loss和acc
            sum_loss += loss.item()
            _, a = torch.max(output.detach(), 1)
            b = label.detach()
            total += label.size(0)
            correct += (a == b).sum()
        # 打印loss和acc
        print("          val_loss:%.2f val_acc:%.2f%%" % (sum_loss / cnt, 100 * correct / total))
        avg_valid_losses.append(sum_loss / cnt)

        # earlyStopping机制
        early_stopping(sum_loss / cnt, model)
        # 学习率调度机制
        scheduler.step(sum_loss / cnt)

        # 保存模型
        torch.save(model.state_dict(), CONFIG.intermediate_model + '/checkpoint_%d.pt' % (epoch + 1))

        # 判断是否停止训练
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return avg_train_losses, avg_valid_losses

################################################################################
# 函数入口
################################################################################
if __name__ == '__main__':
    # 训练集地址
    train_path = CONFIG.traindataset_path
    # 验证集地址
    val_path = CONFIG.testdataset_path

    # 解析参数
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('-m', '--model', type = int, default = 1, help = 'choose models, 1)MFFNet 2)Inception 3)ResNet 4)DenseNet 5)SeNet, default 1',
                        choices = [1, 2, 3, 4, 5])
    parser.add_argument('-e', '--epoch', type = int, default = 100, help = 'set train epoches, default 100')
    parser.add_argument('-b', '--batch_size', type = int, default = 32, help = 'set batch size, default 32')
    parser.add_argument('-lr', '--learn_rate', type = int, default = 2, help = 'choose learn rate, 1)1e-3, 2)1e-4, 3)1e-5, default 1',
                        choices = [1, 2, 3])
    args = parser.parse_args()
    # 打印参数
    print('model', args.model)
    print('epoch', args.epoch)
    print('batch_size', args.batch_size)
    print('learn_rate', args.learn_rate)

    args = {'model' : args.model, 'epoch' : args.epoch, \
            'batch_size' : args.batch_size, 'learn_rate' : args.learn_rate}

    # 训练模型
    avg_train_losses, avg_valid_losses = train(train_path, val_path, **args)

    # 存储训练集loss、验证集loss
    with open(CONFIG.loss_log, 'wt') as f:
        f.write(str(avg_train_losses))
        f.write('\n')
        f.write(str(avg_valid_losses))

    # 可视化损失图
    plot_loss(avg_train_losses, avg_valid_losses)





