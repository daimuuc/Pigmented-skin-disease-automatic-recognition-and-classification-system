# -*- coding: <encoding name> -*-
"""
测试模型性能.
"""
from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data import SkinDiseaseDataset
from sklearn.metrics import confusion_matrix, classification_report
from utils import plot_confusion_matrix, balanced_multiclass_accuracy
from models import inception, resnet, densenet, senet
from model import MFFNet
import argparse
import numpy as np
import CONFIG

################################################################################
# 测试模型
################################################################################
def test(test_path, agumentation, **kwargs):
    """
    测试模型性能
    :param
        test_path(str) -- 测试集地址
        agumentation(bool) -- 是否对单个图片多次复制
    :kwargs
        model(int) -- 模型
    """
    # 设置超参数
    if agumentation:
        BATCH_SIZE = 1
    else:
        BATCH_SIZE = 32

    # 选择运行的cpu或gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义损失函数
    # N / n，权重为各类别频率的倒数
    weight = torch.Tensor([9., 1.5, 19.48, 30.62, 9.11, 86.86, 71.])
    weight = weight.to(device)
    criterion = nn.CrossEntropyLoss(weight = weight)

    # 数据处理
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize]
    )

    # 加载数据
    # 定义test_loader
    test_dataset = SkinDiseaseDataset(test_path, transforms = test_transform, agumentation = agumentation)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE)

    # 加载模型
    if kwargs['model'] == 1:
        model = MFFNet()
    elif kwargs['model'] == 2:
        # 299 x 299
        model = inception(pretrained = False)
    elif kwargs['model'] == 3:
        model = resnet(pretrained = False)
    elif kwargs['model'] == 4:
        model = densenet(pretrained = False)
    elif kwargs['model'] == 5:
        model = senet(pretrained = False)
    # 加载模型权重
    model.load_state_dict(torch.load(CONFIG.best_model))
    model = model.to(device)

    # 测试模式
    model.eval()
    # 各类别预测正确个数
    class_correct = list(0. for i in range(7))
    # 各类别总个数
    class_total = list(0. for i in range(7))
    # 损失
    sum_loss = 0.0
    # 总预测正确个数
    correct = 0
    # 总个数
    total = 0
    # 总迭代次数
    cnt = 0
    # 测试集增强模式
    if agumentation:
        # 预测标签情况
        x = []
        # 真实标签情况
        y = []
        for data in test_loader:
            cnt += 1

            # 加载数据
            image, label = data
            image = image.view(-1, 3, 224, 224)
            label = label[0]
            image, label = image.to(device), label.to(device)

            # 前向传播
            output = model(image)

            # 使用平均策略获取预测值
            output = output.detach()
            # 平均策略
            val = None
            for i in range(output.size(0)):
                if val is None:
                    val = output[i]
                else:
                    val = val + output[i]
            val = val / output.size(0)
            _, a = torch.max(val, 0)

            # 统计各个类预测正确的个数
            m = label.detach()
            class_correct[m] += 1 if a == m else 0
            class_total[m] += 1
            # 统计预测正确总个数
            correct += 1 if a == m else 0

            x.append(a.item())
            y.append(m.item())
        # list转化为numpy
        x = np.array(x)
        y = np.array(y)
    else:
        # 预测标签情况
        x = None
        # 真实标签情况
        y = None
        for data in test_loader:
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

            # 预测和真实标签情况
            if x is None:
                x = a
                y = b
            else:
                x = torch.cat((x, a))
                y = torch.cat((y, b))

            # 统计每个类别的正确预测情况
            for i in range(label.size(0)):
                m = b[i]
                class_correct[m] += 1 if a[i] == m else 0
                class_total[m] += 1
        # tensor转化为numpy
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()

    # 打印结果
    cm_plot_labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    # 判断测试集是否增强
    if agumentation:
        # 打印acc
        print("test_acc:%.2f%%\n" % (100 * correct / cnt))
    else:
        # 打印loss和acc
        print("test_loss:%.2f test_acc:%.2f%%\n" % (sum_loss / cnt, 100 * correct / total))
    # 打印每个类别的acc
    for i in range(7):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %.2f%% (%2d/%2d)' % (
                cm_plot_labels[i], 100 * class_correct[i] / class_total[i],
                    class_correct[i], class_total[i]))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % cm_plot_labels[i])
    print('')

    # 计算混淆矩阵
    cm = confusion_matrix(y, x)
    print('')

    # 计算BMC
    balanced_multiclass_accuracy(cm)
    print('')

    # 可视化混淆矩阵
    plot_confusion_matrix(cm, cm_plot_labels, title = 'Confusion Matrix')
    print('')

    # 打印分类报告
    report = classification_report(y, x, target_names=cm_plot_labels)
    print(report)

################################################################################
# 函数入口
################################################################################
if __name__ == '__main__':
    # 测试集地址
    test_path = CONFIG.testdataset_path

    # 解析参数
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('-m', '--model', type=int, default=1,
                        help='choose models, 1)MFFNet 2)Inception 3)ResNet 4)DenseNet 5)SeNet, default 1',
                        choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    args = {'model': args.model}

    # 测试模型
    test(test_path, agumentation = False, **args)