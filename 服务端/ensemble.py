# -*- coding: <encoding name> -*-
"""
测试集成模型性能.
"""
from __future__ import print_function, division
from torch.utils.data import DataLoader
from torchvision import transforms
from data import SkinDiseaseDataset
from models import inception, resnet, densenet, senet
from model import MFFNet
import torch
import torch.nn as nn
import CONFIG
import numpy as np
from utils import plot_confusion_matrix, balanced_multiclass_accuracy
from sklearn.metrics import confusion_matrix, classification_report

################################################################################
# 测试集成模型
################################################################################
def test(test_path, agumentation, **kwargs):
    """
        测试集成模型性能
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
    models = []
    for k, v in kwargs.items():
        if k == '1':
            model = MFFNet()
        elif k == '2':
            # 299 x 299
            model = inception(pretrained = False)
        elif k == '3':
            model = resnet(pretrained = False)
        elif k == '4':
            model = densenet(pretrained = False)
        elif k == '5':
            model = senet(pretrained = False)
        model.load_state_dict(torch.load(v))
        model = model.to(device)
        # 测试模式
        model.eval()
        models.append(model)

    # 测试模型
    # 各类别预测正确个数
    class_correct = list(0. for i in range(7))
    # 各类别总个数
    class_total = list(0. for i in range(7))
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

            # 使用平均策略获取预测值，即最终的输出为各模型输出和的平均
            sum = None
            # 平均策略
            for model in models:
                output = model(image)

                # 使用平均策略获取模型的输出
                output = output.detach()
                # 平均策略
                val = None
                for i in range(output.size(0)):
                    if val is None:
                        val = output[i]
                    else:
                        val = val + output[i]
                val = val / output.size(0)

                if sum is None:
                    sum = val
                else:
                    sum += val
            val = sum / len(models)
            _, a = torch.max(val, 0)

            # 统计各个类预测正确的个数
            m = label.detach()
            class_correct[m] += 1 if a == m else 0
            correct += 1 if a == m else 0
            class_total[m] += 1

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

            # 使用平均策略，获取输出，即最终的输出为各模型输出和的平均
            output = None
            # 平均策略
            for model in models:
                if output is None:
                    output = model(image).detach()
                else:
                    output += model(image).detach()
            output = output / len(models)

            # acc
            _, a = torch.max(output, 1)
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
    # 打印acc
    print("test_acc:%.2f%%\n" % (100 * correct / total))
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
    plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
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

    # 集成模型配置. 1)MFFNet 2)Inception 3)ResNet 4)DenseNet 5)SeNet
    args = {  # '1' : 'train_models/mffnet_model.pt',
        # '2': CONFIG.inception_model,
        # '3': CONFIG.resnet_model,
        '4': CONFIG.densenet_model,
        '5': CONFIG.senet_model}

    # 测试集成模型
    test(test_path, agumentation = False, **args)