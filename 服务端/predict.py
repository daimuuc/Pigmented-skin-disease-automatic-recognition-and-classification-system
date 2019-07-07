# -*- coding: <encoding name> -*-
"""
模型预测结果.
"""
from __future__ import print_function, division
import os
from PIL import Image
import torch
from torchvision import transforms
from models import inception, resnet, densenet, senet
from model import MFFNet
import numpy as np
from utils import plot_image
import CONFIG

################################################################################
# 获取预测图片路径，并以文本文件存储
################################################################################
def image_path(path):
    """
    获取预测图片路径，并以文本文件存储
    :param
        path(str) -- 存储图片目录地址
    """
    print("start image_path")
    # 判断路径是否是有效目录
    if not os.path.isdir(path):
        print('It is not a directory')
        return

    # 获取图片路径
    paths = []
    for img_path in os.listdir(path):
        if img_path.__contains__('.jpg'):
            paths.append(os.path.join(path, img_path))

    # 存储图片路径
    with open(os.path.join(path, 'image_path.txt'), 'wt') as f:
        for img_path in paths:
            f.write(img_path + '\n')
    print("stop image_path")

################################################################################
# 图片处理，单张图片不复制
################################################################################
def process_data(path):
    """
    图片处理，单张图片不复制

    :param
        path(str) -- 图片路径文件地址
    :return(tensor)
        图片数据，N X 3 X 224 X 224
    """
    # 图片处理
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize]
    )

    # 读取文件
    with open(path, 'rt') as f:
        lines = f.readlines()

    # 加载图片
    data = torch.zeros((0, 3, 224, 224))
    for img_path in lines:
        img_path = img_path.strip()
        # 加载图片
        img = Image.open(img_path)
        # 裁剪图片
        img = img.resize((300, 300))
        img = transform(img)
        img = img.view(1, 3, 224, 224)
        data = torch.cat((data, img), 0)

    return data

################################################################################
# 图片处理，单张图片复制
################################################################################
def process_data_agument(path, num = 16):
    """
    图片处理，单张图片不复制
    :param
        path(str) -- 图片路径地址
        num(int) -- 单张图片复制次数
    :return(tensor)
        图片数据，N X 3 X 224 X 224
    """
    # 图片处理
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize]
    )

    # 加载图片
    data = torch.zeros((0, 3, 224, 224))
    img = Image.open(path)
    # 裁剪图片
    img = img.resize((300, 300))
    for i in range(num):
        data = torch.cat((data, transform(img).view(1, 3, 224, 224)), 0)
    return data


################################################################################
# 模型预测，单张图片不复制
################################################################################
def predict(data, **kwargs):
    """
    模型预测，单张图片不复制
    :param
        data(tensor) -- 图片数据
    :kwargs

    :return(numpy)
        预测结果
    """
    # 选择运行的cpu或gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # 加载权重
        model.load_state_dict(torch.load(v))
        model = model.to(device)
        # 测试模式
        model.eval()
        models.append(model)

    # 使用平均策略获取集成模型输出
    data = data.to(device)
    output = None
    # 平均策略
    for model in models:
        if output is None:
            output = model(data).detach()
        else:
            output += model(data).detach()
    output = output / len(models)
    _, a = torch.max(output, 1)
    a = a.cpu().detach().numpy()

    # 预测结果
    return a


################################################################################
# 模型预测，单张图片复制
################################################################################
def predict_agument(data, **kwargs):
    """
        模型预测，单张图片复制
        :param
            data(tensor) -- 图片数据
        :kwargs

        :return(numpy)
            预测结果
    """
    # 选择运行的cpu或gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # 加载权重
        model.load_state_dict(torch.load(v))
        model = model.to(device)
        # 测试模式
        model.eval()
        models.append(model)

    # 使用平均策略获取集成模型输出
    data = data.to(device)
    sum = None
    # 平均策略
    for model in models:
        output = model(data)
        output = output.detach()
        val = torch.zeros(7)
        for i in range(output.size(0)):
            val = val + output[i]
        val = val / output.size(0)

        if sum is None:
            sum = val
        else:
            sum += val
    val = sum / len(models)
    _, a = torch.max(val, 0)

    return a.item()

################################################################################
# 可视化预测结果，单张图片不复制
################################################################################
def plot(path, **kwargs):
    """
    可视化预测结果，单张图片不复制
    :param
        path(str) --  图片路径文件地址
    :kwargs
    """
    # 模型预测
    cm_plot_labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    data = process_data(path)
    outcome = predict(data, kwargs)

    # 可视化预测结果
    title = []
    for val in outcome:
        title.append(cm_plot_labels[val])
    with open(path, 'rt') as f:
        lines = f.readlines()
    ims = np.zeros((0, 224, 224, 3))
    for line in lines:
        line = line.strip()
        img = Image.open(line)
        img = np.array(img)
        img = img[np.newaxis, :, :, :]
        ims = np.concatenate((ims, img), 0)
    plot_image(ims, title)


################################################################################
# 可视化预测结果，单张图片复制
################################################################################
def plot_agument(path, **kwargs):
    """
        可视化预测结果，单张图片复制
        :param
            path(str) --  图片路径文件地址
        :kwargs
    """
    # 模型预测
    cm_plot_labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    with open(path, 'rt') as f:
        lines = f.readlines()
    outcome = []
    for line in lines:
        line = line.strip()
        data = process_data_agument(line)
        output = predict_agument(data, kwargs)
        outcome.append(output)

    # 可视化预测结果
    title = []
    for val in outcome:
        title.append(cm_plot_labels[val])
    ims = np.zeros((0, 224, 224, 3))
    for line in lines:
        line = line.strip()
        img = Image.open(line)
        img = np.array(img)
        img = img[np.newaxis, :, :, :]
        ims = np.concatenate((ims, img), 0)

    plot_image(ims, title)


################################################################################
# 函数入口
################################################################################
if __name__ == '__main__':
    # 预测图片目录
    path = CONFIG.predict_dir

    # 模型配置 1)MFFNet 2)Inception 3)ResNet 4)DenseNet 5)SeNet
    args = {  # '1' : 'train_models/mffnet_model.pt',
        # '2': CONFIG.inception_model,
        # '3': CONFIG.resnet_model,
        '4': CONFIG.densenet_model,
        '5': CONFIG.senet_model}

    # 获取预测图片路径
    # image_path(path)
    path = os.path.join(path, 'image_path.txt')

    # 可视化预测结果，单张图片不复制
    plot(path, args)
    # 可视化预测结果，单张图片复制
    plot_agument(path, args)
