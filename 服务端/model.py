# -*- coding: <encoding name> -*-
"""
加载模型.
"""
from __future__ import print_function, division
from torchvision.models import resnet152, inception_v3
import torch.nn as nn
import pretrainedmodels # pip install pretrainedmodels
from efficientnet_pytorch import EfficientNet # pip install --upgrade efficientnet-pytorch
from torchsummary import summary
import torch

################################################################################
# 加载Resnet152模型，输入为224x224, 论文地址 https://arxiv.org/pdf/1512.03385v1.pdf
################################################################################
def resnet(pretrained = True):
    """
    加载Resnet152模型

    :param
        pretrained(bool) -- 是否预训练
    :return
        输出为7的Resnet152模型
    """
    # 加载模型
    model = resnet152(pretrained = pretrained)

    # 修改模型输出为7
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 7)

    return model


################################################################################
# 加载InceptionV3模型，输入为299X299, 论文地址 https://arxiv.org/pdf/1512.00567v3.pdf
################################################################################
def inception(pretrained = True):
    """
       加载InceptionV3模型

       :param
           pretrained(bool) -- 是否预训练
       :return
           输出为7的InceptionV3模型
       """
    # 加载模型
    model = inception_v3(pretrained = pretrained)

    # 修改模型输出为7
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 7)
    fc_features = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(fc_features, 7)

    return model


################################################################################
# 加载EfficientNet模型，输入为224X224, 论文地址 https://arxiv.org/pdf/1905.11946.pdf
################################################################################
def efficient(pretrained = True):
    """
       加载EfficientNet模型

       :param
           pretrained(bool) -- 是否预训练
       :return
           输出为7的EfficientNet模型
       """
    # 加载模型
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=7)
    else:
        model = EfficientNet.from_name('efficientnet-b7')
        # 修改模型输出为7
        fc_features = model._fc.in_features
        model._fc = nn.Linear(fc_features, 7)

    return model


################################################################################
# 函数入口
################################################################################
if __name__ == '__main__':
    # 选择在cpu或cuda运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # 可视化Resnet152
    # model = resnet(pretrained=False).to(device)
    # summary(model, (3, 224, 224))

    # # 可视化InceptionV3
    # model = inception(pretrained=False).to(device)
    # summary(model, (3, 299, 299))

    # 可视化EfficientNet
    model = efficient(pretrained=False).to(device)
    summary(model, (3, 224, 224))