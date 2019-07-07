# -*- coding: <encoding name> -*-
"""
加载模型.
"""
from __future__ import print_function, division
from torchvision.models import resnet152, densenet161, inception_v3
import torch.nn as nn
import pretrainedmodels

################################################################################
# 加载Resnet152模型
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
# 加载Densenet161模型
################################################################################
def densenet(pretrained = True):
    """
    加载Densenet161模型

    :param
        pretrained(bool) -- 是否预训练
    :return
        输出为7的Densenet161模型
    """
    # 加载模型
    model = densenet161(pretrained = pretrained)

    # 修改模型输出为7
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, 7)
    return model

################################################################################
# 加载InceptionV3模型，输入为299X299
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
# 加载SeNet154模型
################################################################################
def senet(pretrained = True):
    """
        加载SeNet154模型

        :param
            pretrained(bool) -- 是否预训练
        :return
            输出为7的SeNet154模型
    """
    # 加载模型
    if pretrained:
        model = pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')
    else:
        model = pretrainedmodels.__dict__['senet154'](num_classes=1000)

    # 修改模型输出为7
    fc_features = model.last_linear.in_features
    model.last_linear = nn.Linear(fc_features, 7)

    return model
