# -*- coding: <encoding name> -*-
"""
自定义模型MFFNet(Multiple Feature Fusion Network)
"""
from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as f
import torch
from torchvision.models import vgg16, densenet121

################################################################################
# 自定义模型MFFNet(Multiple Feature Fusion Network)
################################################################################
class MFFNet(nn.Module):
    """
    自定义模型MFFNet(Multiple Feature Fusion Network)
    """
    def __init__(self, num_classes = 7):
        super(MFFNet, self).__init__()

        self.features = nn.Sequential(*(list(vgg16().children())[0][ : -8]))

        self.bb = nn.Sequential(
                        nn.MaxPool2d(kernel_size = 2, stride = 2),
                        _BasicBlock(512),
                        _BasicConv2d(512, 256, kernel_size = 3, stride = 1, padding = 1),

                        nn.MaxPool2d(kernel_size = 2, stride = 2),
                        _BasicBlock(256),
                        _BasicConv2d(256, 128, kernel_size = 3, stride = 1, padding = 1),

                        nn.AdaptiveAvgPool2d((1, 1))
                        )

        self.classifier = nn.Sequential( \
                        nn.Linear(128, num_classes)
                        )

        # initialize weight of layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 28 x 28
        x = self.bb(x)
        # N x 128 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 128
        x = self.classifier(x)
        # N x num_classes
        x = f.softmax(x, dim = 1)

        return x

'''
achieve basicblock which consists three layers
'''
class _BasicBlock(nn.Module):

    def __init__(self, in_channels):
        super(_BasicBlock, self).__init__()

        self.bc_1 = _BasicConv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bc_2 = _BasicConv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bc_3 = _BasicConv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)


    def forward(self, x):
        bc_1 = self.bc_1(x)
        bc_2 = self.bc_2(x)
        bc_2 = x + bc_2
        bc_3 = self.bc_3(bc_2)
        bc_3 = bc_3 + bc_1
        return bc_3

'''
conv -> bn -> relu
'''
class _BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(_BasicConv2d, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        # self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)

    def forward(self, x):
        x = self.bn(x)
        # x = self.instance_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x