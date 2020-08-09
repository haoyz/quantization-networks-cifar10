#!/usr/bin/env python
# -*- coding: utf-8 -*-
# resnet.py is used to quantize the weight and activation of ResNet-18.
from __future__ import print_function, absolute_import

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
from .qa import *
import pdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
           'resnet1202']

# Pre-trained models are from official pytorch model zoo and github.com/akamaster/pytorch_resnet_cifar10
# https://pytorch.org/docs/stable/model_zoo.html by PyTorch contributors.
# https://github.com/akamaster/pytorch_resnet_cifar10 by Yerlan Idelbayev.
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet20': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20-12fca82f.th',
    'resnet32': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet32-d509ac18.th',
    'resnet44': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet44-014dd654.th',
    'resnet56': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet56-4bfd9763.th',
    'resnet110': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet110-1d1ed7c2.th',
    'resnet1202': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet1202-f3b1deed.th'
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, QA_flag=False, QA_values=None,
                 QA_outlier_gamma=0.001):
        super(BasicBlock, self).__init__()
        self.QA_flag = QA_flag
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.ac_T = 1

        if self.QA_flag:
            self.quan1 = Quantization(quant_values=QA_values, outlier_gamma=QA_outlier_gamma)
            self.quan2 = Quantization(quant_values=QA_values, outlier_gamma=QA_outlier_gamma)

    def set_activation_T(self, activation_T):
        self.ac_T = activation_T

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        # Quantization after ReLU activation
        if self.QA_flag:
            out = self.quan1(out, self.ac_T)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        # Quantization after ReLU activation
        if self.QA_flag:
            out = self.quan2(out, self.ac_T)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, QA_flag=False, QA_values=None,
                 QA_outlier_gamma=0.001):
        super(Bottleneck, self).__init__()
        self.QA_flag = QA_flag
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample
        self.stride = stride
        self.ac_T = 1

        if self.QA_flag:
            self.quan1 = Quantization(quant_values=QA_values, outlier_gamma=QA_outlier_gamma)
            self.quan2 = Quantization(quant_values=QA_values, outlier_gamma=QA_outlier_gamma)
            self.quan3 = Quantization(quant_values=QA_values, outlier_gamma=QA_outlier_gamma)

    def set_activation_T(self, activation_T):
        self.ac_T = activation_T

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        # Quantization after ReLU activation
        if self.QA_flag:
            out = self.quan1(out, self.ac_T)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        # Quantization after ReLU activation
        if self.QA_flag:
            out = self.quan2(out, self.ac_T)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        # Quantization after ReLU activation
        if self.QA_flag:
            out = self.quan3(out, self.ac_T)

        return out


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, QA_flag=False, QA_values=None, QA_outlier_gamma=0.001):
        """
        Constructor function of a quantized ResNet.
        :param block: The basic block class.
        :param layers: A list of integers. The number of layers in each stage.
        :param num_classes: The number of output classes.
        :param QA_flag: A boolean value. Set to True to enable quantization. This will pass down to each block.
        :param QA_values: A list of integers, for example, [0, 1] The exact values to quantize to.
            Its length depends on the `ak` argument. (l = 2^ak)
        :param QA_outlier_gamma: A floating number in range (0, 0.5). The gamma value of outliers in activation quantization.
        """
        self.QA_values = QA_values
        self.count = 0
        super(ResNet, self).__init__()
        self.QA_flag = QA_flag
        self.QA_inited = False

        assert len(layers) == 3 or len(layers) == 4, 'Only ResNets with 3 or 4 layers are supported'
        if len(layers) == 3:
            # ResNets for CIFAR-10
            self.inplanes = 16
            num_planes = [16, 32, 64]
            self.conv1 = nn.Conv2d(3, num_planes[0], kernel_size=3, stride=1, padding=1, bias=False)
        else:
            # ResNets for ImageNet
            self.inplanes = 64
            num_planes = [64, 128, 256, 512]
            self.conv1 = nn.Conv2d(3, num_planes[0], kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(num_planes[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.QA_flag:
            # print(self.count)
            self.quan0 = Quantization(quant_values=self.QA_values, outlier_gamma=QA_outlier_gamma)
            self.count += 1

        if len(layers) == 3:
            # ResNets for CIFAR-10
            self.layer1 = self._make_layer(block, num_planes[0], layers[0], stride=1, downsample_option='pad0',
                                           QA_flag=self.QA_flag, QA_outlier_gamma=QA_outlier_gamma)
            self.layer2 = self._make_layer(block, num_planes[1], layers[1], stride=2, downsample_option='pad0',
                                           QA_flag=self.QA_flag, QA_outlier_gamma=QA_outlier_gamma)
            self.layer3 = self._make_layer(block, num_planes[2], layers[2], stride=2, downsample_option='pad0',
                                           QA_flag=self.QA_flag, QA_outlier_gamma=QA_outlier_gamma)
        elif len(layers) == 4:
            # ResNets for ImageNet
            self.layer1 = self._make_layer(block, num_planes[0], layers[0], stride=1, downsample_option='conv',
                                           QA_flag=self.QA_flag)
            self.layer2 = self._make_layer(block, num_planes[1], layers[1], stride=2, downsample_option='conv',
                                           QA_flag=self.QA_flag)
            self.layer3 = self._make_layer(block, num_planes[2], layers[2], stride=2, downsample_option='conv',
                                           QA_flag=self.QA_flag)
            self.layer4 = self._make_layer(block, num_planes[3], layers[3], stride=2, downsample_option='conv',
                                           QA_flag=self.QA_flag)

        self.fc = nn.Linear(num_planes[-1] * block.expansion, num_classes)

        self.set_params()

    #        for m in self.modules():
    #            if isinstance(m, nn.Conv2d):
    #                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #                m.weight.data.normal_(0, math.sqrt(2. / n))
    #            elif isinstance(m, nn.BatchNorm2d):
    #                m.weight.data.fill_(1)
    #                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, downsample_option='conv', QA_flag=False,
                    QA_outlier_gamma=0.001):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if downsample_option == 'conv':
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            elif downsample_option == 'pad0':
                downsample = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, planes // 4, planes // 4], "constant", 0)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, QA_flag=QA_flag, QA_values=self.QA_values,
                            QA_outlier_gamma=QA_outlier_gamma))
        self.inplanes = planes * block.expansion
        self.count += 2
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, QA_flag=QA_flag, QA_values=self.QA_values,
                                QA_outlier_gamma=QA_outlier_gamma))
            self.count += 2

        return nn.Sequential(*layers)

    def set_resnet_ac_T(self, input_ac_T):
        for m in self.layer1:
            if isinstance(m, BasicBlock):
                m.set_activation_T(input_ac_T)
        for m in self.layer2:
            if isinstance(m, BasicBlock):
                m.set_activation_T(input_ac_T)
        for m in self.layer3:
            if isinstance(m, BasicBlock):
                m.set_activation_T(input_ac_T)
        if hasattr(self, 'layer4'):
            for m in self.layer4:
                if isinstance(m, BasicBlock):
                    m.set_activation_T(input_ac_T)

    def set_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    init.constant(m.bias, 0)

    # def set_params(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.kaiming_normal(m.weight, mode='fan_in')
    #             if m.bias is not None:
    #                 init.constant(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init.constant(m.weight, 1)
    #             init.constant(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             init.normal(m.weight, std=0.001)
    #             if m.bias is not None:
    #                 init.constant(m.bias, 0)

    def forward(self, x, input_ac_T=0):
        if self.QA_flag:
            self.set_resnet_ac_T(input_ac_T)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        if hasattr(self, 'layer4'):
            x = self.maxpool(x)

        if self.QA_flag:
            x = self.quan0(x, input_ac_T)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if hasattr(self, 'layer4'):
            x = self.layer4(x)

        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet_imagenet(name, block, layers, pretrained, **kwargs):
    """Constructs a ResNet model for ImageNet classification.

    Args:
        name: A string. Name of the ResNet model. Choose from 'resnet18', 'resnet34', 'resnet50'
        block: The building block class.
        layers: A list of 3 integers. Number of layers in the model.
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print('Loading pre-trained model from PyTorch model zoo')
        state_dict = model_zoo.load_url(model_urls[name])
        if 'num_classes' in kwargs.keys() and kwargs['num_classes'] != 1000:
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)
    return model


def _resnet_cifar10(name, block, layers, pretrained, **kwargs):
    """Constructs a ResNet model for CIFAR-10 classification.

    Args:
        name: A string. Name of the ResNet model. Choose from 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202'
        block: The building block class.
        layers: A list of 3 integers. Number of layers in the model.
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print('Loading pre-trained model from Yerlan Idelbayev\'s model zoo')
        state_dict = model_zoo.load_url(model_urls[name])['state_dict']
        # Remove 'module.' prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        if 'num_classes' in kwargs.keys() and kwargs['num_classes'] != 10:
            new_state_dict.pop('linear.weight')
            new_state_dict.pop('linear.bias')
        else:
            new_state_dict['fc.weight'] = new_state_dict.pop('linear.weight')
            new_state_dict['fc.bias'] = new_state_dict.pop('linear.bias')
        try:
            model.load_state_dict(new_state_dict)
        except RuntimeError as ex:
            print(ex)
    return model


def resnet18(pretrained=True, **kwargs):
    return _resnet_imagenet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, **kwargs)


def resnet34(pretrained=True, **kwargs):
    return _resnet_imagenet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, **kwargs)


def resnet50(pretrained=True, **kwargs):
    return _resnet_imagenet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def resnet20(pretrained=True, **kwargs):
    return _resnet_cifar10('resnet20', BasicBlock, [3, 3, 3], pretrained, **kwargs)


def resnet32(pretrained=True, **kwargs):
    return _resnet_cifar10('resnet32', BasicBlock, [5, 5, 5], pretrained, **kwargs)


def resnet44(pretrained=True, **kwargs):
    return _resnet_cifar10('resnet44', BasicBlock, [7, 7, 7], pretrained, **kwargs)


def resnet56(pretrained=True, **kwargs):
    return _resnet_cifar10('resnet56', BasicBlock, [9, 9, 9], pretrained, **kwargs)


def resnet110(pretrained=True, **kwargs):
    return _resnet_cifar10('resnet110', BasicBlock, [18, 18, 18], pretrained, **kwargs)


def resnet1202(pretrained=True, **kwargs):
    return _resnet_cifar10('resnet1202', BasicBlock, [200, 200, 200], pretrained, **kwargs)
