# Code modified from the AlexNet implementation of torchvision(https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)

import torch
import logging
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .qa import Quantization

_name_translation = {
    'features.0': 'features_0.0',
    'features.3': 'features_1.1',
    'features.6': 'features_2.1',
    'features.8': 'features_2.3',
    'features.10': 'features_2.5',
    'classifier.1': 'classifier_0.1',
    'classifier.4': 'classifier_1.1',
    'classifier.6': 'classifier_1.3'
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, QA_flag=False, QA_bias=None, QA_values=None, QA_beta=None,
                 QA_outlier_gamma=0.001):
        super(AlexNet, self).__init__()
        self.ac_quan_values = QA_values
        self.ac_quan_bias = QA_bias
        self.ac_beta = QA_beta
        self.QA_flag = QA_flag
        self.QA_inited = False
        self.count = 0

        if num_classes == 1000:
            self.c = [64, 192, 384, 256, 256]
            self.ks = [11, 5, 3, 3, 3]
            self.s = [4, 1, 1, 1, 1]
            self.p = [2, 2, 1, 1, 1]
            self.pool_ks = 3
        else:
            self.c = [64, 192, 384, 256, 256]
            self.ks = [3, 3, 3, 3, 3]
            self.s = [2, 1, 1, 1, 1]
            self.p = [1, 1, 1, 1, 1]
            self.pool_ks = 2

        self.features_0 = nn.Sequential(
            nn.Conv2d(3, self.c[0], kernel_size=self.ks[0], stride=self.s[0], padding=self.p[0]),
            nn.ReLU(inplace=True),
        )
        self.features_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=self.pool_ks, stride=2),
            nn.Conv2d(self.c[0], self.c[1], kernel_size=self.ks[1], stride=self.s[1], padding=self.p[1]),
            nn.ReLU(inplace=True),
        )
        self.features_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=self.pool_ks, stride=2),
            nn.Conv2d(self.c[1], self.c[2], kernel_size=self.ks[2], stride=self.s[2], padding=self.p[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c[2], self.c[3], kernel_size=self.ks[3], stride=self.s[3], padding=self.p[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c[3], self.c[4], kernel_size=self.ks[4], stride=self.s[4], padding=self.p[4]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=self.pool_ks, stride=2)
        self.classifier_0 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(256 * 6 * 6 if num_classes == 1000 else 256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
        )
        self.classifier_1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        if self.QA_flag:
            self.quan0 = Quantization(quant_values=self.ac_quan_values, outlier_gamma=QA_outlier_gamma)
            self.quan1 = Quantization(quant_values=self.ac_quan_values, outlier_gamma=QA_outlier_gamma)
            self.quan2 = Quantization(quant_values=self.ac_quan_values, outlier_gamma=QA_outlier_gamma)
            self.quan3 = Quantization(quant_values=self.ac_quan_values, outlier_gamma=QA_outlier_gamma)
            self.count = 4

    def forward(self, x, input_ac_T=1):
        x = self.features_0(x)
        if self.QA_flag:
            x = self.quan0(x, input_ac_T)
        x = self.features_1(x)
        if self.QA_flag:
            x = self.quan1(x, input_ac_T)
        x = self.features_2(x)
        if self.QA_flag:
            x = self.quan2(x, input_ac_T)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_0(x)
        if self.QA_flag:
            x = self.quan3(x, input_ac_T)
        x = self.classifier_1(x)
        return x


def alexnet(pretrained=False, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        if 'num_classes' in kwargs.keys() and kwargs['num_classes'] != 1000:
            logging.info('Can\'t load pre-trained model because target classes number isn\'t 1000(ImageNet).')
            return model
        model_path = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
        state_dict = load_state_dict_from_url(model_path)
        new_state_dict = dict()
        # Names should be translated to meet
        for key, value in state_dict.items():
            split_name = key.split('.')
            module_name = split_name[0] + '.' + split_name[1]
            if module_name in _name_translation.keys():
                new_state_dict[_name_translation[module_name] + key[len(module_name):]] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
    return model
