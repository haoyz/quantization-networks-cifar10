import torch.nn as nn
import torch.nn.functional as F
from models.qa import Quantization


class SimpleNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, QA_flag=False, QA_values=None, QA_outlier_gamma=0.001):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.QA_flag = QA_flag
        self.QA_values = QA_values
        self.QA_inited = False

        if QA_flag:
            assert isinstance(QA_values, list)
            self.quan1 = Quantization(QA_values, outlier_gamma=QA_outlier_gamma)
            self.quan2 = Quantization(QA_values, outlier_gamma=QA_outlier_gamma)
            self.quan3 = Quantization(QA_values, outlier_gamma=QA_outlier_gamma)
            self.quan4 = Quantization(QA_values, outlier_gamma=QA_outlier_gamma)

    def forward(self, x, input_ac_T):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if self.QA_flag:
            x = self.quan1(x, input_ac_T)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        if self.QA_flag:
            x = self.quan2(x, input_ac_T)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        if self.QA_flag:
            x = self.quan3(x, input_ac_T)
        x = self.avg_pool(x)
        x = x.view((-1, 128))
        x = self.fc1(x)
        x = F.sigmoid(x)
        if self.QA_flag:
            x = self.quan4(x, input_ac_T)
        x = self.fc2(x)
        return x


def simplenet(pretrained=False, num_classes=10, **kwargs):
    return SimpleNet(pretrained, num_classes, **kwargs)
