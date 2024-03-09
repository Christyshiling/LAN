import os
from PIL import Image
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ResNet import Bottleneck, _resnet

# 定义的 Step1Network 和 Step2Network 类
# 定义神经网络类 Step1Network
class Step1Network(nn.Module):
    def __init__(self, image_height, image_width):
        super(Step1Network, self).__init__()
        # 卷积层
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),  # 3通道输入，32通道输出，3x3的卷积核
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2的最大池化层
        )
        # 全连接层
        self.fc_layer = nn.Sequential(
            nn.Linear(32 * ((image_height - 2) // 2) * ((image_width - 2) // 2), 64),  # 输入尺寸为 input_size，输出尺寸为 64
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(64, 6),  # 输入尺寸为 64，输出尺寸为 6
        )

    def forward(self, image):
        x = self.conv_layer(image)  # 通过卷积层处理输入图像
        x = x.reshape(x.size(0), -1)  # 将卷积层的输出展平为一维向量
        # x = torch.cat((x, params), dim=1)  # 将展平后的特征张量与给定的参数张量 params 拼接在一起
        # 将两个变量分开传
        output = self.fc_layer(x)  # 通过全连接层处理拼接后的特征张量，得到最终的输出
        return output  # 返回最终的输出



def Step2Network(num_classes=9, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [2, 2, 2, 2], num_classes, **kwargs)

def Step2Network_resnet50(num_classes=9, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


