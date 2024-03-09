# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from network_v2 import Step2Network_resnet50, Step1Network, Step2Network
from utility import get_average_image_size
from train_resnet import work

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 指定包含训练集和测试集子文件夹的根目录
root_directory = 'result/log24_03_06_resnet18_data_hsv'  # 请替换为你的根目录路径
if not os.path.exists(root_directory):
   os.makedirs(root_directory)

# 调用函数以获取训练集和测试集的平均图像尺寸
train_image_width, train_image_height, test_image_width, test_image_height = get_average_image_size('data_hsv')

# 输出训练集和测试集的平均图像尺寸
print(f"训练集平均图像宽度: {train_image_width}, 训练集平均图像高度: {train_image_height}")
print(f"测试集平均图像宽度: {test_image_width}, 测试集平均图像高度: {test_image_height}")

# 定义 Step1Network 和 Step2Network 的实例
num_classes = 8  # 有 8 个类别
step1_network = Step1Network(train_image_height, train_image_width).to(device)
# step2_network = models.vit_b_16(num_classes=num_classes).to(device)
step2_network = Step2Network(num_classes).to(device)

# 设置用于训练和测试数据集的数据加载器
train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # 如果需要，可以添加更多的数据转换

val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # 如果需要，可以添加更多的数据转换

batch_size = 64
test_bs = 128
train_dataset = datasets.ImageFolder(root="data_hsv/train", transform=train_transform)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root="data_hsv/test", transform=val_transform)
testloader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False)

# 为两个网络定义优化器和损失函数
lr1 = 1e-4
optimizer1 = optim.Adam(step2_network.parameters(), lr=lr1)
criterion1 = nn.CrossEntropyLoss()  # 第二步多类别分类任务的交叉熵损失

num_epochs = 20

work(step2_network, optimizer1, criterion1, trainloader, testloader, num_epochs,
     root_directory)
