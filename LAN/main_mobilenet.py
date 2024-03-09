# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
import torchvision
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from network_v2 import Step1Network, Step2Network_resnet50
from utility import get_average_image_size
from train3 import work1, work2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 指定包含训练集和测试集子文件夹的根目录
# autodl-tmp/code/result/log24_01_31_bs64_lr0.0001_aug
root_directory = 'result/log24_02_02_aug_mobilenet_data'  # 请替换为你的根目录路径
if not os.path.exists(root_directory):
    os.makedirs(root_directory)

# 调用函数以获取训练集和测试集的平均图像尺寸
train_image_width, train_image_height, test_image_width, test_image_height = get_average_image_size("data_2")

# 输出训练集和测试集的平均图像尺寸
print(f"训练集平均图像宽度: {train_image_width}, 训练集平均图像高度: {train_image_height}")
print(f"测试集平均图像宽度: {test_image_width}, 测试集平均图像高度: {test_image_height}")

# 定义 Step1Network 和 Step2Network 的实例
num_classes = 8  # 有 8 个类别
step1_network = Step1Network(train_image_height, train_image_width).to(device)
step2_network = torchvision.models.mobilenet_v3_large(num_classes=num_classes).to(device)

# 设置用于训练和测试数据集的数据加载器
train_transform1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_transform2 = transforms.Compose([
    transforms.RandomChoice([transforms.Resize((224, 224)),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip(),
                             transforms.RandomRotation(degrees=30)]),
    # transforms.RandomAffine(degrees=0, translate=(0.0, 0.2), scale=[0.8, 1.2]),
    transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5)),
    transforms.ToTensor(),
    transforms.RandomErasing()])  # 如果需要，可以添加更多的数据转换

val_transform = transforms.Compose([transforms.ToTensor()])  # 如果需要，可以添加更多的数据转换

train_batch_size = 64
test_batch_size = 128
train_dataset1 = datasets.ImageFolder(root="data/train", transform=train_transform1)
trainloader1 = DataLoader(train_dataset1, batch_size=train_batch_size, shuffle=True)
train_dataset2 = datasets.ImageFolder(root="data/train", transform=train_transform2)
trainloader2 = DataLoader(train_dataset2, batch_size=train_batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root="data/test", transform=val_transform)
testloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# 为两个网络定义优化器和损失函数
lr1 = 0.0001
lr2 = 0.0001
optimizer1 = optim.Adam(step1_network.parameters(), lr=lr1)
optimizer2 = optim.AdamW(step2_network.parameters(), lr=lr2, betas=(0.9, 0.999), weight_decay=0.01)
criterion1 = nn.MSELoss()  # 为第一步选择适当的损失函数
criterion2 = nn.CrossEntropyLoss()  # 第二步多类别分类任务的交叉熵损失

num_epochs1 = 20
work1(step1_network, optimizer1, criterion1, trainloader1, testloader, num_epochs1,
      root_directory)

num_epochs2 = 30
work2(step1_network, step2_network, optimizer2, criterion2, trainloader2, testloader, num_epochs2,
      root_directory)
