# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from train_resnet import work

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 指定包含训练集和测试集子文件夹的根目录
root_directory = 'result/log24_02_22_swin_data2_baseline'  # 请替换为你的根目录路径
if not os.path.exists(root_directory):
   os.makedirs(root_directory)

# 定义 Step1Network 和 Step2Network 的实例
num_classes = 8  # 有 8 个类别
model = models.swin_b(num_classes)
model.to(device)

# 设置用于训练和测试数据集的数据加载器
train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # 如果需要，可以添加更多的数据转换

val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # 如果需要，可以添加更多的数据转换

batch_size = 64
test_bs = 128
train_dataset = datasets.ImageFolder(root="data_2/train", transform=train_transform)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root="data_2/test", transform=val_transform)
testloader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False)

# 为两个网络定义优化器和损失函数
lr1 = 1e-4
optimizer1 = optim.Adam(model.parameters(), lr=lr1)
criterion1 = nn.CrossEntropyLoss()  # 第二步多类别分类任务的交叉熵损失

num_epochs = 20

work(model, optimizer1, criterion1, trainloader, testloader, num_epochs,
     root_directory)
