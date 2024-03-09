# LAN
Learnable Automatic Stain Normalization For Histology Image Analysis

## 1、参数设置说明：
优化函数：第一阶段为Adam，第二阶段为AdamW  
损失函数：第一阶段为MSE、第二阶段为交叉熵  
学习率：0.0001  
Batch size: 64  
Epoch: 两个阶段均为20  

## 2、文件说明：
baseline_xxx.py，其中xxx表示神经网络模型  
main_xxx.py，xxx表示每种网路模型以及本方法的改进  
gen_hsv_images.py，为生成HSV文件的方式  
ResNet.py，为ResNet结构代码  
network_v2.py，为整体模型的结构  
utility.py，为模型训练中所需要调用方法  
train3.py，为整体框架的训练过程  
reconstruct_image.py，为模型重构图像方式  
