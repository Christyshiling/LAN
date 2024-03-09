import numpy as np
from PIL import Image
import os
import torch


def reinhard_normalization(image, target_mean, target_std):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j, :, :] = (image[i, j, :, :]-target_mean[i, j])/(target_std[i, j]+1e-7)

    return image


# def reinhard_normalization(image, target_mean, target_std):
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             mean, std = torch.mean(image[i, j, :, :]), torch.std(image[i, j, :, :])
#             image[i, j, :, :] = ((image[i, j, :, :] - mean) * (target_std[i, j] / std)) + target_mean[i, j]
#             # image[i, j, :, :] = (image[i, j, :, :]-target_mean[i, j])/target_std[i, j]
#
#     return image


def calculate_params(image_data):
    # 计算均值和标准差
    dd = []
    for image in image_data:
        image = image.numpy()  # 将 PyTorch 张量转换为 NumPy 数组
        mean_r = np.mean(image[0, :, :])
        mean_g = np.mean(image[1, :, :])
        mean_b = np.mean(image[2, :, :])

        std_r = np.std(image[0, :, :])
        std_g = np.std(image[1, :, :])
        std_b = np.std(image[2, :, :])

        dd.append([mean_r, mean_g, mean_b, std_r, std_g, std_b])

    # 将计算的参数添加到全局变量
    dd = np.array(dd).reshape(-1, 6)
    return dd



# 定义一个函数来获取平均图像尺寸
def get_average_image_size(root_dir):
    total_train_width = 0
    total_train_height = 0
    total_train_images = 0

    total_test_width = 0
    total_test_height = 0
    total_test_images = 0

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.tif') or file.endswith('.tiff'):
                file_path = os.path.join(subdir, file)
                image = Image.open(file_path)
                width, height = image.size

                if 'train' in subdir.lower():
                    total_train_width += width
                    total_train_height += height
                    total_train_images += 1
                elif 'test' in subdir.lower():
                    total_test_width += width
                    total_test_height += height
                    total_test_images += 1

    # 计算平均尺寸
    train_average_width = total_train_width // total_train_images
    train_average_height = total_train_height // total_train_images
    test_average_width = total_test_width // total_test_images
    test_average_height = total_test_height // total_test_images

    return train_average_width, train_average_height, test_average_width, test_average_height