import numpy as np
import torch
import os

from network_v2 import Step1Network
from PIL import Image
from utility import reinhard_normalization

img_dir = './imges'
target_dir = './target'

cls_list = os.listdir(img_dir)
for cls in cls_list:
    cls_dir = os.path.join(img_dir, cls)
    target_dir_cls = os.path.join(target_dir, cls)
    if not os.path.exists(target_dir_cls):
        os.makedirs(target_dir_cls)
    for img_name in os.listdir(cls_dir):
        img_path = os.path.join(img_dir, cls, img_name)
        print(img_path)
        img = Image.open(img_path)
        img = img.convert('RGB')

        img_arr = np.array(img)
        tensor = torch.from_numpy(img_arr).permute(2, 0, 1) / 255.0

        train_image_height, train_image_width = tensor.shape[1], tensor.shape[2]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = tensor.to(device)
        tensor = tensor.unsqueeze(0)
        step1_network = Step1Network(train_image_height, train_image_width).to(device)
        step1_network.load_state_dict(torch.load(os.path.join('./result/log24_02_02_aug_resnet50', 'step1.pth')))  # 任意选取模型
        step1_network.eval()

        params = step1_network(tensor)
        params = params.detach().cpu().numpy()
        target_mean = np.squeeze(params[:, :3])
        target_std = np.squeeze(params[:, 3:])

        # reinhard_normalization
        img_norm = img_arr / 255.0
        for i in range(img_norm.shape[-1]):
            mean, std = np.mean(img_norm[:, :, i]), np.std(img_norm[:, :, i])
            img_norm[:, :, i] = ((img_norm[:, :, i] - mean)*(target_std[i] / std)) + target_mean[i]

        normalized_image = (img_norm * 255.0).astype(np.uint8)
        # normalized_image = reinhard_normalization(tensor, target_mean, target_std)
        # normalized_image = normalized_image.cpu().clone()
        # normalized_image = normalized_image.squeeze(0)
        # normalized_image = normalized_image.permute(1, 2, 0)
        # image = normalized_image.numpy()
        # image = (image * 255).astype(np.uint8)
        tar_path = os.path.join(target_dir_cls, img_name)
        image = Image.fromarray(normalized_image)
        image.save(tar_path)

print('done')
