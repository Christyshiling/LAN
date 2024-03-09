import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
from utility import calculate_params, reinhard_normalization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def work1(step1_network, optimizer1, criterion1, trainloader, testloader,
         num_epochs, path):
    loss1_list = []
    mse_list = []
    best_mse = 10.0
    for epoch in range(num_epochs):
        # 第一步训练阶段
        step1_network.train()
        running_loss1 = 0.0

        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            optimizer1.zero_grad()
            target_params = calculate_params(inputs.cpu())  # 计算params
            outputs = step1_network(inputs)

            # 将 target_params 删除，在进行定义是它本身乘以 params
            loss1 = criterion1(outputs, torch.tensor(target_params, dtype=torch.float).to(device))

            loss1.backward()
            optimizer1.step()

            running_loss1 += loss1.item()

        loss1_list.append(running_loss1 / len(trainloader))

        print(f"第 {epoch + 1} 时期损失: {running_loss1 / len(trainloader)}")

        total, mse = 0.0, 0.0

        step1_network.eval()
        with torch.no_grad():
            for data in testloader:
                inputs, _ = data
                inputs = inputs.to(device)
                target_params = calculate_params(inputs.cpu())
                outputs = step1_network(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                mse += criterion1(outputs, torch.tensor(target_params, dtype=torch.float).to(device))

        mse_res = mse / total
        if mse_res < best_mse:
            best_mse = mse_res
            torch.save(step1_network.state_dict(), os.path.join(path, 'step1.pth'))

        mse_list.append(mse_res.item())
        print(f"第 {epoch + 1} 时期，测试MSE: {mse_res}")

    df = pd.DataFrame({'epoch': list(range(num_epochs)), 'loss1': loss1_list, 'mse': mse_list})
    df.to_csv(os.path.join(path, 'step1.csv'), index=False)

    print("训练和测试完成！")


def work2(step1_network, step2_network, optimizer2, criterion2, trainloader, testloader,
         num_epochs, path):
    loss2_list = []
    accuracy_list = []
    best_accuracy = 0.0
    # path = 'result/log24_03_05_resnet18_data_lab'
    step1_network.load_state_dict(torch.load(os.path.join(path, 'step1.pth')))
    step1_network.eval()
    step2_network.train()

    for epoch in range(num_epochs):
        # 第一步训练阶段
        running_loss2 = 0.0

        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            params = step1_network(inputs)
            params = params.detach().cpu().numpy()
            target_mean = params[:, :3]
            target_std = params[:, 3:]
            # 使用 reinhard_normalization 函数将当前图像进行 Reinhard 归一化
            normalized_image = reinhard_normalization(inputs, target_mean, target_std)
            # normalized_image = normalized_image.to(device)

            optimizer2.zero_grad()

            outputs, _ = step2_network(normalized_image)
            # outputs, _ = step2_network(inputs)
            loss2 = criterion2(outputs, labels)

            loss2.backward()
            optimizer2.step()

            running_loss2 += loss2.item()

        loss2_list.append(running_loss2 / len(trainloader))

        print(f"第 {epoch + 1} 时期，第二步损失: {running_loss2 / len(trainloader)}")

        total, correct = 0.0, 0.0
        step2_network.eval()
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                params = step1_network(inputs)
                target_mean = params[:, :3]
                target_std = params[:, 3:]
                # 使用 reinhard_normalization 函数将当前图像进行 Reinhard 归一化
                normalized_image = reinhard_normalization(inputs, target_mean, target_std)
                outputs, _ = step2_network(normalized_image)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(step2_network.state_dict(), os.path.join(path, 'step2.pth'))

        accuracy_list.append(accuracy)
        print(f"第 {epoch + 1} 时期，测试准确率: {100 * accuracy}%")

    df = pd.DataFrame(
        {'epoch': list(range(num_epochs)), 'loss2': loss2_list, 'accuracy': accuracy_list})
    df.to_csv(os.path.join(path, 'step2.csv'), index=False)

    print("训练和测试完成！")

