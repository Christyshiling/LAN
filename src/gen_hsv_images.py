import cv2
import os

# 定义输入和输出文件夹路径
input_folder = "data/train"
output_folder = "data_hsv/train"

# 检查并创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的每一个文件
for dir in os.listdir(input_folder):
    input_dir = os.path.join(input_folder, dir)
    output_dir = os.path.join(output_folder, dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        # 只处理.jpg或.png等图像文件（根据实际情况调整后缀）
        if filename.endswith(".tif"):
            # 构建输入和输出文件的完整路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"lab_{filename}")

            # 读取图像
            img_rgb = cv2.imread(input_path)

            # 确保图像已成功读取
            if img_rgb is not None:
                # 将RGB图像转换为LAB图像
                img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

                # 保存转换后的LAB图像
                cv2.imwrite(output_path, img_lab)

print("所有图片的RGB转LAB操作已完成！")
