# 文件名: test.py (最终独立版 - 内置数据集)
#08 train1 dataset1 conf
import os
import shutil
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
import pickle

# 1. 导入您的模型和配置文件
#Wavelet0808
from Wavelet0808 import UNetWavelet
import config as cfg


# =========================================================================
# 2. [内置] 自定义测试数据集类
#    这个类现在是 test.py 的一部分，不再需要外部文件。
# =========================================================================
class CustomTestDataset(Dataset):
    """
    一个用于加载测试图像的数据集类。
    它会遍历指定目录下的所有图像，并返回处理后的图像及其原始文件名。
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # 筛选出目录中所有支持的图像文件
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(supported_formats)]
        if not self.image_files:
            print(f"警告: 在目录 '{data_dir}' 中未找到任何支持的图像文件。")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # 获取原始文件名（这是一个字符串，至关重要！）
        filename = self.image_files[index]

        # 构建完整的文件路径
        img_path = os.path.join(self.data_dir, filename)

        # 使用 OpenCV 读取图像为灰度图
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"警告: 无法读取图像文件 {img_path}。将跳过。")
            # 返回一个占位符，或者在 collate_fn 中处理 None
            return torch.zeros(1, 64, 64), "invalid_image.png"

        # 应用预定义的转换（例如，ToTensor）
        if self.transform:
            image = self.transform(image)

        # 返回 (处理后的图像张量, 原始文件名字符串)
        return image, filename


def test_denoising():
    """
    执行图像降噪测试的主函数。
    """
    res_dir = cfg.res_dir
    if os.path.exists(res_dir):
        print(f"输出目录 '{res_dir}' 已存在，正在清空...")
        shutil.rmtree(res_dir)
    os.makedirs(res_dir)
    print(f"结果将保存在: '{res_dir}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'正在使用设备: {device}')

    transform = transforms.ToTensor()

    # 使用内置的数据集类
    test_dir = cfg.test_dir
    test_dataset = CustomTestDataset(test_dir, transform=transform)

    if len(test_dataset) == 0:
        print("测试集为空，程序退出。")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f'\n测试样本总数: {len(test_dataset)}')
    print(f'测试批次总数: {len(test_loader)} (批大小={cfg.test_bs})')

    model = UNetWavelet(n_classes=1, padding=True, batch_norm=True, wf=6).to(device)

    ckpt_path = os.path.join(cfg.models_dir, cfg.ckpt)
    if not os.path.exists(ckpt_path):
        print(f"错误: 模型权重文件未找到于 {ckpt_path}")
        return

    print(f"正在加载模型: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        print("\n警告: 使用 'weights_only=True' 安全加载模式失败。将切换到常规加载模式。")
        ckpt = torch.load(ckpt_path, map_location=device)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.to(device)

    start_time = time.time()
    print('\n开始进行图像降噪处理...')

    model.eval()
    with torch.no_grad():
        for noisy_imgs, filenames in tqdm(test_loader, desc="正在处理批次"):
            noisy_imgs = noisy_imgs.to(device)

            if noisy_imgs.dim() == 3:
                noisy_imgs = noisy_imgs.unsqueeze(1)

            model_outputs = model(noisy_imgs)
            out = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs

            for i in range(len(out)):
                denoised_tensor = out[i]
                output_filename = filenames[i]

                denoised_np = denoised_tensor.squeeze().cpu().numpy()
                denoised_np = np.clip(denoised_np, 0.0, 1.0) * 255.0
                denoised_img = denoised_np.astype(np.uint8)

                save_path = os.path.join(res_dir, output_filename)
                cv2.imwrite(save_path, denoised_img)

    end_time = time.time()
    total_time = end_time - start_time

    print(f'\n\n处理完成！结果已保存至 "{res_dir}" 目录。')
    print(f'总耗时: {total_time:.2f} 秒')
    print(f'平均每张图片耗时: {total_time / len(test_dataset):.4f} 秒')


if __name__ == '__main__':
    test_denoising()