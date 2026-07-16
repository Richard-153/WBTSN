import numpy as np
import os, glob, cv2, sys
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from pytorch_wavelets import DWTForward
import torch.nn.functional as F


# 归一化到 [0, 255] 并转换为uint8
# def process_subband(subband):
#     subband = (subband - subband.min()) / (subband.max() - subband.min())
#     return subband
class DAE_dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        # print(self.data_dir)
        # self.mask_dir = mask_dir  # 新增掩膜目录参数
        # print(self.mask_dir)
        self.transform = transform
        # self.conv = nn.Conv2d(in_channels=4,out_channels=1,kernel_size=1,stride=1)
        # 加载图像和噪声图像路径
        self.imgs_data = self.get_data(os.path.join(self.data_dir, 'imgs'))
        self.noisy_imgs_data = self.get_data(os.path.join(self.data_dir, 'noisy'))
        # self.mask_imgs_data = self.get_data(os.path.join(self.data_dir, 'mask'))
        # self.relu =nn.ReLU()
        # self.weights = nn.Parameter(torch.ones(3))
        # self.conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1)

    def get_data(self, data_path):
        return [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.bmp'))]

    def __getitem__(self, index):
        # 读取图像、噪声图像和掩膜（均为灰度图）
        img = cv2.imread(self.imgs_data[index], 0)  # 原始图像（干净图像）
        noisy_img = cv2.imread(self.noisy_imgs_data[index], 0)  # 输入的噪声图像
        # print(noisy_img.shape)
        if noisy_img is None:
            raise ValueError(f"Failed to read image: {self.noisy_imgs_data[index]}")
  #      yh_reshaped = yh[0].flatten(1, 2)
  #       mask = cv2.imread(self.mask_imgs_data[index], 0) if self.mask_imgs_data[index] is not None else None  # 掩膜
        # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # 应用数据预处理（例如归一化）

        img = self.transform(img)
        noisy_img = self.transform(noisy_img)
        # mask = self.transform(mask)

        #wavlet
        # dwt = DWTForward(J=1, wave='haar')
        # noisy_4d = noisy_img.unsqueeze(0)
        # yl, yh = dwt(noisy_4d)

        # 分解三个方向
        # lh = yh[0][:,:, 0, :, :]  # 水平细节 (LH)
        # hl = yh[0][:,:, 1, :, :]  # 垂直细节 (HL)
        # hh = yh[0][:,:, 2, :, :]  # 对角线细节 (HH)
        #
        # lh = process_subband(lh)
        # hl  = process_subband(hl)
        # hh = process_subband(hh)
        # wavelet = np.concatenate([lh,hl,hh,yl],axis=1)
        # # print(upsampled.shape)
        # yl = torch.from_numpy(yl).float()
        # lh = torch.from_numpy(lh).float()
        # hl = torch.from_numpy(hl).float()
        # hh = torch.from_numpy(hh).float()

        # yl = F.interpolate(yl, size=(512, 640), mode='bilinear', align_corners=False)
        # lh = F.interpolate(lh, size=(512, 640), mode='bilinear', align_corners=False)
        # hl = F.interpolate(hl, size=(512, 640), mode='bilinear', align_corners=False)
        # hh = F.interpolate(hh, size=(512, 640), mode='bilinear', align_corners=False)
        #
        # yl = yl[ 0, : , : , :]
        # lh = lh[ 0, : , : , :]
        # hl = hl[ 0, : , : , :]
        # hh = hh[ 0, : , : , :]
        #
        # wavelet = np.concatenate([lh, hl, hh, yl], axis=0)
        # wavelet = process_subband(wavelet)
        # wavelet = self.relu(torch.from_numpy(wavelet).float())
        # # wavelet = torch.from_numpy(wavelet)
        # wavelet = wavelet.detach().numpy()
        # # wavelet = self.transform(wavelet)
        # wavelet = torch.from_numpy(wavelet)
        # wavelet = self.conv(wavelet)
        # wavelet = self.transform(wavelet)
        # print('wavelet',wavelet.shape)
        # print('noisy_img',noisy_img.shape)

        # print(wavlet.shape)
        # combined_input = torch.cat([noisy_img], dim=0)  # (3,H, W)

        # 假设你的训练数据集实例是 train_dataset
        # torch.save(DAE_dataset.weights, 'trained_weights.pth')
        return noisy_img, img

    def __len__(self):
        return len(self.imgs_data)


class custom_test_dataset(Dataset):
    def __init__(self, data_dir, transform=None, out_size=(472, 600), weights_path='trained_weights.pth'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.out_size = out_size
        self.imgs_data = self.get_data(self.data_dir)
        # self.mask_dir_data = self.get_data(self.mask_dir)

        # 加载训练好的权重
        # self.weights = torch.load(weights_path)

        # 其他初始化
        # self.conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1)
        # self.relu = nn.ReLU()

    def get_data(self, data_path):
        data = []
        for img_path in glob.glob(data_path + os.sep + '*'):
            data.append(img_path)
        return data

    def __getitem__(self, index):
        # 读取图像和掩膜
        img = cv2.imread(self.imgs_data[index], 0)
        # mask = cv2.imread(self.mask_dir_data[index], 0)
        # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 调整图像尺寸和填充（保持与训练时一致）
        if img.shape[0] > self.out_size[0]:
            resize_factor = self.out_size[0] / img.shape[0]
            img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        if img.shape[1] > self.out_size[1]:
            resize_factor = self.out_size[1] / img.shape[1]
            img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

        # 填充到固定尺寸
        pad_height = self.out_size[0] - img.shape[0]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_width = self.out_size[1] - img.shape[1]
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0)

        # 应用数据预处理（归一化等）
        img = self.transform(img)
        # mask = self.transform(mask)

        # ==============================
        # 生成小波特征（与训练时一致）
        # ==============================
        # dwt = DWTForward(J=1, wave='haar')
        # img_4d = img.unsqueeze(0)  # 添加 channel 维度
        # yl, yh = dwt(img_4d)
        #
        # # 分解三个方向
        # lh = yh[0][:, :, 0, :, :]  # 水平细节 (LH)
        # hl = yh[0][:, :, 1, :, :]  # 垂直细节 (HL)
        # hh = yh[0][:, :, 2, :, :]  # 对角线细节 (HH)
        #
        # # # 归一化
        # # lh = process_subband(lh)
        # # hl = process_subband(hl)
        # # hh = process_subband(hh)
        #
        # # 上采样到原图尺寸
        # yl = F.interpolate(yl, size=self.out_size, mode='bilinear', align_corners=False)
        # lh = F.interpolate(lh, size=self.out_size, mode='bilinear', align_corners=False)
        # hl = F.interpolate(hl, size=self.out_size, mode='bilinear', align_corners=False)
        # hh = F.interpolate(hh, size=self.out_size, mode='bilinear', align_corners=False)
        #
        # # 移除多余的维度
        # yl = yl[0, :, :, :]
        # lh = lh[0, :, :, :]
        # hl = hl[0, :, :, :]
        # hh = hh[0, :, :, :]
        #
        # # 拼接小波分量
        # wavelet = torch.cat([lh, hl, hh, yl], dim=0)
        # wavelet = process_subband(wavelet)
        # wavelet = self.relu(wavelet)
        #
        # # 通过卷积调整通道数
        # wavelet = self.conv(wavelet.unsqueeze(0)).squeeze(0)  # [1, H, W]

        # ==============================
        # 应用训练好的权重并拼接输入
        # ==============================
        # combined_input = torch.cat([
        #     img,
        #     mask
        # ], dim=0)  # 沿通道维度拼接

        return img

    def __len__(self):
        return len(self.imgs_data)
