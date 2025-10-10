import numpy as np
import os, glob, cv2
from torch.utils.data import Dataset
from torchvision import transforms  # ### --- 修改点 --- ###: 导入 torchvision.transforms
import config as cfg

class DAE_dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 加载图像和噪声图像路径（保持不变）
        self.imgs_data = self.get_data(os.path.join(self.data_dir, 'imgs'))
        self.noisy_imgs_data = self.get_data(os.path.join(self.data_dir, 'noisy'))

        # 确保文件列表是排序的，以保证 img 和 noisy_img 一一对应
        self.imgs_data.sort()
        self.noisy_imgs_data.sort()

    def get_data(self, data_path):
        # 确保目录存在
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"数据目录不存在: {data_path}")
        return [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.bmp'))]

    def __getitem__(self, index):
        # 读取图像、噪声图像（均为灰度图）
        img = cv2.imread(self.imgs_data[index], 0)  # 原始图像（干净图像）
        noisy_img = cv2.imread(self.noisy_imgs_data[index], 0)  # 输入的噪声图像

        if img is None:
            raise ValueError(f"无法读取干净图像: {self.imgs_data[index]}")
        if noisy_img is None:
            raise ValueError(f"无法读取噪声图像: {self.noisy_imgs_data[index]}")

        # ### --- 修改点 --- ###
        # 为了对干净图像和噪声图像应用相同的随机变换，我们使用“堆叠技巧”
        # 1. 将两个numpy数组沿着新的轴堆叠起来。(H, W) -> (H, W, 2)
        stacked_images = np.stack([img, noisy_img], axis=-1)

        # 2. 如果定义了transform，就对这个堆叠后的“双通道”图像进行变换
        if self.transform:
            # transform中的ToTensor()会将 (H, W, 2) 的Numpy数组转为 (2, H, W) 的Tensor
            # 后续的随机变换（如翻转）会同时作用于这两个通道
            transformed_stack = self.transform(stacked_images)

            # 3. 将变换后的Tensor拆分回两个独立的图像
            clean_transformed = transformed_stack[0, :, :].unsqueeze(0)  # (1, H, W)
            noisy_transformed = transformed_stack[1, :, :].unsqueeze(0)  # (1, H, W)

            return noisy_transformed, clean_transformed

        # 如果没有transform，则执行默认的转换
        else:
            # 删除了原有的分别变换，以防出错
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)
            noisy_img = to_tensor(noisy_img)
            return noisy_img, img

    def __len__(self):
        # 返回两个列表中较小的一个的长度，更加健壮
        return min(len(self.imgs_data), len(self.noisy_imgs_data))


class custom_test_dataset(Dataset):
    def __init__(self, data_dir, transform=None, out_size=cfg.test_img_size):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.out_size = out_size
        self.imgs_data = self.get_data(self.data_dir)

    def get_data(self, data_path):
        return [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.bmp'))]

    def __getitem__(self, index):
        # 获取图像的完整路径
        img_path = self.imgs_data[index]

        # 读取图像为灰度图，此时 img 是一个二维的 NumPy 数组 (H, W)
        img = cv2.imread(img_path, 0)

        # 从完整路径中提取文件名
        filename = os.path.basename(img_path)

        # 调整图像尺寸和填充 (这部分逻辑保持不变)
        if img.shape[0] > self.out_size[0]:
            resize_factor = self.out_size[0] / img.shape[0]
            img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        if img.shape[1] > self.out_size[1]:
            resize_factor = self.out_size[1] / img.shape[1]
            img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

        pad_height = self.out_size[0] - img.shape[0]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_width = self.out_size[1] - img.shape[1]
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0)

        # ### --- 这是最关键的一步 --- ###
        # 必须确保 transform 被应用。
        # transforms.ToTensor() 会将 (H, W) 的 NumPy 数组转换为 (1, H, W) 的张量。
        if self.transform:
            img = self.transform(img)
        else:
            # 作为备用方案，如果忘记传入 transform，也手动进行转换
            img = transforms.ToTensor()(img)

        # 返回处理后的图像张量和它的原始文件名
        return img, filename

    def __len__(self):
        return len(self.imgs_data)