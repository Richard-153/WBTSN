# Full two-stage inference script.

import os
import shutil
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import time

from model import UNetWavelet
from train_stage2 import CorrectionNet
import config_stage2 as cfg

class CustomTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(supported_formats)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        filename = self.image_files[index]
        img_path = os.path.join(self.data_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image)
        return image, filename


def test_correction_denoising():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.test_res_dir, exist_ok=True)

    transform = transforms.ToTensor()
    test_dataset = CustomTestDataset(cfg.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.test_bs, shuffle=False, num_workers=4)

    # 加载 base model
    base_model = UNetWavelet(n_classes=1, padding=True, batch_norm=True, wf=6).to(device)
    ckpt_base = torch.load(cfg.generator_ckpt_path, map_location=device)
    base_model.load_state_dict(ckpt_base.get('model_state_dict', ckpt_base))
    base_model.eval()

    # 加载 CorrectionNet
    correction_model = CorrectionNet(base_ch=cfg.g_base_ch, num_rb=cfg.g_num_rb).to(device)
    ckpt_corr = torch.load(cfg.test_correction_model_path, map_location=device)
    correction_model.load_state_dict(ckpt_corr)
    correction_model.eval()

    psnr_list, ssim_list = [], []
    start_time = time.time()

    with torch.no_grad():
        for noisy_imgs, filenames in tqdm(test_loader, desc='Testing'):
            noisy_imgs = noisy_imgs.to(device)
            if noisy_imgs.dim() == 3:
                noisy_imgs = noisy_imgs.unsqueeze(1)

            smooth_outputs_tuple = base_model(noisy_imgs)
            smooth_outputs = smooth_outputs_tuple[0] if isinstance(smooth_outputs_tuple, (tuple, list)) else smooth_outputs_tuple

            predicted_residual = correction_model(smooth_outputs, noisy_imgs)
            final_outputs = smooth_outputs + predicted_residual

            for i in range(final_outputs.shape[0]):
                out_tensor = final_outputs[i].cpu().squeeze(0).numpy()
                out_tensor = np.clip(out_tensor, 0, 1)

                gt_path = os.path.join(cfg.test_gt_dir, filenames[i])
                if os.path.exists(gt_path):
                    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
                    psnr_list.append(compare_psnr(gt_img, out_tensor, data_range=cfg.ssim_data_range))
                    ssim_list.append(compare_ssim(gt_img, out_tensor, data_range=cfg.ssim_data_range))

                save_path = os.path.join(cfg.test_res_dir, filenames[i])
                cv2.imwrite(save_path, (out_tensor*255).astype(np.uint8))

    total_time = time.time() - start_time
    print(f"Testing completed in {total_time:.2f}s")
    if psnr_list:
        print(f"Average PSNR: {np.mean(psnr_list):.2f} dB")
        print(f"Average SSIM: {np.mean(ssim_list):.4f}")


if __name__=='__main__':
    test_correction_denoising()
