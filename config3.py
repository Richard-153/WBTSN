# ================================
# 文件 2/2: config_correction_gan.py (与 train10.py 配套)
# ================================

import os

# -------- 路径 --------
# data_dir = '/home/vision/users/cgh/denoising/unet12/unet12_xiaorong/DV'
data_dir = '/home/vision/users/cgh/denoising/unet12/unet+att+mask+wavelet/BPD'
#data_dir = '/home/vision/users/cgh/denoising/unet13/IRE3'
#data_dir = '/home/vision/users/cgh/denoising/unet12/unet12_zhuanyi/FLIR'
train_dir = 'train'
val_dir   = 'test'

# 一阶段基座模型权重（必须存在）
generator_ckpt_path = '/home/vision/users/cgh/denoising/unet13/models/model_epoch573.pth'

# 输出/中间结果
models_dir_gan = 'models_correction_gan'
res_dir_correction_gan = 'results_correction_gan'
test_res_dir = 'test_results'

# -------- DataLoader --------
batch_size  = 4
num_workers = 4
test_bs = 1  # 测试时可单张或小批量
print_interval = 20
epochs = 500
save_interval = 1
log_interval  = 50

# 判别器更新次数
n_critic = 1

# AMP 混合精度
use_amp = True

# R1 正则
r1_gamma = 0.0

# -------- 优化器 --------
g_lr = 2e-4
g_betas = (0.5, 0.999)
d_lr = 2e-4
d_betas = (0.5, 0.999)

# -------- CorrectionNet 结构 --------
g_base_ch = 48
g_num_rb  = 2

# -------- 损失权重 --------
lambda_l1_residual = 2.0
lambda_l1_final   = 8.0
lambda_adv        = 0.0035
#0.0035
# -------- 恢复训练 --------
resume = True
resume_g_path = os.path.join(models_dir_gan, 'correction_net_epoch_64.pth')
resume_d_path = os.path.join(models_dir_gan, 'discriminator_epoch_64.pth')

# -------- 测试 --------
# test_dir = '/home/vision/users/cgh/denoising/unet12/unet12_xiaorong/DV/test/noisy'
# test_gt_dir = '/home/vision/users/cgh/denoising/unet12/unet12_xiaorong/DV/test/imgs'
#test_dir = '/home/vision/users/cgh/denoising/unet12/unet+att+mask+wavelet/BPD/test/noisy'
#test_gt_dir = '/home/vision/users/cgh/denoising/unet12/unet+att+mask+wavelet/BPD/test/imgs'
#test_dir = '/home/vision/users/cgh/denoising/unet13/IRE3/test/noisy'
#test_gt_dir = '/home/vision/users/cgh/denoising/unet13/IRE3/test/imgs'
#test_dir = '/home/vision/users/cgh/denoising/unet12/unet12_zhuanyi/FLIR/test/noisy'
#test_gt_dir = '/home/vision/users/cgh/denoising/unet12/unet12_zhuanyi/FLIR/test/imgs'
test_dir = '/home/vision/users/cgh/denoising/unet13/image2'
test_gt_dir = '/home/vision/users/cgh/denoising/unet13/image2'
test_correction_model_path = os.path.join(models_dir_gan, 'correction_net_epoch_64.pth') #316 30.85 315 30.9 430 30.9

ssim_data_range = 1.0
