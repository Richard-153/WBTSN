import os

# Stage II residual-correction configuration for WBTSN.
# Stage II uses a frozen Stage I model and trains a residual correction network.

# Dataset root. Expected structure:
# DATA_ROOT/train/imgs, DATA_ROOT/train/noisy
# DATA_ROOT/val/imgs,   DATA_ROOT/val/noisy
# DATA_ROOT/test/noisy, DATA_ROOT/test/imgs optional for metric reporting
data_dir = "/path/to/DATA_ROOT"
train_dir = "train"
val_dir = "val"

# Path to a trained Stage I checkpoint.
generator_ckpt_path = os.path.join("models", "model_epoch200.pth")

# Output directories.
models_dir_gan = "models_stage2"
res_dir_correction_gan = "results_stage2"
test_res_dir = "results_stage2"

# DataLoader and training schedule.
batch_size = 4
num_workers = 4
test_bs = 1
print_interval = 20
epochs = 200
save_interval = 1
log_interval = 50

# Discriminator update frequency.
n_critic = 1

# Mixed precision flag retained for compatibility with the original code.
use_amp = True

# R1 regularization coefficient retained for compatibility.
r1_gamma = 0.0

# Optimizers.
g_lr = 2e-4
g_betas = (0.5, 0.999)
d_lr = 2e-4
d_betas = (0.5, 0.999)

# CorrectionNet architecture.
g_base_ch = 48
g_num_rb = 2

# Loss weights.
lambda_l1_residual = 2.0
lambda_l1_final = 8.0
lambda_adv = 0.0035

# Resume Stage II training.
resume = False
resume_g_path = os.path.join(models_dir_gan, "correction_net_epoch_1.pth")
resume_d_path = os.path.join(models_dir_gan, "discriminator_epoch_1.pth")

# Inference settings.
test_dir = os.path.join(data_dir, "test", "noisy")
test_gt_dir = os.path.join(data_dir, "test", "imgs")
test_correction_model_path = os.path.join(models_dir_gan, "correction_net_epoch_200.pth")
ssim_data_range = 1.0

