import os

# Stage I configuration for WBTSN.
# Edit these paths before training or inference.

# Output directories.
models_dir = "models"
losses_dir = "losses"
res_dir = "results_stage1"

# Dataset root. Expected structure:
# DATA_ROOT/train/imgs, DATA_ROOT/train/noisy
# DATA_ROOT/val/imgs,   DATA_ROOT/val/noisy
# DATA_ROOT/test/noisy
data_dir = "/path/to/DATA_ROOT"
train_dir = "train"
val_dir = "val"
imgs_dir = "imgs"
noisy_dir = "noisy"
debug_dir = "debug"

# Resume training from an existing Stage I checkpoint.
# ckpt can be either an absolute path or a file under models_dir.
resume = False
ckpt = ""

# Training hyperparameters.
lr = 1e-5
epochs = 200
batch_size = 2
log_interval = 25

# Inference settings.
test_img_size = (512, 640)
test_dir = os.path.join(data_dir, "test", "noisy")
test_bs = 2

