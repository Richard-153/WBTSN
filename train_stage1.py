import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
import config_stage1 as cfg
# 1. 导入新的 UNetWavelet 模型和 WT 辅助模块
from model import UNetWavelet, WT
from datasets_stage1 import DAE_dataset
import numpy as np


# =====================================================================
# 2. 定义新的复合损失函数
# =====================================================================
class CompositeLoss(nn.Module):
    """
    一个复合损失函数，它结合了图像空间的MSE损失和小波空间的MSE损失。
    """

    def __init__(self, w_main=1.0, w_wavelet=0.25):
        """
        Args:
            w_main (float): 主图像MSE损失的权重。
            w_wavelet (float): 每个小波分量损失的权重。
        """
        super(CompositeLoss, self).__init__()
        self.w_main = w_main
        self.w_wavelet = w_wavelet
        # 用于计算小波损失的MSE函数
        self.mse_loss = nn.MSELoss()
        # 用于对目标图像进行小波变换的模块
        # 注意：这里的 in_ch 应该与模型输出的通道数匹配
        self.wt_transform = WT(in_ch=1).to(device)

    def forward(self, model_outputs, targets):
        """
        Args:
            model_outputs (tuple): 来自UNetWavelet模型的输出元组
                                   (output, yL, y_HL, y_LH, y_HH)。
            targets (torch.Tensor): 目标图像张量。
        """
        # 解包模型输出
        pred_main, pred_L, pred_HL, pred_LH, pred_HH = model_outputs

        # --- 计算主损失 (图像空间) ---
        loss_main = self.mse_loss(pred_main, targets)

        # --- 计算小波损失 (频域空间) ---
        # 首先，获取目标图像的小波分量
        with torch.no_grad():  # 此操作不应影响梯度计算
            target_L, target_HL, target_LH, target_HH = self.wt_transform(targets)

        # 计算每个小波分量的损失
        loss_L = self.mse_loss(pred_L, target_L)
        loss_HL = self.mse_loss(pred_HL, target_HL)
        loss_LH = self.mse_loss(pred_LH, target_LH)
        loss_HH = self.mse_loss(pred_HH, target_HH)

        # 将所有小波损失相加
        total_wavelet_loss = loss_L + loss_HL + loss_LH + loss_HH

        # --- 计算最终加权总损失 ---
        total_loss = self.w_main * loss_main + self.w_wavelet * total_wavelet_loss

        return total_loss


# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 计时开始
start_time = time.time()

# 创建模型目录
models_dir = cfg.models_dir
os.makedirs(models_dir, exist_ok=True)

# 数据集路径
train_dir = os.path.join(cfg.data_dir, cfg.train_dir)
val_dir = os.path.join(cfg.data_dir, cfg.val_dir)

# 转换函数
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# 创建数据集
train_dataset = DAE_dataset(data_dir=train_dir, transform=transform)
val_dataset = DAE_dataset(data_dir=val_dir, transform=transform)
print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

# 创建数据加载器
batch_size = cfg.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print(f'Train batches: {len(train_loader)} (bs={batch_size})')

# 3. 模型初始化 - 使用新的 UNetWavelet
model = UNetWavelet(n_classes=1, padding=True, batch_norm=True, wf=6).to(device)


# 参数统计
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# 加载或创建模型
resume = cfg.resume
ckpt_path = cfg.ckpt if os.path.isabs(cfg.ckpt) else os.path.join(models_dir, cfg.ckpt)
if resume and cfg.ckpt and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    train_epoch_loss = ckpt.get('train_epoch_loss', [])  # 使用 .get() 增加兼容性
    val_epoch_loss = ckpt.get('val_epoch_loss', [])
    epochs_till_now = ckpt.get('epoch', 0)
    print(f'Resumed from checkpoint, starting from epoch {epochs_till_now + 1}')
else:
    train_epoch_loss, val_epoch_loss = [], []
    epochs_till_now = 0
    print('Training from scratch')

# 4. 优化器和损失函数 - 使用新的 CompositeLoss
lr = 3 * cfg.lr
optimizer = Adam(model.parameters(), lr=lr)


loss_fn = CompositeLoss(w_main=1.0, w_wavelet=0.25).to(device)  # <--- 使用新的复合损失


# loss_fn = nn.MSELoss()
log_interval = cfg.log_interval
epochs = cfg.epochs
total_batches = epochs * len(train_loader)
scheduler = CosineAnnealingLR(optimizer, T_max=total_batches, eta_min=1e-6)

# 训练信息
print(f'\nModel params: {count_params(model):.1f}M')
print(f'Loss function: {type(loss_fn).__name__}')  # 将打印 'CompositeLoss'
print(f'Learning rate: {lr:.7f}')
print(f'Epochs: {epochs_till_now + 1}/{epochs_till_now + epochs}')

# 训练循环
for epoch in range(epochs_till_now, epochs_till_now + epochs):
    epoch_start = time.time()
    model.train()
    epoch_loss = []

    print(f'\nEpoch {epoch + 1}/{epochs_till_now + epochs} [TRAINING]')

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_start = time.time()

        # 数据移动到设备
        inputs, targets = inputs.to(device), targets.to(device)

        # 5. 前向传播 - 输出现在是一个元组
        model_outputs = model(inputs)

        # 6. 损失计算 - 直接将元组和目标传入新的损失函数
        loss = loss_fn(model_outputs, targets)
        epoch_loss.append(loss.item())

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 日志记录
        if (batch_idx + 1) % log_interval == 0:
            batch_time = time.time() - batch_start
            minutes, seconds = divmod(batch_time, 60)
            print(f'Batch {batch_idx + 1}/{len(train_loader)}: '
                  f'Loss={loss.item():.7f}, '
                  f'Time={int(minutes)}m {seconds:.1f}s, '
                  f'LR={optimizer.param_groups[0]["lr"]:.2e}')

    # 记录epoch平均损失
    avg_loss = np.mean(epoch_loss)
    train_epoch_loss.append(avg_loss)

    # 保存模型
    # 注意：保存的字典中也应包含损失历史，以便恢复
    checkpoint_path = os.path.join(models_dir, f'model_epoch{epoch + 1:02d}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_epoch_loss': train_epoch_loss,
        'val_epoch_loss': val_epoch_loss,  # 即使目前没有验证循环，也保存以保持一致性
        'epoch': epoch + 1
    }, checkpoint_path)
    print(f'Saved checkpoint: {checkpoint_path}')

# 训练完成
total_time = time.time() - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f'\nTraining complete! Total time: {int(hours)}h {int(minutes)}m {seconds:.1f}s')
