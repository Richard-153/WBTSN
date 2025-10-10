# ================================
# 文件: train10.py (二阶段训练 + 残差约束 + GAN, 去掉感知损失)
# ================================

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

import config3 as cfg
from datasets1 import DAE_dataset
from Wavelet0808 import UNetWavelet

# ---------------- Utils ----------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

# ---------------- Blocks ----------------

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(ch, affine=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(ch, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        idt = x
        x = self.act(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        return self.act(x + idt)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        feat = self.conv(x)
        down = self.down(feat)
        return feat, down

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ---------------- CorrectionNet ----------------

class CorrectionNet(nn.Module):
    def __init__(self, in_channels=2, base_ch=48, num_rb=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.d1 = DownBlock(base_ch, base_ch*2)
        self.d2 = DownBlock(base_ch*2, base_ch*4)
        self.d3 = DownBlock(base_ch*4, base_ch*4)
        self.bottle = nn.Sequential(*[ResidualBlock(base_ch*4) for _ in range(num_rb+1)])
        self.u3 = UpBlock(in_ch=base_ch*4, skip_ch=base_ch*4, out_ch=base_ch*4)
        self.u2 = UpBlock(in_ch=base_ch*4, skip_ch=base_ch*4, out_ch=base_ch*2)
        self.u1 = UpBlock(in_ch=base_ch*2, skip_ch=base_ch*2, out_ch=base_ch)
        self.tail = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, smooth_img, noisy_img):
        x = torch.cat([smooth_img, noisy_img], dim=1)
        x = self.head(x)
        s1, x = self.d1(x)
        s2, x = self.d2(x)
        s3, x = self.d3(x)
        x = self.bottle(x)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        res = self.tail(x)
        res = F.interpolate(res, size=noisy_img.shape[-2:], mode='bilinear', align_corners=False)
        return res

# ---------------- Discriminator ----------------

class PatchDiscriminatorSN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(512,1,4,1,1))
        )
    def forward(self,x):
        return self.model(x)

# ---------------- GAN Loss ----------------

def ragan_d_loss(d_real, d_fake):
    bce = nn.BCEWithLogitsLoss()
    return 0.5*(bce(d_real - d_fake.mean(), torch.ones_like(d_real)) + bce(d_fake - d_real.mean(), torch.zeros_like(d_fake)))

def ragan_g_loss(d_real, d_fake):
    bce = nn.BCEWithLogitsLoss()
    return bce(d_fake - d_real.mean(), torch.ones_like(d_fake))

# ---------------- Main ----------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ensure_dir(cfg.models_dir_gan)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = DAE_dataset(data_dir=os.path.join(cfg.data_dir,cfg.train_dir), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    # ---------------- Base UNet ----------------
    base_model = UNetWavelet(n_classes=1, padding=True, batch_norm=True).to(device)
    ckpt = torch.load(cfg.generator_ckpt_path, map_location=device)
    base_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    base_model.eval(); [p.requires_grad_(False) for p in base_model.parameters()]

    # ---------------- Networks ----------------
    G = CorrectionNet(in_channels=2, base_ch=cfg.g_base_ch, num_rb=cfg.g_num_rb).to(device)
    D = PatchDiscriminatorSN(in_channels=1).to(device)

    # ---------------- Optimizers ----------------
    g_optimizer = Adam(G.parameters(), lr=cfg.g_lr, betas=cfg.g_betas)
    d_optimizer = Adam(D.parameters(), lr=cfg.d_lr, betas=cfg.d_betas)

    # ---------------- Resume ----------------
    start_epoch = 0
    if getattr(cfg,'resume',False):
        if os.path.exists(cfg.resume_g_path):
            print(f"Resuming G from {cfg.resume_g_path}")
            G.load_state_dict(torch.load(cfg.resume_g_path, map_location=device))
        if os.path.exists(cfg.resume_d_path):
            print(f"Resuming D from {cfg.resume_d_path}")
            D.load_state_dict(torch.load(cfg.resume_d_path, map_location=device))

    l1_loss = nn.L1Loss()

    G.train(); D.train()

    for epoch in range(start_epoch, cfg.epochs):
        for step,(inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                smooth = base_model(inputs)
                smooth = smooth[0] if isinstance(smooth,(tuple,list)) else smooth

            # ---------------- G Step ----------------
            g_optimizer.zero_grad()
            pred_res = G(smooth, inputs)
            final_out = smooth + pred_res
            true_res = targets - smooth

            loss_l1_res = cfg.lambda_l1_residual * l1_loss(pred_res, true_res)
            loss_l1_fin = cfg.lambda_l1_final * l1_loss(final_out, targets)

            d_real = D(targets).detach()
            d_fake = D(final_out)
            loss_adv = cfg.lambda_adv * ragan_g_loss(d_real, d_fake)

            g_loss = loss_l1_res + loss_l1_fin + loss_adv
            g_loss.backward()
            g_optimizer.step()

            # ---------------- D Step ----------------
            for _ in range(cfg.n_critic):
                d_optimizer.zero_grad()
                d_real = D(targets)
                d_fake = D(final_out.detach())
                loss_D = ragan_d_loss(d_real, d_fake)
                loss_D.backward()
                d_optimizer.step()

            if step % cfg.print_interval == 0:
                print(f"Epoch {epoch+1}/{cfg.epochs} | Step {step}/{len(train_loader)} | L1_res:{loss_l1_res.item():.4f} | L1_fin:{loss_l1_fin.item():.4f} | Adv:{loss_adv.item():.4f} | D:{loss_D.item():.4f}")

        # ---------------- Save ----------------
        if (epoch+1)%cfg.save_interval==0:
            torch.save(G.state_dict(), os.path.join(cfg.models_dir_gan, f'correction_net_epoch_{epoch+1}.pth'))
            torch.save(D.state_dict(), os.path.join(cfg.models_dir_gan, f'discriminator_epoch_{epoch+1}.pth'))

if __name__=='__main__':
    main()