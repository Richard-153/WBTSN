
import torch
from torch import nn
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F

# =====================================================================
# 模块1: 小波残差块 (WaveletResidualBlock)
# 这是您原始残差块的升级版，在内部集成了小波变换以增强多尺度特征提取能力。
# 我保留了类名 "ResidualBlock" 以便在U-Net中无缝替换，但其内部实现已是新版。
# =====================================================================
# =====================================================================
# =====================================================================
# 模块1: 标准残差块 (替换 UNetConvBlock)
# =====================================================================
class ResidualBlock(nn.Module):
    # 👇 确保 __init__ 方法有 self, in_size, out_size, padding, batch_norm 这五个参数
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(ResidualBlock, self).__init__()
        # 如果输入输出通道数不同，使用1x1卷积进行维度匹配
        self.skip_connection = nn.Conv2d(in_size, out_size, kernel_size=1, bias=False) if in_size != out_size else None

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding), bias=False)
        self.bn1 = nn.BatchNorm2d(out_size) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding), bias=False)
        self.bn2 = nn.BatchNorm2d(out_size) if batch_norm else nn.Identity()

    def forward(self, x):
        identity = self.skip_connection(x) if self.skip_connection is not None else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# =====================================================================
# 模块2: 混合注意力模块 (Hybrid Attention Block, HAB) - (保持不变)
# =====================================================================
class HybridAttentionBlock(nn.Module):
    def __init__(self, gating_channels, high_freq_channels, bridge_channels, reduction=16):
        super(HybridAttentionBlock, self).__init__()

        # 1. 通道注意力分支
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(bridge_channels, bridge_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bridge_channels // reduction, bridge_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 2. 空间注意力分支
        inter_channels = bridge_channels // 2
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv2d(high_freq_channels, inter_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 3. 用于生成最终注意力残差的卷积层
        self.final_conv = nn.Conv2d(bridge_channels, bridge_channels, kernel_size=1)

    def forward(self, g, bridge, x_high_freq):
        channel_gate = self.channel_attention(bridge)
        bridge_ca = bridge * channel_gate

        g_processed = self.W_g(g)
        x_processed = self.W_x(x_high_freq)
        fused_info = self.relu(g_processed + x_processed)
        spatial_gate = self.psi(fused_info)

        bridge_sa = bridge_ca * spatial_gate

        attention_residual = self.final_conv(bridge_sa)
        output = bridge + attention_residual

        return output


# =====================================================================
# 模块3: U-Net上采样模块 (集成HAB) - (已修正)
# =====================================================================
# =====================================================================
# 模块3: U-Net上采样模块 (集成HAB) - 【针对新ResidualBlock的修改版】
# =====================================================================
class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, use_attention=False, high_freq_channels=None):
        super(UNetUpBlock, self).__init__()
        self.use_attention = use_attention
        self.up = WaveletUpsample(in_size, out_size)

        if self.use_attention:
            if high_freq_channels is None:
                raise ValueError("high_freq_channels must be provided if use_attention is True")
            self.attention = HybridAttentionBlock(
                gating_channels=out_size,
                high_freq_channels=high_freq_channels,
                bridge_channels=out_size
            )

        # ### --- 核心修正点在这里 --- ###
        # 我们将调用 ResidualBlock，它的输入通道数是拼接后的 2 * out_size
        # 输出通道数是该上采样阶段的目标 out_size
        self.conv_block = ResidualBlock(2 * out_size, out_size, padding, batch_norm)

    # 我们之前添加的插值方案，比 center_crop 更好，保留它
    def interpolate_to_match(self, tensor_to_resize, target_tensor):
        target_size = target_tensor.shape[2:]
        if tensor_to_resize.shape[2:] != target_size:
            return F.interpolate(tensor_to_resize, size=target_size, mode='bilinear', align_corners=False)
        return tensor_to_resize

    def forward(self, x, bridge, high_freq_features=None):
        up = self.up(x)

        # 将上采样后的特征图 up 和高频特征图 high_freq_features 的尺寸都对齐到 bridge
        up = self.interpolate_to_match(up, bridge)

        bridge_guided = bridge
        if self.use_attention and high_freq_features is not None:
            high_freq_features_aligned = self.interpolate_to_match(high_freq_features, bridge)
            bridge_guided = self.attention(g=up, bridge=bridge, x_high_freq=high_freq_features_aligned)

        # 拼接特征图，总通道数为 2 * out_size
        out = torch.cat([up, bridge_guided], 1)

        # 使用 conv_block 将通道数从 2 * out_size 降为 out_size 并进行特征提取
        out = self.conv_block(out)

        return out

# =====================================================================
# 模块4: 核心U-Net模型 (最终版) - 【修正版】
# 核心改动：修正 UNetWavelet 中对 ResidualBlock 的调用方式
# =====================================================================
class UNetWavelet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,
            wf=6,
            padding=False,
            batch_norm=False
    ):
        super(UNetWavelet, self).__init__()
        self.padding = padding
        self.dwt_multiscale = DWTForward(J=2, mode='zero', wave='haar')
        high_freq_ch = in_channels * 3

        # ### --- 核心修正点在这里 --- ###
        # 编码器路径全部使用 ResidualBlock
        # 调用方式从 (num_features=...) 改为 (in_size, out_size, ...)

        # down1: 输入是 in_channels, 输出是 2**wf
        self.down1 = ResidualBlock(in_channels, 2 ** wf, padding, batch_norm)
        self.wavelet_down1 = WaveletDownsample(2 ** wf, 2 ** (wf + 1))

        # down2: 输入和输出都是 2**(wf+1)
        self.down2 = ResidualBlock(2 ** (wf + 1), 2 ** (wf + 1), padding, batch_norm)
        self.wavelet_down2 = WaveletDownsample(2 ** (wf + 1), 2 ** (wf + 2))

        # down3: 输入和输出都是 2**(wf+2)
        self.down3 = ResidualBlock(2 ** (wf + 2), 2 ** (wf + 2), padding, batch_norm)
        self.wavelet_down3 = WaveletDownsample(2 ** (wf + 2), 2 ** (wf + 3))

        # down4: 输入和输出都是 2**(wf+3)
        self.down4 = ResidualBlock(2 ** (wf + 3), 2 ** (wf + 3), padding, batch_norm)

        # 解码器路径 (UNetUpBlock 内部的调用也需要是正确的，我们之前的修改已经保证了这一点)
        self.up3 = UNetUpBlock(2 ** (wf + 3), 2 ** (wf + 2), padding, batch_norm, use_attention=True,
                               high_freq_channels=high_freq_ch)
        self.up2 = UNetUpBlock(2 ** (wf + 2), 2 ** (wf + 1), padding, batch_norm, use_attention=True,
                               high_freq_channels=high_freq_ch)
        self.up1 = UNetUpBlock(2 ** (wf + 1), 2 ** wf, padding, batch_norm, use_attention=False)

        self.last = nn.Conv2d(2 ** wf, n_classes, kernel_size=1)
        self.WT = WT(in_ch=n_classes)

    def forward(self, x):
        identity = x

        _, yH = self.dwt_multiscale(x)
        wavelet_h2 = torch.cat([yH[0][:, :, 0, :, :], yH[0][:, :, 1, :, :], yH[0][:, :, 2, :, :]], dim=1)
        wavelet_h4 = torch.cat([yH[1][:, :, 0, :, :], yH[1][:, :, 1, :, :], yH[1][:, :, 2, :, :]], dim=1)

        x1 = self.down1(x)
        x1_down = self.wavelet_down1(x1)
        x2 = self.down2(x1_down)
        x2_down = self.wavelet_down2(x2)
        x3 = self.down3(x2_down)
        x3_down = self.wavelet_down3(x3)
        x4 = self.down4(x3_down)

        # 确保使用我们修复过尺寸问题的 UNetUpBlock
        u3 = self.up3(x4, x3, high_freq_features=wavelet_h4)
        u2 = self.up2(u3, x2, high_freq_features=wavelet_h2)
        u1 = self.up1(u2, x1)

        output_residual = self.last(u1)
        final_output = identity + output_residual
        yL, y_HL, y_LH, y_HH = self.WT(final_output)

        return final_output, yL, y_HL, y_LH, y_HH


# =====================================================================
# 辅助模块 (保持不变)
# =====================================================================
class WT(nn.Module):
    def __init__(self, in_ch):
        super(WT, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]
        return yL, y_HL, y_LH, y_HH


class WaveletDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WaveletDownsample, self).__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        yL, yH = self.dwt(x)
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]
        x_wavelet = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x_out = self.conv(x_wavelet)
        return x_out


class WaveletUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WaveletUpsample, self).__init__()
        self.idwt = DWTInverse(mode='zero', wave='haar')
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1)

    def forward(self, x):
        x_conv = self.conv(x)
        B, C, H, W = x_conv.shape
        ch_comp = C // 4
        yL = x_conv[:, :ch_comp, :, :]
        y_HL = x_conv[:, ch_comp:2 * ch_comp, :, :]
        y_LH = x_conv[:, 2 * ch_comp:3 * ch_comp, :, :]
        y_HH = x_conv[:, 3 * ch_comp:, :, :]
        yH = [torch.stack([y_HL, y_LH, y_HH], dim=2)]
        x_out = self.idwt((yL, yH))
        return x_out