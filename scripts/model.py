# model.py
import torch
import torch.nn as nn
import torch.fft as fft
from typing import Tuple
import torch.nn.functional as F


# ----------------------------
# Utilities
# ----------------------------

def make_coord_grid(B: int, H: int, W: int, device=None, dtype=torch.float32):
    """
    Normalized (Q,E) coords in [-1,1], shape (B, 2, H, W)
    channel 0 -> Q axis (vertical); channel 1 -> E axis (horizontal)
    """
    q = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    e = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    qq, ee = torch.meshgrid(q, e, indexing='ij')
    grid = torch.stack([qq, ee], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,2,H,W)
    return grid


class ResidualBlock(nn.Module):
    def __init__(self, nf=64, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size, padding=pad)
        )

    def forward(self, x):
        return x + self.body(x)


# ----------------------------
# 1) SRCNN (residual on HR)
# ----------------------------
class SRCNN(nn.Module):
    """
    Classic SRCNN-style model that predicts HR residual given LR input.
    It first upsamples LR -> HR, then applies SRCNN and outputs residual.
    """
    def __init__(self, scale_factor: Tuple[float, float]=(5, 10)):
        super().__init__()
        self.pre = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        # operate at HR
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 5, padding=2)
        )

    def forward(self, x):
        # x: (B,1,Hl,Wl) -> upsample to HR
        up = self.pre(x)            # (B,1,Hh,Wh)
        res = self.net(up).float()  # residual on HR
        return res


# ----------------------------
# 2) "PowderUNet" implemented as EDSR-like SR model
# ----------------------------
class PowderUNet(nn.Module):
    """
    EDSR-style residual super-resolution network (kept class name for compatibility).
    - LR input -> bicubic upsample to HR inside the network
    - concat normalized (Q,E) coordinate channels at HR
    - a shallow head + N residual blocks + tail conv -> 1-channel residual
    """
    def __init__(self, scale_factor=(5, 10), nf=64, n_blocks=16):
        super().__init__()
        self.scale_factor = scale_factor

        # head after upsampling (+coords)
        # We'll have 1 (intensity) + 2 (coords) = 3 input channels at HR
        self.head = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # body
        self.body = nn.Sequential(*[ResidualBlock(nf=nf, kernel_size=3) for _ in range(n_blocks)])

        # tail
        self.tail = nn.Conv2d(nf, 1, kernel_size=3, padding=1)

        # internal upsampler
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)

    def forward(self, x):
        """
        x: (B,1,Hl,Wl) -> upsample -> concat coords -> residual blocks -> residual HR
        """
        B, C, Hl, Wl = x.shape
        up = self.upsample(x)  # (B,1,Hh,Wh)
        Hh, Wh = up.shape[-2], up.shape[-1]

        # coords
        coords = make_coord_grid(B, Hh, Wh, device=up.device, dtype=up.dtype)  # (B,2,Hh,Wh)
        inp = torch.cat([up, coords], dim=1)  # (B,3,Hh,Wh)

        feat = self.head(inp)
        feat = self.body(feat)
        res  = self.tail(feat).float()
        return res  # residual on HR


# ----------------------------
# 3) FNO2d: Fourier Neural Operator on HR grid (residual)
# ----------------------------
class SpectralConv2d(nn.Module):
    """
    2D spectral convolution layer (FNO). Expects input (B,C,H,W) on HR grid.
    Uses parameterized complex weights on low-frequency modes and keeps the rest zero.
    """
    def __init__(self, in_c, out_c, m1, m2):
        super().__init__()
        # store real+imag separately in last dim=2, then view_as_complex
        self.weights = nn.Parameter(torch.randn(in_c, out_c, m1, m2, 2) * (1 / (in_c * out_c)))
        self.m1, self.m2 = m1, m2

    def forward(self, x):
        # x: (B,in_c,H,W), real
        B, C, H, W = x.shape
        x_ft = fft.rfftn(x, dim=(2, 3), norm="ortho")  # complex: (B,in_c,H,W//2+1)
        Hf, Wf = x_ft.shape[2], x_ft.shape[3]
        m1, m2 = min(self.m1, Hf), min(self.m2, Wf)

        out_ft = torch.zeros((B, self.weights.shape[1], H, Wf), device=x.device, dtype=torch.cfloat)
        w = torch.view_as_complex(self.weights)  # (in_c,out_c,self.m1,self.m2)
        x_slice = x_ft[:, :, :m1, :m2]          # (B,in_c,m1,m2)
        w_slice = w[:, :, :m1, :m2]             # (in_c,out_c,m1,m2)
        # contract in_c
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_slice, w_slice)
        out = fft.irfftn(out_ft, s=(H, W), dim=(2, 3), norm="ortho").real
        return out


class FNO2d(nn.Module):
    """
    FNO-based residual super-resolution head:
    - LR input -> bicubic upsample to HR
    - concat (Q,E) coords -> 1x1 conv to width channels
    - several spectral conv blocks with skip 1x1
    - 1x1 -> 1 residual channel
    """
    def __init__(self, modes1=20, modes2=10, width=64, output_size: Tuple[int,int]=(100,200), n_layers=4):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.modes1, self.modes2 = modes1, modes2

        # embed 3 channels (intensity+coords) -> width
        self.embed = nn.Conv2d(3, width, kernel_size=1)

        # spectral conv blocks
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)])
        self.ws    = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])

        # head/tail MLP on pixels (1x1 conv acts like per-pixel linear)
        self.head = nn.Conv2d(width, width, 1)
        self.tail = nn.Conv2d(width, 1, 1)

        # upsampler (LR->HR)
        self.upsample = nn.Upsample(size=output_size, mode='bicubic', align_corners=False)

    def forward(self, x):
        """
        x: (B,1,Hl,Wl) -> HR -> add coords -> spectral blocks -> residual (B,1,Hh,Wh)
        """
        B, C, Hl, Wl = x.shape
        up = self.upsample(x)  # (B,1,Hh,Wh)
        Hh, Wh = up.shape[-2], up.shape[-1]

        coords = make_coord_grid(B, Hh, Wh, device=up.device, dtype=up.dtype)  # (B,2,Hh,Wh)
        h = torch.cat([up, coords], dim=1)  # (B,3,Hh,Wh)
        h = self.embed(h)

        for sc, w1 in zip(self.convs, self.ws):
            h = torch.nn.functional.gelu(sc(h) + w1(h))

        h = torch.nn.functional.gelu(self.head(h))
        res = self.tail(h).float()
        return res


# -------- Wide Activation Distillation Block (WFDN) --------
class WFDNBlock(nn.Module):
    def __init__(self, in_channels=64, distill_rate=0.25, expansion=4):
        super().__init__()
        distilled_channels = int(in_channels * distill_rate)
        remaining_channels = in_channels - distilled_channels

        # 1x1 Conv 扩展通道
        self.conv1 = nn.Conv2d(in_channels, in_channels * expansion, 1)
        self.relu = nn.ReLU(inplace=True)
        # 3x3 Conv 压缩回去
        self.conv2 = nn.Conv2d(in_channels * expansion, in_channels, 3, padding=1)

        # 蒸馏与残差分支
        self.distilled = nn.Conv2d(in_channels, distilled_channels, 1)
        self.remaining = nn.Conv2d(in_channels, remaining_channels, 3, padding=1)

        # 融合
        self.fuse = nn.Conv2d(distilled_channels + remaining_channels, in_channels, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        d = self.distilled(out)
        r = self.remaining(out)
        out = torch.cat([d, r], dim=1)
        out = self.fuse(out)

        return out + x  # 残差连接


# -------- FNO2d --------
class FNO2dBlock(nn.Module):
    def __init__(self, in_c=64, modes1=20, modes2=10):
        super().__init__()
        self.spectral_conv = SpectralConv2d(in_c, in_c, modes1, modes2)
        self.w = nn.Conv2d(in_c, in_c, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.spectral_conv(x) + self.w(x))


# -------- Hybrid Model: WFDN + FNO --------
class Hybrid_WFDN_FNO(nn.Module):
    def __init__(self, in_channels=1, base_channels=64,
                 num_wfdn=4, num_fno=2, output_size=(100, 200)):
        super().__init__()
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # 局部分支：多层 WFDN Block
        self.local_branch = nn.Sequential(*[WFDNBlock(base_channels) for _ in range(num_wfdn)])

        # 全局分支：多层 FNO Block
        self.global_branch = nn.Sequential(*[FNO2dBlock(base_channels) for _ in range(num_fno)])

        # 融合
        self.fuse = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.tail = nn.Conv2d(base_channels, 1, 3, padding=1)

        self.output_size = output_size

    def forward(self, x):
        # x: (B,1,H,W)
        feat = self.head(x)

        local_feat = self.local_branch(feat)
        global_feat = self.global_branch(feat)

        fused = torch.cat([local_feat, global_feat], dim=1)
        fused = self.fuse(fused)
        fused = self.tail(fused)

        # 上采样到目标尺寸
        out = F.interpolate(fused, size=self.output_size,
                            mode="bilinear", align_corners=False)
        return out.float()  # 残差


# ====================== Ghost Module ======================
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, kernel_size=1, dw_kernel_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = int(out_channels / ratio)
        new_channels = out_channels - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1,
                      dw_kernel_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class GhostResidual(nn.Module):
    # A residual block with ghost depthwise convolutions
    def __init__(self, nf, ratio=2, kernel_size=1, dw_kernel_size=3, stride=1):
        super(GhostResidual, self).__init__()
        init_channels = int(nf / ratio)
        new_channels = nf - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(nf, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1,
                      dw_kernel_size // 2, groups=init_channels, bias=False),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return x + out

# ====================== Ghost WFDN Block ======================
class GhostWFDNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2):
        super(GhostWFDNBlock, self).__init__()
        mid_channels = in_channels * expansion
        self.conv1 = GhostModule(in_channels, mid_channels, ratio=2, kernel_size=1)
        self.conv2 = GhostModule(mid_channels, out_channels, ratio=2, kernel_size=3)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + self.shortcut(x)

class Hybrid_GhostWFDN_FNO(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_wfdn=4, num_fno=2, output_size=(100, 200)):
        super(Hybrid_GhostWFDN_FNO, self).__init__()
        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.wfdn_blocks = nn.Sequential(*[GhostWFDNBlock(base_channels, base_channels) for _ in range(num_wfdn)])
        self.fno_convs = nn.ModuleList(
            [SpectralConv2d(base_channels, base_channels, m1=20, m2=10) for _ in range(num_fno)])
        self.fno_ws = nn.ModuleList([nn.Conv2d(base_channels, base_channels, 1) for _ in range(num_fno)])
        self.final = nn.Conv2d(base_channels, 1, 1)
        self.output_size = output_size
    def forward(self, x):
        x = self.stem(x)
        x = self.wfdn_blocks(x)
        for conv, w in zip(self.fno_convs, self.fno_ws):
            x = F.gelu(conv(x) + w(x))
        x = F.interpolate(x, size=self.output_size, mode="bicubic", align_corners=False)
        out = self.final(x).float()
        return out

# Ghosted residual unet
class GhostUNet(nn.Module):
    def __init__(self, scale_factor=(5, 10), nf=64, n_blocks=4, nfno=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.head = nn.Sequential(
            nn.Conv2d(1, nf, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(*[GhostResidual(nf=nf, kernel_size=5) for _ in range(n_blocks)])
        self.nfno = nfno
        self.fno_conv = SpectralConv2d(nf, nf, m1=50, m2=1)
        self.fno_ws = nn.Conv2d(nf, nf, 1)
        self.tail = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.body(self.upsample(self.body(self.head(x))))
        for i in range(self.nfno):
            x = F.gelu(self.fno_conv(x) + self.fno_ws(x))
        return self.tail(self.body(x)).float()
