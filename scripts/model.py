import torch
import torch.nn as nn
import torch.fft as fft
import warnings

# -------- SRCNN（输出残差） --------
class SRCNN(nn.Module):
    def __init__(self, scale_factor=(5, 10)):
        super().__init__()
        self.pre = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 5, padding=2)
        )

    def forward(self, x):
        up = self.pre(x)
        return self.net(up).float()  # 残差


# -------- U-Net（输出残差） --------
class PowderUNet(nn.Module):
    def __init__(self, scale_factor=(5, 10)):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.dec2 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        u = self.up(x2)
        x1u = nn.functional.interpolate(x1, size=u.shape[-2:], mode='bilinear', align_corners=False)
        out = self.dec2(self.dec1(torch.cat([u, x1u], dim=1)))
        return out.float()  # 残差


# -------- FNO2d（输出残差） --------
class SpectralConv2d(nn.Module):
    def __init__(self, in_c, out_c, m1, m2):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_c, out_c, m1, m2, 2) * (1 / (in_c * out_c)))
        self.m1, self.m2 = m1, m2

    def forward(self, x):
        # x: (B,in_c,H,W)
        B, C, H, W = x.shape
        x_ft = fft.rfftn(x, dim=(2, 3), norm="ortho")  # (B,in_c,H,W//2+1)
        Hf, Wf = x_ft.shape[2], x_ft.shape[3]
        m1, m2 = min(self.m1, Hf), min(self.m2, Wf)
        out_ft = torch.zeros((B, self.weights.shape[1], H, Wf), device=x.device, dtype=torch.cfloat)
        w = torch.view_as_complex(self.weights)  # (in_c,out_c,self.m1,self.m2)
        x_slice = x_ft[:, :, :m1, :m2]  # (B,in_c,m1,m2)
        w_slice = w[:, :, :m1, :m2]  # (in_c,out_c,m1,m2)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_slice, w_slice)
        return fft.irfftn(out_ft, s=(H, W), dim=(2, 3), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, modes1=20, modes2=10, width=64, output_size=(100,200)):
        super().__init__()
        self.fc0 = nn.Linear(1, width)
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.output_size = output_size

    def forward(self, x):
        # x: (B,1,20,20) -> 残差
        x = x.permute(0, 2, 3, 1)  # (B,H,W,1)
        x = self.fc0(x).permute(0, 3, 1, 2)  # (B,width,H,W)
        for c, w in zip(self.convs, self.ws):
            x = torch.nn.functional.gelu(c(x) + w(x))
        x = torch.nn.functional.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1)  # (B,100,200,width)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x).permute(0, 3, 1, 2).float()  # (B,1,100,200)
        return x  # 残差

# -------- EDSR ( https://arxiv.org/pdf/1707.02921.pdf ) --------
# Based on: https://github.com/Lornatang/EDSR-PyTorch/
class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.add(torch.mul(self.rcb(x), 0.1), x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels, upscale_factor):
        super().__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * upscale_factor**2, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample_block(x)


class EDSR(nn.Module):
    def __init__(self, scale_factor=(4,4)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.net = nn.Sequential(
            *tuple([ResidualConvBlock(64)]*16),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        )
        if scale_factor[0] != scale_factor[1]:
            warnings.warn(f'x- and y- scale_factors not the same. Will use a less efficient algorithm')
            self.upsampling = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1))
        else: # Use PixelShuffle like in original implementation
            if scale_factor[0] > 10 or scale_factor[0] < 2:
                raise RuntimeError('scale factor must be between 2 and 10')
            scales = {2:[2], 3:[3], 4:[2,2], 5:[5], 6:[3,2], 7:[7], 8:[2,2,2], 9:[3,3], 10:[5,2]}
            self.upsampling = nn.Sequential(
                *tuple([UpsampleBlock(64, n) for n in scales[scale_factor[0]]]),
                nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1),
            )
    def forward(self, x):
        out1 = self.conv1(x)
        return self.upsampling(torch.add(self.net(out1), out1)).float()
