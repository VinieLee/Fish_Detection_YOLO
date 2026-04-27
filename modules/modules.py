import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class PConv(nn.Module):
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        # Division de canales para reducir redundancia
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), dim=1)

class FasterBlock(nn.Module):
    def __init__(self, c1, c2, n_div=4, expansion=2):
        super().__init__()
        # c1 es la entrada, c2 es la salida (deben ser iguales para el shortcut)
        self.conv = PConv(c1, n_div)
        hidden_dim = int(c1 * expansion)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, c2, 1, bias=False)
        )
        self.shortcut = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x):
        return self.shortcut(x) + self.mlp(self.conv(x))

class C2ICAREBlock(nn.Module):
    def __init__(self, dim, mem_ratio=0.25):
        super().__init__()
        mem_dim = int(dim * mem_ratio)
        feat_dim = dim - mem_dim

        self.mem_dim = mem_dim
        self.feat_dim = feat_dim

        # Multi-scale depthwise
        self.dw3 = nn.Conv2d(feat_dim, feat_dim, 3, padding=1, groups=feat_dim)
        self.dw7 = nn.Conv2d(feat_dim, feat_dim, 7, padding=3, groups=feat_dim)

        # Mem → Feat interaction
        self.mem_proj = nn.Conv2d(mem_dim, feat_dim, 1)

        # FFN tipo ConvNeXt (más estable)
        self.norm = nn.BatchNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * 2, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * 2, dim, 1)

        # Layer scale estable
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(self, x):
        identity = x

        mem, feat = torch.split(x, [self.mem_dim, self.feat_dim], dim=1)

        # Multi-scale
        feat = self.dw3(feat) + self.dw7(feat)

        # Interacción mem → feat
        feat = feat + self.mem_proj(mem)

        x = torch.cat([mem, feat], dim=1)

        # FFN
        x = self.pw2(self.act(self.pw1(self.norm(x))))

        return identity + x * self.gamma.view(1, -1, 1, 1)

class C2ICARE(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2

        self.c = int(c1 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.blocks = nn.Sequential(*[
            C2ICAREBlock(self.c, mem_ratio=0.25)
            for _ in range(n)
        ])

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)

        b = self.blocks(b)

        return self.cv2(torch.cat((a, b), dim=1))
