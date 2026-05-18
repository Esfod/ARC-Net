"""
Multi-scale degradation encoder.

Produces a (B, 736) feature vector by Global-Average-Pooling each of five
conv stages and concatenating — the AirNet/MCDRNet-style representation.
Used as a dense conditioning signal for the Diffusion U-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEG_VEC_DIM = 32 + 64 + 128 + 256 + 256  # = 736


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DegradationEncoder(nn.Module):
    """
    Input:  (B, 3, H, W)  — any crop size
    Output: (B, 736)      — multi-scale degradation feature vector
    """

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(_conv_block(3,   32),  nn.MaxPool2d(2))   # H/2
        self.stage2 = nn.Sequential(_conv_block(32,  64),  nn.MaxPool2d(2))   # H/4
        self.stage3 = nn.Sequential(_conv_block(64,  128), nn.MaxPool2d(2))   # H/8
        self.stage4 = nn.Sequential(_conv_block(128, 256), nn.MaxPool2d(2))   # H/16
        self.stage5 = _conv_block(256, 256)                                    # H/16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.stage1(x)
        h2 = self.stage2(h1)
        h3 = self.stage3(h2)
        h4 = self.stage4(h3)
        h5 = self.stage5(h4)

        def gap(t):
            return F.adaptive_avg_pool2d(t, 1).flatten(1)

        return torch.cat([gap(h1), gap(h2), gap(h3), gap(h4), gap(h5)], dim=1)
