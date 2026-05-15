import torch
import torch.nn.functional as F
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()

        self.pad = dilation * (kernel_size - 1)

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()

        self.net = nn.Sequential(
            nn.ELU(),
            CausalConv1d(channels, channels, kernel_size=7, dilation=dilation),
            nn.ELU(),
            CausalConv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.net = nn.Sequential(
            ResidualUnit(in_channels, dilation=1),
            ResidualUnit(in_channels, dilation=3),
            ResidualUnit(in_channels, dilation=9),
            nn.ELU(),
            CausalConv1d(
                in_channels, out_channels, kernel_size=2 * stride, stride=stride
            ),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=(stride + 1) // 2,
                output_padding=stride % 2,
            ),
            ResidualUnit(out_channels, dilation=1),
            ResidualUnit(out_channels, dilation=3),
            ResidualUnit(out_channels, dilation=9),
        )

    def forward(self, x):
        return self.net(x)
