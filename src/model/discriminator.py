import torch
import torch.nn.functional as F
from torch import nn


class ScaleDiscriminator(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()

        C = base_channels
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(1, C, kernel_size=15, padding=7),
                nn.Conv1d(C, C * 4, kernel_size=41, stride=4, padding=20, groups=4),
                nn.Conv1d(
                    C * 4, C * 16, kernel_size=41, stride=4, padding=20, groups=16
                ),
                nn.Conv1d(
                    C * 16,
                    min(C * 64, 1024),
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=64,
                ),
                nn.Conv1d(
                    min(C * 64, 1024), min(C * 64, 1024), kernel_size=5, padding=2
                ),
                nn.Conv1d(min(C * 64, 1024), 1, kernel_size=3, padding=1),
            ]
        )

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        features = []

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            features.append(x)

            if i < len(self.layers) - 1:
                x = self.act(x)

        return features


class MultiScaleWaveDiscriminator(nn.Module):
    def __init__(self, base_channels=32, num_scales=3):
        super().__init__()

        self.discriminators = nn.ModuleList(
            [ScaleDiscriminator(base_channels) for _ in range(num_scales)]
        )

    def forward(self, x):
        outputs = []

        for i in range(len(self.discriminators)):
            if i > 0:
                x = F.avg_pool1d(x, kernel_size=4, stride=2, padding=1)

            outputs.append(self.discriminators[i](x))

        return outputs


class STFTDiscriminator(nn.Module):
    def __init__(self, base_channels=32, n_fft=1024, hop_length=256):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        C = base_channels

        channels = [C, C, C * 2, C * 4, C * 4, C * 8, C * 8]
        strides = [(1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)]

        layers = [nn.Conv2d(2, channels[0], kernel_size=7, padding=3)]

        for i in range(len(strides)):
            layers.append(
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=(3, 4),
                    stride=strides[i],
                    padding=(1, 1),
                )
            )

        layers.append(nn.Conv2d(channels[-1], 1, kernel_size=(1, 8)))

        self.layers = nn.ModuleList(layers)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.squeeze(1)

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )

        spec = spec.transpose(1, 2)
        features_2d = torch.stack([spec.real, spec.imag], dim=1)

        outputs = []

        for i in range(len(self.layers)):
            features_2d = self.layers[i](features_2d)
            outputs.append(features_2d)

            if i < len(self.layers) - 1:
                features_2d = self.act(features_2d)

        return outputs


class SoundStreamDiscriminator(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()

        self.wave = MultiScaleWaveDiscriminator(base_channels)
        self.stft = STFTDiscriminator(base_channels)

    def forward(self, audio):
        return self.wave(audio) + [self.stft(audio)]
