import math

import torch
import torch.nn.functional as F
from torch import nn
from torchaudio.functional import melscale_fbanks


class MultiScaleSpectralLoss(nn.Module):
    def __init__(
        self, sample_rate=16000, fft_sizes=(64, 128, 256, 512, 1024, 2048), n_mels=64
    ):
        super().__init__()

        self.fft_sizes = fft_sizes
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        for n_fft in fft_sizes:
            n_freqs = n_fft // 2 + 1
            n_mels_actual = min(n_mels, max(1, (n_freqs - 1) * 3 // 8))

            self.register_buffer(
                f"window_{n_fft}", torch.hann_window(n_fft), persistent=False
            )
            self.register_buffer(
                f"mel_{n_fft}",
                melscale_fbanks(
                    n_freqs=n_freqs,
                    f_min=0.0,
                    f_max=float(sample_rate // 2),
                    n_mels=n_mels_actual,
                    sample_rate=sample_rate,
                    norm=None,
                    mel_scale="htk",
                ),
                persistent=False,
            )

    def forward(self, real, fake):
        real = real.squeeze(1)
        fake = fake.squeeze(1)

        losses = []

        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            window = getattr(self, f"window_{n_fft}")
            mel_fb = getattr(self, f"mel_{n_fft}")

            real_mag = self.magnitude(real, n_fft, hop, window)
            fake_mag = self.magnitude(fake, n_fft, hop, window)

            real_mel = real_mag.transpose(1, 2) @ mel_fb
            fake_mel = fake_mag.transpose(1, 2) @ mel_fb

            l1 = F.l1_loss(fake_mel, real_mel)
            log = F.mse_loss((fake_mel + 1e-7).log(), (real_mel + 1e-7).log())

            losses.append(l1 + math.sqrt(n_fft / 2.0) * log)

        return torch.stack(losses).mean()

    @staticmethod
    def magnitude(audio, n_fft, hop, window):
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        return spec.abs()
