import torch
import torch.nn.functional as F
from torch import nn

from src.model.causal import CausalConv1d, DecoderBlock, EncoderBlock
from src.model.rvq import ResidualVectorQuantizer


class Encoder(nn.Module):
    def __init__(self, channels=32, latent_dim=128, strides=(2, 4, 5, 5)):
        super().__init__()

        layers = [CausalConv1d(1, channels, kernel_size=7)]

        in_ch = channels

        for i in range(len(strides)):
            out_ch = channels * 2 ** (i + 1)

            layers.append(EncoderBlock(in_ch, out_ch, strides[i]))
            in_ch = out_ch

        layers.append(CausalConv1d(in_ch, latent_dim, kernel_size=3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, channels=32, latent_dim=128, strides=(2, 4, 5, 5)):
        super().__init__()

        n = len(strides)
        in_ch = channels * (2**n)
        strides_rev = list(reversed(strides))

        layers = [CausalConv1d(latent_dim, in_ch, kernel_size=7)]

        for i in range(len(strides_rev)):
            out_ch = channels * 2 ** (n - i - 1)

            layers.append(DecoderBlock(in_ch, out_ch, strides_rev[i]))
            in_ch = out_ch

        layers.append(CausalConv1d(channels, 1, kernel_size=7))
        self.net = nn.Sequential(*layers)

    def forward(self, z, target_len=None):
        x = self.net(z)
        if target_len is not None:
            x = self.match_length(x, target_len)
        return x

    @staticmethod
    def match_length(x, target_len):
        if x.shape[-1] > target_len:
            return x[..., :target_len]
        if x.shape[-1] < target_len:
            return F.pad(x, (0, target_len - x.shape[-1]))
        return x


class SoundStream(nn.Module):
    def __init__(
        self,
        channels=32,
        latent_dim=128,
        strides=(2, 4, 5, 5),
        num_quantizers=8,
        codebook_size=1024,
        ema_decay=0.99,
    ):
        super().__init__()

        self.encoder = Encoder(channels, latent_dim, strides)
        self.rvq = ResidualVectorQuantizer(
            latent_dim, num_quantizers, codebook_size, ema_decay
        )
        self.decoder = Decoder(channels, latent_dim, strides)

    def forward(self, audio, **_):
        target_len = audio.shape[-1]
        z = self.encoder(audio)
        q = self.rvq(z)
        audio_hat = self.decoder(q["quantized"], target_len=target_len)

        return {
            "audio_hat": audio_hat,
            "commitment_loss": q["commitment_loss"],
            "codebook_perplexity": q["codebook_perplexity"],
        }

    @torch.no_grad()
    def encode(self, audio):
        return self.rvq(self.encoder(audio))["indices"]

    @torch.no_grad()
    def reconstruct(self, audio):
        return self.forward(audio)["audio_hat"]
