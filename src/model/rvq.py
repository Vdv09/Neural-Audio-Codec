import math

import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, dim, codebook_size, ema_decay=0.99):
        super().__init__()

        self.codebook_size = codebook_size
        self.ema_decay = ema_decay

        scale = 1.0 / math.sqrt(dim)

        self.register_buffer(
            "codebook", torch.empty(codebook_size, dim).uniform_(-scale, scale)
        )
        self.register_buffer("ema_counts", torch.ones(codebook_size))
        self.register_buffer("ema_sums", self.codebook.clone())

    def find_nearest(self, x):
        dists = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )
        return dists.argmin(dim=1)

    @torch.no_grad()
    def update_ema(self, x, indices):
        one_hot = F.one_hot(indices, self.codebook_size).float()

        counts = one_hot.sum(0)
        sums = one_hot.t() @ x

        self.ema_counts.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        self.ema_sums.mul_(self.ema_decay).add_(sums, alpha=1 - self.ema_decay)
        self.codebook.copy_(
            self.ema_sums / self.ema_counts.clamp_min(1e-5).unsqueeze(1)
        )

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        indices = self.find_nearest(flat)

        quantized = F.embedding(indices, self.codebook).view_as(x)
        commitment_loss = F.mse_loss(x, quantized.detach())

        quantized_stable = x + (quantized - x).detach()

        if self.training:
            self.update_ema(flat.detach(), indices)

        return quantized_stable, indices.view(x.shape[:-1]), commitment_loss


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dim, num_quantizers=8, codebook_size=1024, ema_decay=0.99):
        super().__init__()

        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(dim, codebook_size, ema_decay)
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z):
        z_t = z.transpose(1, 2)

        residual = z_t
        quantized = torch.zeros_like(z_t)
        all_indices = []
        total_commitment = 0.0

        for vq in self.quantizers:
            q, idx, commitment = vq(residual)

            quantized = quantized + q
            residual = residual - q.detach()

            all_indices.append(idx)

            total_commitment = total_commitment + commitment

        indices = torch.stack(all_indices, dim=-1)

        return {
            "quantized": quantized.transpose(1, 2),
            "indices": indices,
            "commitment_loss": total_commitment / len(self.quantizers),
            "codebook_perplexity": self.perplexity(indices),
        }

    @torch.no_grad()
    def perplexity(self, indices):
        perplexities = []
        K = self.quantizers[0].codebook_size

        for i in range(indices.shape[-1]):
            counts = torch.bincount(indices[:, :, i].reshape(-1), minlength=K).float()
            probs = counts / counts.sum().clamp_min(1.0)

            entropy = -(probs * (probs + 1e-10).log()).sum()

            perplexities.append(entropy.exp())

        return torch.stack(perplexities).mean()
