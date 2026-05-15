import torch
import torch.nn.functional as F
from torch import nn

from src.loss.spectral_loss import MultiScaleSpectralLoss


class SoundStreamLoss(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        spectral_weight=1.0,
        adversarial_weight=1.0,
        feature_matching_weight=100.0,
        commitment_weight=1.0,
    ):
        super().__init__()

        self.spectral = MultiScaleSpectralLoss(sample_rate=sample_rate)
        self.spectral_weight = spectral_weight

        self.adversarial_weight = adversarial_weight
        self.feature_matching_weight = feature_matching_weight

        self.commitment_weight = commitment_weight

    def discriminator_loss(self, real_outputs, fake_outputs):
        losses = []

        for real_feats, fake_feats in zip(real_outputs, fake_outputs):
            losses.append(F.relu(1.0 - real_feats[-1]).mean())
            losses.append(F.relu(1.0 + fake_feats[-1]).mean())

        return torch.stack(losses).mean()

    def generator_loss(
        self, real_audio, fake_audio, real_outputs, fake_outputs, commitment_loss
    ):
        spectral = self.spectral(real_audio, fake_audio)
        adversarial = self.adversarial_loss(fake_outputs)
        feature_matching = self.feature_matching_loss(real_outputs, fake_outputs)

        total = (
            self.spectral_weight * spectral
            + self.adversarial_weight * adversarial
            + self.feature_matching_weight * feature_matching
            + self.commitment_weight * commitment_loss
        )

        return {
            "generator_loss": total,
            "spectral_loss": spectral,
            "adversarial_loss": adversarial,
            "feature_matching_loss": feature_matching,
            "commitment_loss": commitment_loss,
        }

    @staticmethod
    def adversarial_loss(fake_outputs):
        losses = []

        for fake_feats in fake_outputs:
            losses.append(F.relu(1.0 - fake_feats[-1]).mean())

        return torch.stack(losses).mean()

    @staticmethod
    def feature_matching_loss(real_outputs, fake_outputs):
        losses = []

        for real_feats, fake_feats in zip(real_outputs, fake_outputs):
            for real_f, fake_f in zip(real_feats[:-1], fake_feats[:-1]):
                losses.append(F.l1_loss(fake_f, real_f.detach()))

        return torch.stack(losses).mean()
