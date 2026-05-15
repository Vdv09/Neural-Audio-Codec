import numpy as np
from pystoi import stoi

from src.metrics.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(self, sample_rate=16000, name="STOI"):
        super().__init__(name=name)

        self.sample_rate = sample_rate

    def __call__(self, real, fake):
        real = real.squeeze(1).detach().cpu().numpy()
        fake = fake.squeeze(1).detach().cpu().numpy()

        scores = [
            stoi(real[i], fake[i], self.sample_rate, extended=False)
            for i in range(real.shape[0])
        ]

        return float(np.mean(scores))
