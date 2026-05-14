import random
from pathlib import Path

import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset


class LibrispeechDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        train_mode,
        sample_rate=16000,
        crop_seconds=0.5,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        self.sample_rate = sample_rate
        self.train_mode = train_mode
        self.crop_size = int(crop_seconds * sample_rate)

        index = [
            {"path": str(p), "label": 0} for p in sorted(Path(data_dir).rglob("*.flac"))
        ]

        super().__init__(index, limit, shuffle_index, instance_transforms)

    def load_object(self, path):
        audio = torchaudio.load(path)[0]

        return audio

    def _crop_and_pad(self, audio):
        length = audio.shape[1]

        if length >= self.crop_size:
            start = random.randint(0, length - self.crop_size)
            return audio[:, start : start + self.crop_size]

        pad = audio[:, -1:].expand(-1, self.crop_size - length)
        return torch.cat([audio, pad], dim=1)

    def __getitem__(self, ind):
        item = self._index[ind]
        audio = self.load_object(item["path"])

        if self.train_mode:
            audio = self._crop_and_pad(audio)

        return self.preprocess_data({"audio": audio})
