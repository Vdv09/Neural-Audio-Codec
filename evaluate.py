import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from pystoi import stoi
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
from tqdm import tqdm

from src.model import SoundStream

SAMPLE_RATE = 16000


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SoundStream().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def run(model, data_dir, device):
    paths = sorted(Path(data_dir).rglob("*.flac"))

    nisqa = NonIntrusiveSpeechQualityAssessment(fs=SAMPLE_RATE).to(device)

    stoi_scores = []
    nisqa_scores = []

    for path in tqdm(paths):
        audio, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

        audio = audio[:1].unsqueeze(0).to(device)

        fake = model(audio)["audio_hat"]

        real_np = audio.squeeze().cpu().numpy()
        fake_np = fake.squeeze().cpu().numpy()

        stoi_scores.append(stoi(real_np, fake_np, SAMPLE_RATE, extended=False))

        nisqa_scores.append(nisqa(fake.squeeze(1)).item())

    print(f"STOI: {np.mean(stoi_scores)} (n={len(stoi_scores)})")
    print(f"NISQA: {np.mean(nisqa_scores)} (n={len(nisqa_scores)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data/LibriSpeech/test-clean")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    run(model, args.data_dir, device)
