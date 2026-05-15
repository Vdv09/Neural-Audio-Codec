from collections import defaultdict

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.utils.io_utils import ROOT_PATH


class SoundStreamTrainer:
    def __init__(
        self,
        model,
        discriminator,
        criterion,
        metrics,
        optimizer_g,
        optimizer_d,
        config,
        device,
        dataloaders,
        logger,
        writer,
    ):
        self.model = model

        self.discriminator = discriminator
        self.criterion = criterion

        self.metrics = metrics

        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d

        self.cfg = config.trainer
        self.device = device

        self.logger = logger
        self.writer = writer

        self.train_loader = inf_loop(dataloaders["train"])
        self.eval_loaders = {k: v for k, v in dataloaders.items() if k != "train"}

        self.checkpoint_dir = ROOT_PATH / self.cfg.save_dir / config.writer.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.best_metric = float("-inf")

        if self.cfg.get("resume_from"):
            self.load_checkpoint(self.checkpoint_dir / self.cfg.resume_from)

    def train(self):
        for epoch in range(1, self.cfg.n_epochs + 1):
            self.train_epoch(epoch)

            if self.step >= self.cfg.total_steps:
                break

        self.save_checkpoint("model_last.pth")

    def train_epoch(self, epoch):
        self.model.train()
        self.discriminator.train()

        avg = defaultdict(list)

        for _ in tqdm(range(self.cfg.epoch_len), desc=f"epoch {epoch}"):
            batch = next(self.train_loader)
            batch = self.to_device(batch)

            losses, outputs = self.train_step(batch)
            self.step += 1

            for name, val in losses.items():
                avg[name].append(float(val))

            avg["perplexity"].append(outputs["codebook_perplexity"].item())

            if self.step % self.cfg.log_step == 0:
                self.writer.set_step(self.step, "train")
                self.writer.add_scalars({k: sum(v) / len(v) for k, v in avg.items()})

                avg = defaultdict(list)

            if self.step % self.cfg.audio_log_step == 0:
                self.writer.set_step(self.step, "train")

                self.writer.add_audio(
                    "original", batch["audio"][0], self.cfg.sample_rate
                )
                self.writer.add_audio(
                    "reconstructed", outputs["audio_hat"][0], self.cfg.sample_rate
                )

            if self.step % self.cfg.eval_step == 0:
                self.evaluate_all(epoch)

                self.model.train()
                self.discriminator.train()

            if self.step % self.cfg.save_period == 0:
                self.save_checkpoint(f"checkpoint_step{self.step}.pth")

            if self.step >= self.cfg.total_steps:
                break

    def train_step(self, batch):
        audio = batch["audio"]
        outputs = self.model(audio)
        fake = outputs["audio_hat"]

        self.optimizer_d.zero_grad()
        d_loss = self.criterion.discriminator_loss(
            self.discriminator(audio), self.discriminator(fake.detach())
        )
        d_loss.backward()
        self.optimizer_d.step()

        self.optimizer_g.zero_grad()

        with torch.no_grad():
            real_g = self.discriminator(audio)

        g_losses = self.criterion.generator_loss(
            audio, fake, real_g, self.discriminator(fake), outputs["commitment_loss"]
        )
        g_losses["generator_loss"].backward()

        clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

        self.optimizer_g.step()

        return {"discriminator_loss": d_loss, **g_losses}, outputs

    @torch.no_grad()
    def evaluate_all(self, epoch):
        for part, loader in self.eval_loaders.items():
            logs, last_batch, last_out = self.evaluate(loader)

            self.writer.set_step(self.step, part)

            self.writer.add_scalars(logs)
            self.writer.add_audio(
                "original", last_batch["audio"][0], self.cfg.sample_rate
            )
            self.writer.add_audio(
                "reconstructed", last_out["audio_hat"][0], self.cfg.sample_rate
            )

            log_line = ", ".join(f"{k}: {v:.4f}" for k, v in logs.items())  # noqa: E231
            self.logger.info(f"epoch {epoch} step {self.step} {part}: {log_line}")

            if part == "test":
                score = logs.get(self.cfg.get("monitor_metric", "STOI"), float("-inf"))

                if score > self.best_metric:
                    self.best_metric = score
                    self.save_checkpoint("model_best.pth")

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        self.discriminator.eval()

        avg = defaultdict(list)
        last_batch = last_out = None

        for batch in tqdm(loader, desc="eval"):
            batch = self.to_device(batch)
            out = self.model(batch["audio"])

            real = self.discriminator(batch["audio"])
            fake = self.discriminator(out["audio_hat"])

            avg["discriminator_loss"].append(
                self.criterion.discriminator_loss(real, fake).item()
            )

            for name, val in self.criterion.generator_loss(
                batch["audio"], out["audio_hat"], real, fake, out["commitment_loss"]
            ).items():
                avg[name].append(float(val))

            avg["perplexity"].append(out["codebook_perplexity"].item())

            for metric in self.metrics:
                avg[metric.name].append(metric(batch["audio"], out["audio_hat"]))

            last_batch, last_out = batch, out

        return {k: sum(v) / len(v) for k, v in avg.items()}, last_batch, last_out

    def to_device(self, batch):
        return {
            k: v.to(self.device) if k in self.cfg.device_tensors else v
            for k, v in batch.items()
        }

    def save_checkpoint(self, name):
        path = self.checkpoint_dir / name

        torch.save(
            {
                "step": self.step,
                "best_metric": self.best_metric,
                "model": self.model.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
            },
            path,
        )

        self.logger.info(f"Saved: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model"])

        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d"])

        self.step = ckpt.get("step", 0)
        self.best_metric = ckpt.get("best_metric", float("-inf"))

        self.logger.info(f"Loaded: {path}, step {self.step}")
