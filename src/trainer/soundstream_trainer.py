import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH

LOSS_NAMES = [
    "generator_loss",
    "discriminator_loss",
    "spectral_loss",
    "adversarial_loss",
    "feature_matching_loss",
    "commitment_loss",
]


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

        self.config = config
        self.cfg = config.trainer

        self.device = device
        self.logger = logger
        self.writer = writer

        self.train_loader = inf_loop(dataloaders["train"])
        self.eval_loaders = {k: v for k, v in dataloaders.items() if k != "train"}

        self.checkpoint_dir = ROOT_PATH / self.cfg.save_dir / config.writer.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_metric = float("-inf")

        self.train_metrics = MetricTracker(
            *LOSS_NAMES, "codebook_perplexity", writer=self.writer
        )
        self.eval_metrics = MetricTracker(
            *LOSS_NAMES,
            "codebook_perplexity",
            *[m.name for m in self.metrics],
            writer=self.writer,
        )

        if self.cfg.get("resume_from") is not None:
            self.load_checkpoint(self.checkpoint_dir / self.cfg.resume_from)

    def train(self):
        for epoch in range(1, self.cfg.n_epochs + 1):
            self.train_epoch(epoch)

            if self.global_step >= self.cfg.total_steps:
                break

        self.save_checkpoint("model_last.pth")

    def train_epoch(self, epoch):
        self.model.train()
        self.discriminator.train()
        self.train_metrics.reset()

        progress = tqdm(range(self.cfg.epoch_len), desc=f"train epoch {epoch}")

        for _ in progress:
            batch = next(self.train_loader)
            batch = self.move_batch_to_device(batch)

            losses, outputs = self.train_step(batch)

            self.global_step += 1

            for name, value in losses.items():
                self.train_metrics.update(
                    name, value.item() if hasattr(value, "item") else float(value)
                )

            self.train_metrics.update(
                "codebook_perplexity", outputs["codebook_perplexity"].item()
            )

            if self.global_step % self.cfg.log_step == 0:
                self.writer.set_step(self.global_step, "train")
                self.writer.add_scalars(self.train_metrics.result())
                self.train_metrics.reset()

            if self.global_step % self.cfg.audio_log_step == 0:
                self.log_audio(batch, outputs)

            if self.global_step % self.cfg.eval_step == 0:
                self.evaluate_all(epoch)
                self.model.train()
                self.discriminator.train()

            if self.global_step % self.cfg.save_period == 0:
                self.save_checkpoint(f"checkpoint_step{self.global_step}.pth")

            if self.global_step >= self.cfg.total_steps:
                break

    def train_step(self, batch):
        audio = batch["audio"]
        outputs = self.model(audio)
        fake_audio = outputs["audio_hat"]

        self.optimizer_d.zero_grad()

        real_d = self.discriminator(audio)
        fake_d = self.discriminator(fake_audio.detach())

        d_loss = self.criterion.discriminator_loss(real_d, fake_d)
        d_loss.backward()
        self.optimizer_d.step()

        self.optimizer_g.zero_grad()

        with torch.no_grad():
            real_g = self.discriminator(audio)

        fake_g = self.discriminator(fake_audio)

        g_losses = self.criterion.generator_loss(
            real_audio=audio,
            fake_audio=fake_audio,
            real_outputs=real_g,
            fake_outputs=fake_g,
            commitment_loss=outputs["commitment_loss"],
        )

        g_losses["generator_loss"].backward()
        clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer_g.step()

        return {"discriminator_loss": d_loss, **g_losses}, outputs

    @torch.no_grad()
    def evaluate_all(self, epoch):
        for part, loader in self.eval_loaders.items():
            logs, last_batch, last_outputs = self.evaluate(loader)

            self.writer.set_step(self.global_step, part)
            self.writer.add_scalars(logs)
            self.log_audio(last_batch, last_outputs)

            log_line = ", ".join(f"{k}: {v:.4f}" for k, v in logs.items())  # noqa: E231
            self.logger.info(
                f"epoch {epoch} step {self.global_step} {part}: {log_line}"
            )

            if part == "test":
                monitor_metric = logs.get(
                    self.cfg.get("monitor_metric", "STOI"), float("-inf")
                )
                if monitor_metric > self.best_metric:
                    self.best_metric = monitor_metric
                    self.save_checkpoint("model_best.pth")

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        self.discriminator.eval()
        self.eval_metrics.reset()

        last_batch = None
        last_outputs = None

        for batch in tqdm(loader, desc="eval"):
            batch = self.move_batch_to_device(batch)
            outputs = self.model(batch["audio"])

            real = self.discriminator(batch["audio"])
            fake = self.discriminator(outputs["audio_hat"])

            d_loss = self.criterion.discriminator_loss(real, fake)
            g_losses = self.criterion.generator_loss(
                real_audio=batch["audio"],
                fake_audio=outputs["audio_hat"],
                real_outputs=real,
                fake_outputs=fake,
                commitment_loss=outputs["commitment_loss"],
            )

            self.eval_metrics.update("discriminator_loss", d_loss.item())
            for name, value in g_losses.items():
                self.eval_metrics.update(name, value.item())

            self.eval_metrics.update(
                "codebook_perplexity", outputs["codebook_perplexity"].item()
            )

            for metric in self.metrics:
                self.eval_metrics.update(
                    metric.name, metric(batch["audio"], outputs["audio_hat"])
                )

            last_batch = batch
            last_outputs = outputs

        return self.eval_metrics.result(), last_batch, last_outputs

    def log_audio(self, batch, outputs):
        self.writer.add_audio("original", batch["audio"][0], self.cfg.sample_rate)
        self.writer.add_audio(
            "reconstructed", outputs["audio_hat"][0], self.cfg.sample_rate
        )

    def move_batch_to_device(self, batch):
        for key in self.cfg.device_tensors:
            if key in batch:
                batch[key] = batch[key].to(self.device)
        return batch

    def save_checkpoint(self, name):
        path = self.checkpoint_dir / name
        torch.save(
            {
                "step": self.global_step,
                "best_metric": self.best_metric,
                "model": self.model.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
                "config": self.config,
            },
            path,
        )
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        self.global_step = checkpoint.get("step", 0)
        self.best_metric = checkpoint.get("best_metric", float("-inf"))

        self.logger.info(f"Checkpoint loaded from {path}, step {self.global_step}")
