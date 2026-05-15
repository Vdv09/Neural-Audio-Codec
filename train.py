import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import SoundStreamTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="soundstream")
def main(config):
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.trainer.device != "auto":
        device = config.trainer.device

    dataloaders = get_dataloaders(config)

    model = instantiate(config.model).to(device)
    discriminator = instantiate(config.discriminator).to(device)
    loss = instantiate(config.loss).to(device)
    metrics = list(instantiate(config.metrics).get("inference", []))

    optimizer_g = instantiate(config.optimizer_g, params=model.parameters())
    optimizer_d = instantiate(config.optimizer_d, params=discriminator.parameters())

    trainer = SoundStreamTrainer(
        model=model,
        discriminator=discriminator,
        criterion=loss,
        metrics=metrics,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
