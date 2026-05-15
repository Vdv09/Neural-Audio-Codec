from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    for loader in repeat(dataloader):
        yield from loader


def get_dataloaders(config):
    datasets = instantiate(config.datasets)

    dataloaders = {}
    for partition in config.datasets.keys():
        dataloaders[partition] = instantiate(
            config.dataloader,
            dataset=datasets[partition],
            collate_fn=collate_fn,
            drop_last=(partition == "train"),
            shuffle=(partition == "train"),
            worker_init_fn=set_worker_seed,
        )

    return dataloaders
