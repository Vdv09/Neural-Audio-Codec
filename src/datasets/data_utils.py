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
        is_train = partition == "train"
        dataloaders[partition] = instantiate(
            config.dataloader,
            dataset=datasets[partition],
            collate_fn=collate_fn,
            batch_size=config.dataloader.batch_size if is_train else 1,
            drop_last=is_train,
            shuffle=is_train,
            worker_init_fn=set_worker_seed,
        )

    return dataloaders
