import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate a list of dataset items into a batch.

    Pads audio to the maximum length in the batch (needed for eval mode
    where clips have different lengths). In train mode all crops are the
    same size, so no actual padding occurs.

    Args:
        dataset_items (list[dict]): list of dicts from dataset.__getitem__.
            Each dict must contain "audio" of shape (1, T).
    Returns:
        batch (dict): {
            "audio":        (B, 1, T_max) float tensor,
            "audio_length": (B,)          int tensor — original lengths.
        }
    """
    audios = [item["audio"] for item in dataset_items]
    lengths = torch.tensor([a.shape[1] for a in audios])
    max_len = lengths.max().item()

    padded = torch.zeros(len(audios), 1, max_len)
    for i, a in enumerate(audios):
        padded[i, :, : a.shape[1]] = a

    return {"audio": padded, "audio_length": lengths}
