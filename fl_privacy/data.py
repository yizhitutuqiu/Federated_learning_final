from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class Partition:
    client_id: int
    indices: np.ndarray


def get_mnist(data_dir: str):
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)
    return train, test


def dirichlet_partitions(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx = class_indices[c].copy()
        rng.shuffle(idx)
        proportions = rng.dirichlet(alpha=np.ones(num_clients) * alpha)
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        shards = np.split(idx, splits)
        for k, shard in enumerate(shards):
            client_indices[k].append(shard)

    partitions = []
    for k in range(num_clients):
        merged = np.concatenate(client_indices[k]) if client_indices[k] else np.array([], dtype=int)
        rng.shuffle(merged)
        partitions.append(Partition(client_id=k, indices=merged))
    return partitions


def make_client_loaders(
    train_dataset: Dataset,
    partitions: list[Partition],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    loaders: dict[int, DataLoader] = {}
    for part in partitions:
        subset: Dataset = Subset(train_dataset, indices=part.indices.tolist())
        loaders[part.client_id] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
        )
    return loaders


def make_test_loader(
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
):
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
