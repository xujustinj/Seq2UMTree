from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional, TypeVar

import torch
from torch.utils.data import Dataset, DataLoader

from openjere.config import Hyper


class AbstractData:
    device = torch.device("cpu")

    def pin_memory(self):
        return self

    def to(self, device: torch.device):
        self.device = device
        return self


class AbstractDataset(ABC, Dataset):
    def __init__(self, hyper: Hyper, dataset: str):
        self.hyper = hyper
        self.data_root = hyper.data_root

        self.word_vocab = hyper.word2id
        self.relation_vocab = hyper.rel2id
        self.bio_vocab = hyper.bio_vocab

        self.tokenizer = self.hyper.tokenizer

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass


Data = TypeVar("Data")

class PartialDataLoader(Generic[Data]):
    def __init__(self, collate_fn: Callable[[Any], Data]):
        self.collate_fn = collate_fn

    def __call__(
            self,
            dataset: AbstractDataset,
            batch_size: Optional[int] = 1,
            num_workers: int = 0,
            shuffle: Optional[bool] = None,
    ) -> DataLoader[Data]:
        return DataLoader(
            dataset=dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )
