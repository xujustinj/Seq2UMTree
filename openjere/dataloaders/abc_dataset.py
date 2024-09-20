from abc import ABC, abstractmethod
import json
import os
from typing import Any, Callable, Optional

from torch.utils.data import Dataset, DataLoader

from openjere.config import Hyper


class Abstract_dataset(ABC, Dataset):
    def __init__(self, hyper: Hyper, dataset: str):
        self.hyper = hyper
        self.data_root = hyper.data_root

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, "word_vocab.json"), "r", encoding="utf-8")
        )
        self.relation_vocab = json.load(
            open(
                os.path.join(self.data_root, "relation_vocab.json"),
                "r",
                encoding="utf-8",
            )
        )
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, "bio_vocab.json"), "r", encoding="utf-8")
        )

        self.tokenizer = self.hyper.tokenizer

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

class PartialDataLoader:
    def __init__(self, batch_reader: Callable[[Any], object]):
        self._batch_reader = batch_reader

    def __call__(
            self,
            dataset: Abstract_dataset,
            batch_size: Optional[int] = 1,
            num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            collate_fn=(lambda x: self._batch_reader(x)),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
