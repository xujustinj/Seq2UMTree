from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from torch.utils.data import Dataset, DataLoader

from openjere.config import Hyper


class Abstract_dataset(ABC, Dataset):
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

class PartialDataLoader:
    def __init__(self, batch_reader: Callable[[Any], object]):
        self._batch_reader = batch_reader

    def __call__(
            self,
            dataset: Abstract_dataset,
            batch_size: Optional[int] = 1,
            num_workers: int = 0,
            shuffle: Optional[bool] = None,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            collate_fn=(lambda x: self._batch_reader(x)),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )
