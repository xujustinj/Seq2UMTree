from dataclasses import dataclass
from functools import cached_property
import json
import os
from typing import Optional

import torch

from .types import ComponentName, OptimizerName


@dataclass
class Seq2UMTreeConfig:
    def __init__(self, path: str):
        self.dataset: str

        self.data_root: str
        self.raw_data_root: str
        self.train: str
        self.dev: str
        self.test: str
        self.raw_data_list: list[str]

        self.relation_vocab: str
        self.max_text_len: int

        self.max_decode_len: Optional[int]
        self.max_encode_len: Optional[int]

        self.emb_size: int
        self.hidden_size: int
        self.dropout: float = 0.5
        self.threshold: float = 0.5
        self.optimizer: OptimizerName
        self.epoch_num: int
        self.batch_size_train: int
        self.batch_size_eval: int
        self.seperator: str

        with open(path, "r") as f:
            self.__dict__ = json.load(f)

        o1: ComponentName
        o2: ComponentName
        o3: ComponentName
        o1, o2, o3 = self.__dict__["order"]
        assert isinstance(o1, str)
        assert isinstance(o2, str)
        assert isinstance(o3, str)
        self.order: tuple[ComponentName, ComponentName, ComponentName] = (o1, o2, o3)

    @cached_property
    def threshold_logit(self) -> float:
        assert 0.0 <= self.threshold <= 1.0
        return float(torch.logit(torch.tensor(self.threshold)).item())

    @property
    def word_vocab_path(self) -> str:
        return os.path.join(self.data_root, "word_vocab.json")

    @cached_property
    def word2id(self) -> dict[str, int]:
        with open(self.word_vocab_path, "r", encoding="utf-8") as f:
            word2id = json.load(f)
        assert isinstance(word2id, dict)
        return word2id

    @property
    def relation_vocab_path(self) -> str:
        return os.path.join(self.data_root, "relation_vocab.json")

    @cached_property
    def rel2id(self) -> dict[str, int]:
        with open(self.relation_vocab_path, "r", encoding="utf-8") as f:
            rel2id = json.load(f)
        assert isinstance(rel2id, dict)
        return rel2id

    @cached_property
    def id2word(self) -> dict[int, str]:
        return {k: v for v, k in self.word2id.items()}

    @cached_property
    def id2rel(self) -> dict[int, str]:
        return {k: v for v, k in self.rel2id.items()}

    def join(self, toks: list[str]) -> str:
        if self.seperator == "":
            return "".join(toks)
        elif self.seperator == " ":
            return " ".join(toks)
        else:
            raise NotImplementedError("other tokenizer?")

    def tokenizer(self, text: str) -> list[str]:
        if self.seperator == "":
            return list(text)
        elif self.seperator == " ":
            return text.split(" ")
        else:
            raise NotImplementedError("other tokenizer?")
