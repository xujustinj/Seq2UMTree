from dataclasses import dataclass
from functools import cached_property
import json
import os
from typing import Dict, List, Literal, Optional, Tuple


ModelName = Literal[
    "selection",
    "twotagging",
    "seq2umt",
    "wdec",
    "copymtl",
]

OptimizerName = Literal[
    "adam",
    "sgd",
]

ComponentName = Literal[
    "subject",
    "predicate",
    "object",
]

@dataclass
class Hyper(object):
    def __init__(self, path: str):
        self.dataset: str
        self.model: ModelName

        self.data_root: str
        self.raw_data_root: str
        self.train: str
        self.dev: str
        self.test: str
        self.subsets: List[str]
        self.raw_data_list: List[str]

        self.relation_vocab: str
        self.print_epoch: int
        self.max_text_len: int

        self.max_decode_len: Optional[int]
        self.max_encode_len: Optional[int]

        self.cell_name: str
        self.emb_size: int
        self.char_emb_size: int
        self.rel_emb_size: int
        self.bio_emb_size: int
        self.hidden_size: int
        self.dropout: float = 0.5
        self.threshold: float = 0.5
        self.activation: str
        self.optimizer: OptimizerName
        self.epoch_num: int
        self.batch_size_train: int
        self.batch_size_eval: int
        self.seperator: str
        self.gpu: int

        with open(path, "r") as f:
            self.__dict__ = json.load(f)

        o1: ComponentName
        o2: ComponentName
        o3: ComponentName
        o1, o2, o3 = self.__dict__["order"]
        assert isinstance(o1, str)
        assert isinstance(o2, str)
        assert isinstance(o3, str)
        self.order: Tuple[ComponentName, ComponentName, ComponentName] = (o1, o2, o3)

    @property
    def word_vocab_path(self) -> str:
        return os.path.join(self.data_root, "word_vocab.json")

    @cached_property
    def word2id(self) -> Dict[str, int]:
        with open(self.word_vocab_path, "r", encoding="utf-8") as f:
            word2id = json.load(f)
        assert isinstance(word2id, dict)
        return word2id

    @property
    def relation_vocab_path(self) -> str:
        return os.path.join(self.data_root, "relation_vocab.json")

    @cached_property
    def rel2id(self) -> Dict[str, int]:
        with open(self.relation_vocab_path, "r", encoding="utf-8") as f:
            rel2id = json.load(f)
        assert isinstance(rel2id, dict)
        return rel2id

    @property
    def bio_vocab_path(self) -> str:
        return os.path.join(self.data_root, "bio_vocab.json")

    @cached_property
    def bio_vocab(self) -> Dict[str, int]:
        with open(self.bio_vocab_path, "r", encoding="utf-8") as f:
            bio_vocab = json.load(f)
        assert isinstance(bio_vocab, dict)
        return bio_vocab

    @cached_property
    def id2word(self) -> Dict[int, str]:
        return {k: v for v, k in self.word2id.items()}

    @cached_property
    def id2rel(self) -> Dict[int, str]:
        return {k: v for v, k in self.rel2id.items()}

    @cached_property
    def id2bio(self) -> Dict[int, str]:
        return {k: v for v, k in self.bio_vocab.items()}

    def join(self, toks: List[str]) -> str:
        if self.seperator == "":
            return "".join(toks)
        elif self.seperator == " ":
            return " ".join(toks)
        else:
            raise NotImplementedError("other tokenizer?")

    def tokenizer(self, text: str) -> List[str]:
        if self.seperator == "":
            return list(text)
        elif self.seperator == " ":
            return text.split(" ")
        else:
            raise NotImplementedError("other tokenizer?")
