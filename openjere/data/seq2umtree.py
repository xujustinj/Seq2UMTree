import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from openjere.config import Hyper, ComponentName
from util.type import assert_type
from util.tensors import seq_padding


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


class Seq2UMTreeDataset(Dataset[Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    int,
    List[Dict[ComponentName, str]],
]]):
    def __init__(self, hyper: Hyper, dataset: str):
        self.hyper = hyper
        self.data_root = hyper.data_root

        self.word_vocab = hyper.word2id
        self.relation_vocab = hyper.rel2id

        self.tokenizer = self.hyper.tokenizer

        self.text_list: List[List[str]] = []
        self.spo_list: List[List[Dict[ComponentName, str]]] = []

        T: List[List[int]] = []
        R_in: List[int] = []
        S_K1_in: List[int] = []
        S_K2_in: List[int] = []
        O_K1_in: List[int] = []
        O_K2_in: List[int] = []
        S1: List[List[int]] = []
        S2: List[List[int]] = []
        O1: List[List[int]] = []
        O2: List[List[int]] = []
        R_gt: List[List[int]] = []

        oov_token = self.word_vocab["<oov>"]

        with open(os.path.join(self.data_root, dataset), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                instance = assert_type(json.loads(line), dict)

                text = assert_type(instance["text"], str)
                spo_list: List[Dict[ComponentName, str]] = assert_type(instance["spo_list"], list)

                tokens = self.hyper.tokenizer(text)
                text_id = [self.word_vocab.get(c, oov_token) for c in tokens]

                assert len(text_id) > 0

                T.append(text_id)

                # training
                r = assert_type(instance.get("r", -1), int)
                s_k1 = assert_type(instance.get("s_k1", -1), int)
                s_k2 = assert_type(instance.get("s_k2", -1), int)
                o_k1 = assert_type(instance.get("o_k1", -1), int)
                o_k2 = assert_type(instance.get("o_k2", -1), int)

                rel_gt: List[int] = assert_type(instance.get("rel_gt", []), list)
                s1_gt: List[int] = assert_type(instance.get("s1_gt", []), list)
                s2_gt: List[int] = assert_type(instance.get("s2_gt", []), list)
                o1_gt: List[int] = assert_type(instance.get("o1_gt", []), list)
                o2_gt: List[int] = assert_type(instance.get("o2_gt", []), list)

                self.text_list.append(tokens)
                self.spo_list.append(spo_list)

                R_in.append(r)
                S_K1_in.append(s_k1)
                S_K2_in.append(s_k2)
                O_K1_in.append(o_k1)
                O_K2_in.append(o_k2)

                S1.append(s1_gt)
                S2.append(s2_gt)
                O1.append(o1_gt)
                O2.append(o2_gt)
                R_gt.append(rel_gt)

        self.T = np.array(seq_padding(T))

        # self.K1_in, self.K2_in = np.array(self.K1_in), np.array(self.K2_in)
        # only two time step are used for training
        self.R_in = np.array(R_in)
        self.S_K1_in = np.array(S_K1_in)
        self.S_K2_in = np.array(S_K2_in)
        self.O_K1_in = np.array(O_K1_in)
        self.O_K2_in = np.array(O_K2_in)

        # training
        self.S1 = np.array(seq_padding(S1))
        self.S2 = np.array(seq_padding(S2))
        self.O1 = np.array(seq_padding(O1))
        self.O2 = np.array(seq_padding(O2))
        self.R_gt = np.array(R_gt)

    def __getitem__(self, index: int) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[str],
        int,
        List[Dict[ComponentName, str]],
    ]:
        text = self.text_list[index]
        return (
            self.T[index],
            self.S1[index],
            self.S2[index],
            self.O1[index],
            self.O2[index],
            self.R_gt[index],
            self.R_in[index],
            self.S_K1_in[index],
            self.S_K2_in[index],
            self.O_K1_in[index],
            self.O_K2_in[index],
            text,
            len(text),
            self.spo_list[index],
        )

    def __len__(self) -> int:
        return len(self.text_list)


class Seq2UMTreeData:
    def __init__(self, data):
        self.device = torch.device("cpu")

        transposed_data = list(zip(*data))

        lens = transposed_data[12]
        transposed_data, orig_idx = sort_all(transposed_data, lens)

        self.orig_idx = orig_idx

        self.T = torch.from_numpy(np.stack(transposed_data[0])).long()
        self.S1 = torch.from_numpy(np.stack(transposed_data[1])).float()
        self.S2 = torch.from_numpy(np.stack(transposed_data[2])).float()
        self.O1 = torch.from_numpy(np.stack(transposed_data[3])).float()
        self.O2 = torch.from_numpy(np.stack(transposed_data[4])).float()

        self.R_gt = torch.from_numpy(np.stack(transposed_data[5])).float()
        self.R_in = torch.from_numpy(np.stack(transposed_data[6])).long()

        self.S_K1_in = torch.from_numpy(np.stack(transposed_data[7])).long()
        self.S_K2_in = torch.from_numpy(np.stack(transposed_data[8])).long()
        self.O_K1_in = torch.from_numpy(np.stack(transposed_data[9])).long()
        self.O_K2_in = torch.from_numpy(np.stack(transposed_data[10])).long()
        self.text = transposed_data[11]
        self.length = torch.tensor(transposed_data[12]).long()

        self.spo_gold = transposed_data[-1]

    def pin_memory(self):
        self.T = self.T.pin_memory()
        self.S1 = self.S1.pin_memory()
        self.S2 = self.S2.pin_memory()
        self.O1 = self.O1.pin_memory()
        self.O2 = self.O2.pin_memory()

        self.R_gt = self.R_gt.pin_memory()
        self.R_in = self.R_in.pin_memory()

        self.S_K1_in = self.S_K1_in.pin_memory()
        self.S_K2_in = self.S_K2_in.pin_memory()
        self.O_K1_in = self.O_K1_in.pin_memory()
        self.O_K2_in = self.O_K2_in.pin_memory()

        return self

    def to(self, device: torch.device):
        self.T = self.T.to(device=device)
        self.S1 = self.S1.to(device=device)
        self.S2 = self.S2.to(device=device)
        self.O1 = self.O1.to(device=device)
        self.O2 = self.O2.to(device=device)

        self.R_gt = self.R_gt.to(device=device)
        self.R_in = self.R_in.to(device=device)

        self.S_K1_in = self.S_K1_in.to(device=device)
        self.S_K2_in = self.S_K2_in.to(device=device)
        self.O_K1_in = self.O_K1_in.to(device=device)
        self.O_K2_in = self.O_K2_in.to(device=device)

        self.device = device
        return self


class Seq2UMTreeDataLoader(DataLoader[Seq2UMTreeData]):
    def __init__(
            self,
            dataset: Seq2UMTreeDataset,
            batch_size: Optional[int] = 1,
            num_workers: int = 0,
            shuffle: Optional[bool] = None,
    ):
        super().__init__(
            dataset=dataset,
            collate_fn=(lambda data: Seq2UMTreeData(data)),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )
