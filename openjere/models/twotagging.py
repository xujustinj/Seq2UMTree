# https://github.com/zhengyima/kg-baseline-pytorch/tree/master
from functools import partial
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from openjere.config import Hyper, seq_max_pool, seq_and_vec, seq_gather
from openjere.models.abc_model import ABCModel


class Twotagging(ABCModel):
    def __init__(self, hyper: Hyper) -> None:
        super(Twotagging, self).__init__()
        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu

        with open(os.path.join(self.data_root, "word_vocab.json"), "r") as f:
            self.word_vocab: Dict[str, int] = json.load(f)
        assert isinstance(self.word_vocab, dict)
        with open(os.path.join(self.data_root, "relation_vocab.json"), "r") as f:
            self.relation_vocab: Dict[str, int] = json.load(f)
        assert isinstance(self.relation_vocab, dict)
        self.id2word = {v: k for k, v in self.word_vocab.items()}
        self.id2rel = {v: k for k, v in self.relation_vocab.items()}

        self.S = s_model(
            word_dict_length=len(self.word_vocab),
            word_emb_size=self.hyper.emb_size,
        )
        self.PO = po_model(
            word_emb_size=self.hyper.emb_size,
            num_classes=len(self.relation_vocab),
        )  # 49
        self.CE = torch.nn.CrossEntropyLoss()
        self.BCE = torch.nn.BCEWithLogitsLoss()

    def masked_BCEloss(self, logits, gt, mask):
        loss = self.BCE(logits, gt)
        loss = torch.sum(loss.mul(mask)) / torch.sum(mask)
        return loss

    def inference(self, text_id) -> List[List[Dict[str, str]]]:
        text = text_id.tolist()
        text = [[self.id2word[c] for c in sent] for sent in text]
        result = []
        for i, sent in enumerate(text):
            triplets = self.extract_items(sent, text_id[i, :].unsqueeze(0).contiguous())
            result.append(triplets)
        return result

    def extract_items(self, sent, text_id) -> List[Dict[str, str]]:
        R = []
        _k1, _k2, t, t_max, mask = self.S(text_id)

        _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]

        _kk1s = []
        for i, _kk1 in enumerate(_k1):
            if _kk1 > 0.5:
                _subject = ""
                for j, _kk2 in enumerate(_k2[i:]):
                    if _kk2 > 0.5:
                        _subject = self.hyper.join(sent[i : i + j + 1])
                        break
                if _subject:
                    _k1, _k2 = (
                        torch.LongTensor([[i]]),
                        torch.LongTensor([[i + j]]),
                    )  # np.array([i]), np.array([i+j])
                    _o1, _o2 = self.PO(t.cuda(), t_max.cuda(), _k1.cuda(), _k2.cuda())
                    _o1, _o2 = _o1.cpu().data.numpy(), _o2.cpu().data.numpy()

                    _o1, _o2 = np.argmax(_o1[0], 1), np.argmax(_o2[0], 1)

                    for i, _oo1 in enumerate(_o1):
                        if _oo1 > 0:
                            for j, _oo2 in enumerate(_o2[i:]):
                                if _oo2 == _oo1:
                                    _object = self.hyper.join(sent[i : i + j + 1])
                                    _predicate = self.id2rel[_oo1]
                                    R.append(
                                        {
                                            "subject": _subject,
                                            "predicate": _predicate,
                                            "object": _object,
                                        }
                                    )
                                    break
            _kk1s.append(_kk1.data.cpu().numpy())
        _kk1s = np.array(_kk1s)
        return R

    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        output = {}

        t = text_id = sample.T.cuda(self.gpu)
        ps_1, ps_2, t, t_max, mask = self.S(t)

        if is_train:
            # teacher forcing
            k1 = sample.K1.cuda(self.gpu)
            k2 = sample.K2.cuda(self.gpu)
            s1 = sample.S1.cuda(self.gpu)
            s2 = sample.S2.cuda(self.gpu)
            o1 = sample.O1.cuda(self.gpu)
            o2 = sample.O2.cuda(self.gpu)

            po_1, po_2 = self.PO(t, t_max, k1, k2)

            po_1 = po_1.permute(0, 2, 1)
            po_2 = po_2.permute(0, 2, 1)

            s1 = torch.unsqueeze(s1, 2)
            s2 = torch.unsqueeze(s2, 2)

            s1_loss = self.masked_BCEloss(ps_1, s1, mask)
            s2_loss = self.masked_BCEloss(ps_2, s2, mask)

            o1_loss = self.CE(po_1, o1)
            o1_loss = torch.sum(o1_loss.mul(mask[:, :, 0])) / torch.sum(mask)
            o2_loss = self.CE(po_2, o2)
            o2_loss = torch.sum(o2_loss.mul(mask[:, :, 0])) / torch.sum(mask)

            loss_sum = 2.5 * (s1_loss + s2_loss) + (o1_loss + o2_loss)

            output["loss"] = loss_sum
        else:
            output["decode_result"] = self.inference(text_id)
            output["spo_gold"] = sample.spo_gold

        output["description"] = partial(self.description, output=output)
        return output

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, epoch: {}/{}:".format(
            output["loss"].item(), epoch, epoch_num,
        )

    def run_metrics(self, output):
        self.metrics(output["decode_result"], output["spo_gold"])


class s_model(nn.Module):
    def __init__(self, word_dict_length, word_emb_size):
        super(s_model, self).__init__()

        self.embeds = nn.Embedding(word_dict_length, word_emb_size).cuda()
        self.fc1_dropout = nn.Dropout(0.25)

        self.lstm1 = nn.LSTM(
            input_size=word_emb_size,
            hidden_size=int(word_emb_size / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        ).cuda()

        self.lstm2 = nn.LSTM(
            input_size=word_emb_size,
            hidden_size=int(word_emb_size / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        ).cuda()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size * 2,  # 输入的深度
                out_channels=word_emb_size,  # filter 的个数，输出的高度
                kernel_size=3,  # filter的长与宽
                stride=1,  # 每隔多少步跳一下
                padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ).cuda(),
            nn.ReLU(),
        ).cuda()
        self.fc_ps1 = nn.Linear(word_emb_size, 1).cuda()

        self.fc_ps2 = nn.Linear(word_emb_size, 1).cuda()

    def forward(self, t: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        B, L = t.shape
        mask = (t > 0).float().unsqueeze(dim=-1)
        mask.requires_grad = False
        assert mask.shape == (B, L, 1)
        t = self.embeds(t)

        t = self.fc1_dropout(t)

        t = t.mul(mask)  # (batch_size,sent_len,char_size)

        t, _ = self.lstm1(t, None)
        t, _ = self.lstm2(t, None)

        t_max = seq_max_pool(t, mask)

        h: torch.Tensor = seq_and_vec([t, t_max])
        h = h.permute(0, 2, 1)
        h = self.conv1(h)
        h = h.permute(0, 2, 1)

        ps1: torch.Tensor = self.fc_ps1(h)
        ps2: torch.Tensor = self.fc_ps2(h)

        return ps1, ps2, t, t_max, mask


class po_model(nn.Module):
    def __init__(self, word_emb_size, num_classes):
        super(po_model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size * 4,  # 输入的深度
                out_channels=word_emb_size,  # filter 的个数，输出的高度
                kernel_size=3,  # filter的长与宽
                stride=1,  # 每隔多少步跳一下
                padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ).cuda(),
            nn.ReLU(),
        ).cuda()

        self.fc_ps1 = nn.Linear(word_emb_size, num_classes + 1).cuda()
        self.fc_ps2 = nn.Linear(word_emb_size, num_classes + 1).cuda()

    def forward(self, t, t_max, k1, k2):

        k1 = seq_gather([t, k1])

        k2 = seq_gather([t, k2])

        k = torch.cat([k1, k2], 1)
        h = seq_and_vec([t, t_max])
        h = seq_and_vec([h, k])
        h = h.permute(0, 2, 1)
        h = self.conv1(h)
        h = h.permute(0, 2, 1)

        po1 = self.fc_ps1(h)
        po2 = self.fc_ps2(h)

        return [po1.cuda(), po2.cuda()]
