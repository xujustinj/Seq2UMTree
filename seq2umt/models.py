from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from util.tensors import seq_max_pool, seq_and_vec, seq_gather
from .attention import Attention
from .data import Seq2UMTreeData
from .config import Seq2UMTreeConfig
from .loss import MaskedBCE
from .metrics import F1Triplet
from .types import ComponentName


class Seq2UMTree(nn.Module):
    def __init__(self, config: Seq2UMTreeConfig):
        super().__init__()

        self.metrics = F1Triplet()

        self.config = config
        self.order = config.order
        self.data_root = config.data_root

        self.word_vocab = self.config.word2id

        self.mBCE = MaskedBCE()
        self.BCE = nn.BCEWithLogitsLoss()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.sos = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.config.emb_size,
        )

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        return self.metrics.get_metric(reset=reset)

    def run_metrics(self, output: Dict[str, Any]):
        # # whole triplet
        self.metrics(
            output["decode_result"], output["spo_gold"],
        )

    def forward(
            self,
            sample: Seq2UMTreeData,
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "text": list(map(self.config.join, sample.text)),
        }

        t = text_id = sample.T
        B, L = t.shape
        length = sample.length
        assert length.shape == (B,)
        mask = (text_id > 0).unsqueeze(dim=-1)
        assert mask.shape == (B, L, 1)
        mask.requires_grad = False

        head_gt1 = sample.S1
        head_gt2 = sample.S2

        tail_gt1 = sample.O1
        tail_gt2 = sample.O2

        o, h = self.encoder.encode(t, length)

        if self.training:
            t_outs: Tuple[Any, Any, Any] = self.decoder.train_forward(
                sample=sample,
                encoder_o=o,
                h=h,
            )

            out_map = dict(zip(self.order, (0, 1, 2)))

            rel_out = t_outs[out_map["predicate"]]
            head_out1, head_out2 = t_outs[out_map["subject"]]
            tail_out1, tail_out2 = t_outs[out_map["object"]]

            rel_gt = sample.R_gt
            rel_loss = self.BCE(rel_out, rel_gt)
            head_loss_1 = self.mBCE(head_out1, head_gt1, mask)
            head_loss_2 = self.mBCE(head_out2, head_gt2, mask)
            tail_loss_1 = self.mBCE(tail_out1, tail_gt1, mask)
            tail_loss_2 = self.mBCE(tail_out2, tail_gt2, mask)

            output["loss"] = (
                rel_loss
                + head_loss_1
                + head_loss_2
                + tail_loss_1
                + tail_loss_2
            )

        else:
            result = self.decoder.test_forward(
                sample=sample,
                encoder_o=o,
                decoder_h=h,
            )
            output["decode_result"] = result
            output["spo_gold"] = sample.spo_gold

        return output


class Encoder(nn.Module):
    def __init__(self, config: Seq2UMTreeConfig):
        super().__init__()

        lstm_hidden_size = config.hidden_size

        self.embeds = nn.Embedding(
            num_embeddings=len(config.word2id),
            embedding_dim=config.emb_size,
        )
        self.embeds_dropout = nn.Dropout(config.dropout)

        self.bi_lstm = nn.LSTM(
            input_size=config.emb_size,
            hidden_size=(lstm_hidden_size // 2),
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=lstm_hidden_size * 2,
                out_channels=lstm_hidden_size,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self):
        pass

    def encode(
            self,
            t: Tensor,
            length: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, L = t.shape
        mask = (t > 0).unsqueeze(dim=-1)
        mask.requires_grad = False
        assert mask.shape == (B, L, 1)

        t = self.embeds(t)
        t = self.embeds_dropout(t)
        seq = nn.utils.rnn.pack_padded_sequence(t, lengths=length, batch_first=True)

        seq, (h_n, c_n) = self.bi_lstm(seq, None)
        assert isinstance(seq, torch.nn.utils.rnn.PackedSequence)
        assert isinstance(h_n, Tensor)
        assert isinstance(c_n, Tensor)
        t1, _ = nn.utils.rnn.pad_packed_sequence(seq, batch_first=True, total_length=L)

        t_max = seq_max_pool(t1, mask)

        o = seq_and_vec(t1, t_max)

        o = o.permute(0, 2, 1)
        o = self.conv(o)
        assert isinstance(o, Tensor)
        o = o.permute(0, 2, 1)

        h_n = torch.cat((h_n[0], h_n[1]), dim=-1).unsqueeze(0)
        c_n = torch.cat((c_n[0], c_n[1]), dim=-1).unsqueeze(0)
        return o, (h_n, c_n)


class Decoder(nn.Module):
    def __init__(self, config: Seq2UMTreeConfig):
        super().__init__()

        self.config = config
        self.data_root = config.data_root
        self.threshold = config.threshold_logit
        self.word_emb_size = config.emb_size
        self.hidden_size = config.hidden_size

        self.word_vocab = config.word2id

        self.rel_num = len(config.rel2id)
        self.id2word = config.id2word
        self.id2rel = config.id2rel

        self.lstm = nn.LSTM(
            input_size=self.word_emb_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(config.dropout)

        self.attention = Attention(self.word_emb_size)
        self.conv2_to_1_rel = nn.Conv1d(
            in_channels=self.hidden_size * 2,
            out_channels=self.word_emb_size,
            kernel_size=3,
            padding=1,
        )
        self.conv2_to_1_ent = nn.Conv1d(
            in_channels=self.hidden_size * 2,
            out_channels=self.word_emb_size,
            kernel_size=3,
            padding=1,
        )
        self.sos = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.word_emb_size,
        )
        self.rel_emb = nn.Embedding(
            num_embeddings=self.rel_num,
            embedding_dim=self.word_emb_size,
        )

        self.activation = nn.GELU()
        self.rel = nn.Linear(self.word_emb_size, self.rel_num)
        self.ent1 = nn.Linear(self.word_emb_size, 1)
        self.ent2 = nn.Linear(self.word_emb_size, 1)

        # order
        self.order = self.config.order

        self.state_map: Tuple[Callable, Callable, Callable] = {
            ("predicate", "subject", "object"): (
                self.sos2rel,
                self.rel2ent,
                self.ent2ent,
            ),
            ("predicate", "object", "subject"): (
                self.sos2rel,
                self.rel2ent,
                self.ent2ent,
            ),
            ("subject", "object", "predicate"): (
                self.sos2ent,
                self.ent2ent,
                self.ent2rel,
            ),
            ("subject", "predicate", "object"): (
                self.sos2ent,
                self.ent2rel,
                self.rel2ent,
            ),
            ("object", "subject", "predicate"): (
                self.sos2ent,
                self.ent2ent,
                self.ent2rel,
            ),
            ("object", "predicate", "subject"): (
                self.sos2ent,
                self.ent2rel,
                self.rel2ent,
            ),
        }[self.order]

        self.decode_state_map: Tuple[Callable, Callable, Callable] = {
            ("predicate", "subject", "object"): (
                self._out2rel,
                self._out2entity,
                self._out2entity,
            ),
            ("predicate", "object", "subject"): (
                self._out2rel,
                self._out2entity,
                self._out2entity,
            ),
            ("subject", "object", "predicate"): (
                self._out2entity,
                self._out2entity,
                self._out2rel,
            ),
            ("subject", "predicate", "object"): (
                self._out2entity,
                self._out2rel,
                self._out2entity,
            ),
            ("object", "subject", "predicate"): (
                self._out2entity,
                self._out2entity,
                self._out2rel,
            ),
            ("object", "predicate", "subject"): (
                self._out2entity,
                self._out2rel,
                self._out2entity,
            ),
        }[self.order]

    def forward_step(
            self,
            *,
            input_var: Tensor,
            hidden: Tuple[Tensor, Tensor],
            encoder_outputs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        h_n, c_n = hidden
        B, O, V = input_var.shape
        assert h_n.shape == c_n.shape == (1, B, V)
        _, I, _ = encoder_outputs.shape
        assert encoder_outputs.shape == (B, I, V)

        output, (h_n, c_n) = self.lstm(input_var, hidden)
        assert isinstance(output, Tensor)
        assert isinstance(h_n, Tensor)
        assert isinstance(c_n, Tensor)
        assert output.shape == (B, O, V)
        assert h_n.shape == c_n.shape == (1, B, V)

        output, attn = self.attention(output, encoder_outputs)
        assert isinstance(output, Tensor)
        assert isinstance(attn, Tensor)
        assert output.shape == (B, O, V)
        assert attn.shape == (B, O, I)

        return output, attn, (h_n, c_n)

    def to_rel(
            self,
            input: Tensor,
            h: Tuple[Tensor, Tensor],
            encoder_o: Tensor,
            mask: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        output, attn, h = self.forward_step(
            input_var=input,
            hidden=h,
            encoder_outputs=encoder_o,
        )
        new_encoder_o: Tensor = seq_and_vec(encoder_o, output.squeeze(1))

        new_encoder_o = new_encoder_o.permute(0, 2, 1)
        new_encoder_o = self.conv2_to_1_rel(new_encoder_o)
        new_encoder_o = new_encoder_o.permute(0, 2, 1)

        output = self.dropout(new_encoder_o)
        output = self.activation(output)
        output = self.rel(output)
        output = seq_max_pool(output, mask)

        return output, h, new_encoder_o

    def to_ent(
            self,
            input: Tensor,
            h: Tuple[Tensor, Tensor],
            encoder_o: Tensor,
            mask: Tensor,
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor]:
        output, attn, h = self.forward_step(
            input_var=input,
            hidden=h,
            encoder_outputs=encoder_o,
        )
        output = output.squeeze(1)

        new_encoder_o = seq_and_vec(encoder_o, output)

        new_encoder_o = new_encoder_o.permute(0, 2, 1)
        new_encoder_o: Tensor = self.conv2_to_1_ent(new_encoder_o)
        new_encoder_o = new_encoder_o.permute(0, 2, 1)

        output = self.dropout(new_encoder_o)
        output = self.activation(output)

        ent1: Tensor = self.ent1(output)
        ent2: Tensor = self.ent2(output)
        output = ent1.squeeze(dim=2), ent2.squeeze(dim=2)

        return output, h, new_encoder_o

    def sos2ent(
            self,
            sos: Tensor,
            encoder_o: Tensor,
            h: Tuple[Tensor, Tensor],
            mask: Tensor,
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor]:
        out, h, new_encoder_o = self.to_ent(sos, h, encoder_o, mask)
        return out, h, new_encoder_o

    def sos2rel(
            self,
            sos: Tensor,
            encoder_o: Tensor,
            h: Tuple[Tensor, Tensor],
            mask: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        out, h, new_encoder_o = self.to_rel(sos, h, encoder_o, mask)
        out = out.squeeze(dim=1)
        return out, h, new_encoder_o

    def ent2ent(
            self,
            ent: Tuple[Union[List[int], Tensor], Union[List[int], Tensor]],
            encoder_o: Tensor,
            h: Tuple[Tensor, Tensor],
            mask: Tensor,
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor]:
        k1, k2 = ent
        k1 = seq_gather(encoder_o, k1)
        k2 = seq_gather(encoder_o, k2)

        input = k1 + k2
        input = input.unsqueeze(1)
        t3_out, h, new_encoder_o = self.to_ent(input, h, encoder_o, mask)
        return t3_out, h, new_encoder_o

    def ent2rel(
            self,
            ent: Tuple[Union[List[int], Tensor], Union[List[int], Tensor]],
            encoder_o: Tensor,
            h: Tuple[Tensor, Tensor],
            mask: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        k1, k2 = ent
        k1 = seq_gather(encoder_o, k1)
        k2 = seq_gather(encoder_o, k2)

        input = k1 + k2
        input = input.unsqueeze(1)
        out, h, new_encoder_o = self.to_rel(input, h, encoder_o, mask)
        out = out.squeeze(1)

        return out, h, new_encoder_o

    def rel2ent(
            self,
            rel: Tensor,
            encoder_o: Tensor,
            h: Tuple[Tensor, Tensor],
            mask: Tensor,
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor]:
        input = self.rel_emb(rel)
        assert isinstance(input, Tensor)
        input = input.unsqueeze(dim=1)
        out, h, new_encoder_o = self.to_ent(input, h, encoder_o, mask)
        return out, h, new_encoder_o

    def train_forward(
            self,
            *,
            sample: Seq2UMTreeData,
            encoder_o: Tensor,
            h: Tuple[Tensor, Tensor],
    ) -> Union[
        Tuple[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]],
        Tuple[Tuple[Tensor, Tensor], Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor],
    ]:
        B, L = sample.T.shape
        embed: Tensor = self.sos(torch.tensor(0, device=sample.device))
        t1_in: Tensor = embed.unsqueeze(dim=0).expand(B, -1).unsqueeze(dim=1)

        r_in = sample.R_in
        s_in = (
            sample.S_K1_in.unsqueeze(dim=-1),
            sample.S_K2_in.unsqueeze(dim=-1),
        )
        o_in = (
            sample.O_K1_in.unsqueeze(dim=-1),
            sample.O_K2_in.unsqueeze(dim=-1),
        )

        in_tuple = (r_in, s_in, o_in)
        in_map = {"predicate": 0, "subject": 1, "object": 2}
        t2_in = in_tuple[in_map[self.order[0]]]
        t3_in = in_tuple[in_map[self.order[1]]]

        t = sample.T
        mask = (t > 0).unsqueeze(dim=-1)
        mask.requires_grad = False
        assert mask.shape == (B, L, 1)

        t1_out, h, new_encoder_o = self.state_map[0](t1_in, encoder_o, h, mask)
        t2_out, h, new_encoder_o = self.state_map[1](t2_in, new_encoder_o, h, mask)
        t3_out, h, new_encoder_o = self.state_map[2](t3_in, new_encoder_o, h, mask)

        return t1_out, t2_out, t3_out

    def test_forward(
            self,
            *,
            sample: Seq2UMTreeData,
            encoder_o: Tensor,
            decoder_h: Tuple[Tensor, Tensor],
    ) -> List[List[Dict[ComponentName, str]]]:
        t = sample.T
        B, L = t.shape
        mask = (t > 0).unsqueeze(dim=-1)
        mask.requires_grad = False
        assert mask.shape == (B, L, 1)
        text = [[self.id2word[c] for c in sent] for sent in t.tolist()]
        result: List[List[Dict[ComponentName, str]]] = []
        for i, sent in enumerate(text):
            h, c = (
                decoder_h[0][:,i,:].unsqueeze(dim=1).contiguous(),
                decoder_h[1][:,i,:].unsqueeze(dim=1).contiguous(),
            )
            triplets = self.extract_items(
                sent=sent,
                mask=mask[i,:].unsqueeze(dim=0).contiguous(),
                encoder_o=encoder_o[i,:,:].unsqueeze(dim=0).contiguous(),
                t1_h=(h, c),
            )
            result.append(triplets)
        return result

    def _out2entity(
            self,
            sent: List[str],
            out: Tuple[Tensor, Tensor],
    ) -> List[Tuple[Tuple[int, int], str]]:
        out1, out2 = out
        entities: List[Tuple[Tuple[int, int], str]] = []
        pred1 = (out1.squeeze() > self.threshold)
        pred2 = (out2.squeeze() > self.threshold)
        for (start,) in pred1.nonzero().tolist():
            for (length,) in pred2[start:].nonzero().tolist():
                end = start + length
                entities.append((
                    (start, end),
                    self.config.join(sent[start:end+1]),
                ))
                break
        return entities

    def _out2rel(
            self,
            sent: List[str],
            out: Tensor,
    ) -> List[Tuple[int, str]]:
        pred = (out.squeeze() > self.threshold)
        return [(i, self.id2rel[i]) for (i,) in pred.nonzero().tolist()]

    def _out2in(self, out, component: ComponentName, device: torch.device):
        if component == "predicate":
            assert isinstance(out, int)
            return torch.tensor([out], dtype=torch.long, device=device)
        else:
            s1, s2 = out
            assert isinstance(s1, int)
            assert isinstance(s2, int)
            return (
                torch.tensor([s1], dtype=torch.long, device=device),
                torch.tensor([s2], dtype=torch.long, device=device),
            )

    def extract_items(
        self,
        *,
        sent: List[str],
        mask: Tensor,
        encoder_o: Tensor,
        t1_h: Tuple[Tensor, Tensor],
    ) -> List[Dict[ComponentName, str]]:
        acc: List[Dict[ComponentName, str]] = []
        device = encoder_o.device

        embed: Tensor = self.sos(torch.tensor(0, device=device))
        sos = embed.unsqueeze(dim=0).unsqueeze(dim=1)

        t1_out, t1_h, t1_encoder_o = self.state_map[0](sos, encoder_o, t1_h, mask)

        t1_decode = self.decode_state_map[0](sent, t1_out)

        for id1, name1 in t1_decode:
            t2_in = self._out2in(id1, component=self.order[0], device=device)
            t2_out, t2_h, t2_encoder_o = self.state_map[1](t2_in, t1_encoder_o, t1_h, mask)
            t2_decode = self.decode_state_map[1](sent, t2_out)

            for id2, name2 in t2_decode:
                t3_in = self._out2in(id2, component=self.order[1], device=device)
                t3_out, _, _ = self.state_map[2](t3_in, t2_encoder_o, t2_h, mask)
                t3_decode = self.decode_state_map[2](sent, t3_out)

                for _, name3 in t3_decode:
                    acc.append(
                        {
                            self.order[0]: name1,
                            self.order[1]: name2,
                            self.order[2]: name3,
                        }
                    )

        return acc
