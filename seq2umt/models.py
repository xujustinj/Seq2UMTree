from typing import Any, Callable, Union

import torch
from torch import Tensor
import torch.nn as nn

from util.tensors import seq_max_pool, seq_and_vec, seq_gather
from .attention import CrossAttention
from .data import SchemaData, Seq2UMTreeData
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
        self.hidden_size = config.hidden_size

        self.word_vocab = self.config.word2id

        self.mBCE = MaskedBCE()
        self.BCE = nn.BCEWithLogitsLoss()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.sos = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.config.emb_size,
        )

    def get_metric(self, reset: bool = False) -> dict[str, float]:
        return self.metrics.get_metric(reset=reset)

    def run_metrics(self, output: dict[str, Any]):
        # # whole triplet
        self.metrics(
            output["decode_result"], output["spo_gold"],
        )

    def forward(
            self,
            sample: Seq2UMTreeData,
            schema_data: SchemaData,
    ) -> dict[str, Any]:
        H = self.hidden_size

        output: dict[str, Any] = {
            "text": [self.config.join(t) for t in sample.text],
        }

        token_ids = sample.T
        B, L = token_ids.shape
        length = sample.length
        assert length.shape == (B,)
        mask = (token_ids > 0).unsqueeze(dim=-1)
        assert mask.shape == (B, L, 1)
        mask.requires_grad = False

        head_gt1 = sample.S1
        head_gt2 = sample.S2

        tail_gt1 = sample.O1
        tail_gt2 = sample.O2

        o, h = self.encoder.encode(token_ids=token_ids, lengths=length)
        assert o.shape == (B, L, H)
        assert h[0].shape == h[1].shape == (1, B, H)

        schema_t = schema_data.tokens
        schema_B, schema_L = schema_t.shape
        schema_length = schema_data.length
        assert schema_length.shape == (schema_B,)
        schema_mask = (schema_t > 0).unsqueeze(dim=-1)
        assert schema_mask.shape == (schema_B, schema_L, 1)
        schema_mask.requires_grad = False

        schema_o, _ = self.encoder.encode(token_ids=schema_t, lengths=schema_length)
        assert schema_o.shape == (schema_B, schema_L, H)
        schema_embeds = seq_max_pool(schema_o, schema_mask)
        assert schema_embeds.shape == (schema_B, H)
        schema_embeds[schema_data.orig_idx,:] = schema_embeds.clone()

        if self.training:
            t_outs: tuple[Any, Any, Any] = self.decoder.train_forward(
                sample=sample,
                encoder_o=o,
                h=h,
                relation_embeds=schema_embeds,
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
                relation_embeds=schema_embeds,
            )
            output["decode_result"] = result
            output["spo_gold"] = sample.spo_gold

        return output


class Encoder(nn.Module):
    def __init__(self, config: Seq2UMTreeConfig):
        super().__init__()

        E = self.embed_size = config.emb_size
        H = self.hidden_size = config.hidden_size
        assert H % 2 == 0

        self.embeds = nn.Embedding(
            num_embeddings=len(config.word2id),
            embedding_dim=E,
        )
        self.embeds_dropout = nn.Dropout(config.dropout)

        self.bi_lstm = nn.LSTM(
            input_size=E,
            hidden_size=(H // 2),
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=(H * 2),
                out_channels=H,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self):
        pass

    def encode(
            self,
            *,
            token_ids: Tensor,
            lengths: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Parameters:
            token_ids: [B x L]
            lengths: [B]

        Returns:
            output: [B x L x H]
            hidden: tuple
                h_n: [1 x B x H]
                c_n: [1 x B x H]

        The h_n, c_n output of the 2-layer BiLSTM has shape (2*2, B, H//2).
        Meanwhile, the decoder uses a 1-layer LSTM which expects initial embeds
        of (1, B, H).
        This makes the output shape of the encoder make a bit more sense.
        """

        E = self.embed_size
        H = self.hidden_size

        B, L = token_ids.shape
        assert lengths.shape == (B,)
        assert (0 < lengths).all()
        assert (lengths <= L).all()

        mask = (token_ids > 0).unsqueeze(dim=-1)
        mask.requires_grad = False
        assert mask.shape == (B, L, 1)

        x: Tensor = self.embeds(token_ids)
        assert x.shape == (B, L, E)
        x = self.embeds_dropout(x)
        assert x.shape == (B, L, E)

        seq = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True)
        seq, (h_n, c_n) = self.bi_lstm(seq, None)
        assert isinstance(seq, torch.nn.utils.rnn.PackedSequence)
        assert isinstance(h_n, Tensor)
        assert isinstance(c_n, Tensor)
        assert h_n.shape == c_n.shape == (4, B, H//2)

        x, _ = nn.utils.rnn.pad_packed_sequence(seq, batch_first=True, total_length=L)
        assert isinstance(x, Tensor)
        assert x.shape == (B, L, H)

        x_max = seq_max_pool(x, mask)
        assert x_max.shape == (B, H)

        x = seq_and_vec(x, x_max)
        assert x.shape == (B, L, 2*H)

        x = x.permute(0, 2, 1)
        assert x.shape == (B, 2*H, L)
        x = self.conv(x)
        assert x.shape == (B, H, L)
        x = x.permute(0, 2, 1)
        assert x.shape == (B, L, H)

        h_n = torch.cat((h_n[0,:,:], h_n[1,:,:]), dim=-1).unsqueeze(0)
        c_n = torch.cat((c_n[0,:,:], c_n[1,:,:]), dim=-1).unsqueeze(0)
        assert h_n.shape == c_n.shape == (1, B, H)
        return x, (h_n, c_n)


class Decoder(nn.Module):
    def __init__(self, config: Seq2UMTreeConfig):
        super().__init__()

        self.config = config
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

        self.attention = CrossAttention(self.word_emb_size)
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

        self.state_map: tuple[Callable, Callable, Callable] = {
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

        self.decode_state_map: tuple[Callable, Callable, Callable] = {
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
            states: Tensor,
            hidden: tuple[Tensor, Tensor],
            encoder_outputs: Tensor,
            relation_embeds: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Parameters:
            states: [B x E]
            hidden: tuple
                h_n: [1 x B x E]
                c_n: [1 x B x E]
            encoder_outputs: [B x I x E]
            relation_embeds: [B x R x E]

        Returns:
            output: []
            attn: tuple
                relation_attn: [B x R]
                entity_attn: [B x L]
            hidden: tuple
                h_n: [1 x B x E]
                c_n: [1 x B x E]
        """
        E = self.word_emb_size
        h_n, c_n = hidden
        B, _ = states.shape
        assert states.shape == (B, E)
        assert h_n.shape == c_n.shape == (1, B, E)
        _, L, _ = encoder_outputs.shape
        assert encoder_outputs.shape == (B, L, E)

        next_states, (h_n, c_n) = self.lstm(states.unsqueeze(dim=1), (h_n, c_n))
        assert isinstance(next_states, Tensor)
        assert isinstance(h_n, Tensor)
        assert isinstance(c_n, Tensor)
        assert next_states.shape == (B, 1, E)
        assert h_n.shape == c_n.shape == (1, B, E)

        R, _ = relation_embeds.shape
        assert relation_embeds.shape == (R, E)
        relation_embeds = relation_embeds.tile(B, 1, 1)
        assert relation_embeds.shape == (B, R, E)

        joint = torch.cat((relation_embeds, encoder_outputs), dim=1)
        assert joint.shape == (B, R+L, E)

        next_states, attn = self.attention(next_states, joint)
        assert isinstance(next_states, Tensor)
        assert isinstance(attn, Tensor)
        assert next_states.shape == (B, 1, E)
        assert attn.shape == (B, 1, R+L)
        next_states = next_states.squeeze(dim=1)
        attn = attn.squeeze(dim=1)

        relation_attn = attn[:,:R]
        entity_attn = attn[:,R:]
        assert relation_attn.shape == (B, R)
        assert entity_attn.shape == (B, L)

        return next_states, (relation_attn, entity_attn), (h_n, c_n)

    def to_rel(
            self,
            states: Tensor,
            h: tuple[Tensor, Tensor],
            encoder_o: Tensor,
            mask: Tensor,
            relation_embeds: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        output, (relation_attn, _), h = self.forward_step(
            states=states,
            hidden=h,
            encoder_outputs=encoder_o,
            relation_embeds=relation_embeds,
        )

        new_encoder_o: Tensor = seq_and_vec(encoder_o, output.squeeze(1))
        new_encoder_o = new_encoder_o.permute(0, 2, 1)
        new_encoder_o = self.conv2_to_1_rel(new_encoder_o)
        new_encoder_o = new_encoder_o.permute(0, 2, 1)

        B, R = relation_attn.shape
        return relation_attn, h, new_encoder_o

    def to_ent(
            self,
            states: Tensor,
            h: tuple[Tensor, Tensor],
            encoder_o: Tensor,
            mask: Tensor,
            relation_embeds: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor]:
        output, attn, h = self.forward_step(
            states=states,
            hidden=h,
            encoder_outputs=encoder_o,
            relation_embeds=relation_embeds,
        )

        new_encoder_o: Tensor = seq_and_vec(encoder_o, output.squeeze(1))
        new_encoder_o = new_encoder_o.permute(0, 2, 1)
        new_encoder_o = self.conv2_to_1_ent(new_encoder_o)
        new_encoder_o = new_encoder_o.permute(0, 2, 1)

        output = self.dropout(new_encoder_o)
        output = self.activation(output)

        ent1: Tensor = self.ent1(output)
        ent2: Tensor = self.ent2(output)
        output = ent1.squeeze(dim=2), ent2.squeeze(dim=2)

        return output, h, new_encoder_o

    def sos2ent(
            self,
            states: Tensor,
            encoder_o: Tensor,
            h: tuple[Tensor, Tensor],
            mask: Tensor,
            relation_embeds: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor]:
        out, h, new_encoder_o = self.to_ent(states, h, encoder_o, mask, relation_embeds)
        return out, h, new_encoder_o

    def sos2rel(
            self,
            states: Tensor,
            encoder_o: Tensor,
            h: tuple[Tensor, Tensor],
            mask: Tensor,
            relation_embeds: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        out, h, new_encoder_o = self.to_rel(states, h, encoder_o, mask, relation_embeds)
        out = out.squeeze(dim=1)
        return out, h, new_encoder_o

    def ent2ent(
            self,
            ent: tuple[Union[list[int], Tensor], Union[list[int], Tensor]],
            encoder_o: Tensor,
            h: tuple[Tensor, Tensor],
            mask: Tensor,
            relation_embeds: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor]:
        k1, k2 = ent
        k1 = seq_gather(encoder_o, k1)
        k2 = seq_gather(encoder_o, k2)

        input = k1 + k2
        t3_out, h, new_encoder_o = self.to_ent(input, h, encoder_o, mask, relation_embeds)
        return t3_out, h, new_encoder_o

    def ent2rel(
            self,
            ent: tuple[Union[list[int], Tensor], Union[list[int], Tensor]],
            encoder_o: Tensor,
            h: tuple[Tensor, Tensor],
            mask: Tensor,
            relation_embeds: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        k1, k2 = ent
        k1 = seq_gather(encoder_o, k1)
        k2 = seq_gather(encoder_o, k2)

        input = k1 + k2
        out, h, new_encoder_o = self.to_rel(input, h, encoder_o, mask, relation_embeds)
        out = out.squeeze(1)

        return out, h, new_encoder_o

    def rel2ent(
            self,
            rel: Tensor,
            encoder_o: Tensor,
            h: tuple[Tensor, Tensor],
            mask: Tensor,
            relation_embeds: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor]:
        B, = rel.shape
        R, E = relation_embeds.shape
        input = relation_embeds[rel,:]
        assert input.shape == (B, E)
        out, h, new_encoder_o = self.to_ent(input, h, encoder_o, mask, relation_embeds)
        return out, h, new_encoder_o

    def train_forward(
            self,
            *,
            sample: Seq2UMTreeData,
            encoder_o: Tensor,
            h: tuple[Tensor, Tensor],
            relation_embeds: Tensor,
    ) -> Union[
        tuple[Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        tuple[tuple[Tensor, Tensor], Tensor, tuple[Tensor, Tensor]],
        tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor],
    ]:
        B, L = sample.T.shape
        E = self.word_emb_size
        state: Tensor = self.sos(torch.tensor(0, device=sample.device))
        assert state.shape == (E,)
        t1_in: Tensor = state.unsqueeze(dim=0).expand(B, -1)
        assert t1_in.shape == (B, E)

        R, _ = relation_embeds.shape
        assert relation_embeds.shape == (R, E)

        r_in = sample.R_in
        assert r_in.shape == (B,)

        s_in = (
            sample.S_K1_in.unsqueeze(dim=-1),
            sample.S_K2_in.unsqueeze(dim=-1),
        )
        assert s_in[0].shape == s_in[1].shape == (B, 1)

        o_in = (
            sample.O_K1_in.unsqueeze(dim=-1),
            sample.O_K2_in.unsqueeze(dim=-1),
        )
        assert o_in[0].shape == o_in[1].shape == (B, 1)

        in_tuple = (r_in, s_in, o_in)
        in_map = {"predicate": 0, "subject": 1, "object": 2}
        t2_in = in_tuple[in_map[self.order[0]]]
        t3_in = in_tuple[in_map[self.order[1]]]

        t = sample.T
        mask = (t > 0).unsqueeze(dim=-1)
        mask.requires_grad = False
        assert mask.shape == (B, L, 1)

        t1_out, h, new_encoder_o = self.state_map[0](t1_in, encoder_o, h, mask, relation_embeds)
        t2_out, h, new_encoder_o = self.state_map[1](t2_in, new_encoder_o, h, mask, relation_embeds)
        t3_out, h, new_encoder_o = self.state_map[2](t3_in, new_encoder_o, h, mask, relation_embeds)

        return t1_out, t2_out, t3_out

    def test_forward(
            self,
            *,
            sample: Seq2UMTreeData,
            encoder_o: Tensor,
            decoder_h: tuple[Tensor, Tensor],
            relation_embeds: Tensor,
    ) -> list[list[dict[ComponentName, str]]]:
        t = sample.T
        B, L = t.shape
        mask = (t > 0).unsqueeze(dim=-1)
        mask.requires_grad = False
        assert mask.shape == (B, L, 1)
        text = [[self.id2word[c] for c in sent] for sent in t.tolist()]
        result: list[list[dict[ComponentName, str]]] = []
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
                relation_embeds=relation_embeds,
            )
            result.append(triplets)
        return result

    def _out2entity(
            self,
            sent: list[str],
            out: tuple[Tensor, Tensor],
    ) -> list[tuple[tuple[int, int], str]]:
        out1, out2 = out
        entities: list[tuple[tuple[int, int], str]] = []
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
            sent: list[str],
            out: Tensor,
    ) -> list[tuple[int, str]]:
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
        sent: list[str],
        mask: Tensor,
        encoder_o: Tensor,
        t1_h: tuple[Tensor, Tensor],
        relation_embeds: Tensor,
    ) -> list[dict[ComponentName, str]]:
        acc: list[dict[ComponentName, str]] = []
        device = encoder_o.device

        embed: Tensor = self.sos(torch.tensor(0, device=device))
        sos = embed.unsqueeze(dim=0)

        t1_out, t1_h, t1_encoder_o = self.state_map[0](sos, encoder_o, t1_h, mask, relation_embeds)

        t1_decode = self.decode_state_map[0](sent, t1_out)

        for id1, name1 in t1_decode:
            t2_in = self._out2in(id1, component=self.order[0], device=device)
            t2_out, t2_h, t2_encoder_o = self.state_map[1](t2_in, t1_encoder_o, t1_h, mask, relation_embeds)
            t2_decode = self.decode_state_map[1](sent, t2_out)

            for id2, name2 in t2_decode:
                t3_in = self._out2in(id2, component=self.order[1], device=device)
                t3_out, _, _ = self.state_map[2](t3_in, t2_encoder_o, t2_h, mask, relation_embeds)
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
