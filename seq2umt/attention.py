from typing import Optional

import torch
from torch import nn, Tensor

class CrossAttention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}
    Args:
        dim(int): The number of expected features in the output
    Inputs:
        - **q** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **kv** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **o** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn_logits** (batch, output_len, input_len): tensor containing attention logits.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the outgoing data
    """

    def __init__(self, dim: int):
        super(CrossAttention, self).__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.linear_out = nn.Sequential(
            nn.Linear(in_features=dim*2, out_features=dim),
            nn.Tanh(),
        )

    def forward(
            self,
            q: Tensor,
            kv: Tensor,
            mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        B, Q, H = q.shape
        _, KV, _ = kv.shape
        assert kv.shape == (B, KV, H)

        attn_logits = torch.bmm(q, kv.transpose(1, 2))
        assert attn_logits.shape == (B, Q, KV)
        if mask is not None:
            assert mask.shape == (B, Q, KV)
            attn_logits.data.masked_fill_(mask, -float("inf"))
        weights = self.softmax(attn_logits)

        mix = torch.bmm(weights, kv)
        assert mix.shape == (B, Q, H)

        combined = torch.cat((mix, q), dim=-1)
        assert combined.shape == (B, Q, H+H)

        output = self.linear_out(combined)
        assert isinstance(output, Tensor)
        assert output.shape == (B, Q, H)

        return output, attn_logits
