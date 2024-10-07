from typing import Union

import torch
from torch import Tensor


def seq_max_pool(x: Tensor, mask: Tensor) -> Tensor:
    x = torch.where(mask, x, -torch.inf)
    m, _ = torch.max(x, dim=1)
    return m


def seq_and_vec(seq: Tensor, vec: Tensor) -> Tensor:
    """seq is [None, seq_len, s_size]
    vec is [None, v_size] replicate vec by seq_len times, then concat to seq
    outputs [None, seq_len, s_size+v_size]。
    """
    B, L, S = seq.shape
    _, V = vec.shape
    assert vec.shape == (B, V)
    vec = torch.unsqueeze(vec, dim=1)
    assert vec.shape == (B, 1, V)
    vec = torch.zeros_like(seq[:, :, :1]) + vec
    assert vec.shape == (B, L, V)
    cat = torch.cat((seq, vec), dim=2)
    assert cat.shape == (B, L, S+V)
    return cat


def seq_gather(seq: Tensor, idxs: Union[list[int], Tensor]) -> Tensor:
    """seq is [None, seq_len, s_size]
    idxs is [None, 1], select idxs[i] vec,
    output is [None, s_size]
    """
    B, L, S = seq.shape
    if isinstance(idxs, Tensor):
        idxs = idxs.reshape(B).tolist()
    assert len(idxs) == B

    # TODO: this is faster than the original implementation but we should still
    # use torch.gather
    res = []
    for i, j in enumerate(idxs):
        vec = seq[i,j,:]
        assert vec.shape == (S,)
        res.append(vec)

    res = torch.stack(res, dim=0)
    assert res.shape == (B, S)
    return res


def seq_padding(X: list[list[int]]) -> list[list[int]]:
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]
