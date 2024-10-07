from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")
    return torch.device(device)
