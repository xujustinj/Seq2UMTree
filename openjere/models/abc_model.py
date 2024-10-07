from abc import ABC, abstractmethod
from typing import Dict

from torch import nn

from openjere.metrics import F1Triplet

class ABCModel(ABC, nn.Module):
    def __init__(self):
        super(ABCModel, self).__init__()

        self.metrics = F1Triplet()

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        return self.metrics.get_metric(reset=reset)

    @abstractmethod
    def run_metrics(self, output):
        pass
