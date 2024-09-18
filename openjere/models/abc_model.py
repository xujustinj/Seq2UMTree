from abc import ABC, abstractmethod
from typing import Dict

from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from openjere.metrics import F1_triplet

class ABCModel(ABC, nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

        self.metrics = F1_triplet()

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        return self.metrics.get_metric(reset=reset)

    @abstractmethod
    def run_metrics(self, output):
        pass

    @staticmethod
    @abstractmethod
    def description(epoch, epoch_num, output):
        pass

    def set_bert_encoder(self):
        pass


class BERT_encoder(nn.Module):
    def __init__(self, hyper) -> None:
        self.hyper = hyper
        self.model_name = 'bert-base-cased'
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_config(self.model_name)

    def encode(self, sample):
        # begin with text encoding
        pass

class LSTM_encoder(nn.Module):
    def __init__(self, hyper) -> None:
        self.hyper = hyper
        self.model_name = 'bert-base-cased'
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_config(self.model_name)

    def encode(self, sample):
        # begin with text encoding
        pass
