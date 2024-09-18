from abc import ABC, abstractmethod

from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

class ABCModel(ABC, nn.Module):
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
