import argparse
from collections import Counter
import logging
import os
from typing import Tuple, Type

import numpy as np
import torch
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
import torch.utils
import torch.utils.data
from tqdm import tqdm

from openjere.config import Hyper, ModelName, OptimizerName
from openjere.dataloaders import (
    Abstract_dataset,
    PartialDataLoader,
    Selection_Dataset,
    Selection_loader,
    Twotagging_Dataset,
    Twotagging_loader,
    Seq2umt_Dataset,
    Seq2umt_loader,
    WDec_Dataset,
    WDec_loader,
    Copymtl_Dataset,
    Copymtl_loader,
)
from openjere.models import (
    ABCModel,
    MultiHeadSelection,
    Twotagging,
    Seq2umt,
    WDec,
    CopyMTL,
)
from openjere.preprocessings import (
    ABC_data_preprocessing,
    Selection_preprocessing,
    Twotagging_preprocessing,
    Seq2umt_preprocessing,
    WDec_preprocessing,
    Copymtl_preprocessing,
)


class Runner(object):
    model: ABCModel
    optimizer: Optimizer

    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.model_dir = "saved_models"

        self.hyper = Hyper(os.path.join("experiments", self.exp_name + ".json"))

        self.gpu = self.hyper.gpu
        self.preprocessor = self._preprocessor()

        self.Dataset, self.Loader = self._init_loader()

        logging.basicConfig(
            filename=os.path.join("experiments", self.exp_name + ".log"),
            filemode="w",
            format="%(asctime)s - %(message)s",
            level=logging.INFO,
        )

    def _init_loader(self) -> Tuple[Type[Abstract_dataset], PartialDataLoader]:
        name: ModelName = self.hyper.model
        if name == "selection":
            return Selection_Dataset, Selection_loader
        elif name == "twotagging":
            return Twotagging_Dataset, Twotagging_loader
        elif name == "seq2umt":
            return Seq2umt_Dataset, Seq2umt_loader
        elif name == "wdec":
            return WDec_Dataset, WDec_loader
        elif name == "copymtl":
            return Copymtl_Dataset, Copymtl_loader

    def _init_optimizer(self):
        name: OptimizerName = self.hyper.optimizer
        if name == "adam":
            self.optimizer = Adam(self.model.parameters())
        elif name == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=0.5)

    def _preprocessor(self) -> ABC_data_preprocessing:
        name: ModelName = self.hyper.model
        if name == "selection":
            return Selection_preprocessing(self.hyper)
        elif name == "twotagging":
            return Twotagging_preprocessing(self.hyper)
        elif name == "seq2umt":
            return Seq2umt_preprocessing(self.hyper)
        elif name == "wdec":
            return WDec_preprocessing(self.hyper)
        elif name == "copymtl":
            return Copymtl_preprocessing(self.hyper)

    def _init_model(self):
        name = self.hyper.model
        logging.info(name)

        if name == "selection":
            self.model = MultiHeadSelection(self.hyper)
        elif name == "twotagging":
            self.model = Twotagging(self.hyper)
        elif name == "seq2umt":
            self.model = Seq2umt(self.hyper)
        elif name == "wdec":
            self.model = WDec(self.hyper)
        elif name == "copymtl":
            self.model = CopyMTL(self.hyper)
        self.model.cuda(self.gpu)

    def preprocessing(self):
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=2)
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self, mode: str):
        if mode == "preprocessing":
            self.preprocessing()

        elif mode == "train":
            self.hyper.vocab_init()
            self._init_model()
            self._init_optimizer()
            self.train()

        elif mode == "evaluation":
            self.hyper.vocab_init()
            self._init_model()
            self.load_model("best")
            test_set = self.Dataset(self.hyper, self.hyper.test)
            loader = self.Loader(
                test_set,
                batch_size=self.hyper.batch_size_eval,
                num_workers=8,
            )
            f1, log = self.evaluation(loader)
            print(log)
            print("f1 = ", f1)

        elif mode == "data_summary":
            self.hyper.vocab_init()
            self.summary_data(self.hyper.test)

        elif mode == "model_summary":
            self.hyper.vocab_init()
            self._init_model()
            self.load_model("best")
            parameter_num = np.sum([p.numel() for p in self.model.parameters()]).item()
            print(self.model)
            print(parameter_num)

        elif mode == "subevaluation":
            self.hyper.vocab_init()
            self._init_model()
            self.load_model("best")
            for data in self.hyper.subsets:
                test_set = self.Dataset(self.hyper, data)
                loader = self.Loader(
                    test_set,
                    batch_size=self.hyper.batch_size_eval,
                    num_workers=8,
                )
                f1, log = self.evaluation(loader)
                print(log)
                print("f1 = ", f1)

        elif mode == "debug":
            self.hyper.vocab_init()
            train_set = self.Dataset(self.hyper, self.hyper.dev)
            loader = self.Loader(
                train_set,
                batch_size=self.hyper.batch_size_train,
                num_workers=0,
            )
            for sample in tqdm(loader):
                print(sample.__dict__)
                exit()

        else:
            raise ValueError("invalid mode")

    def load_model(self, name: str):
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_dir, self.exp_name + "_" + name))
        )

    def save_model(self, name: str):
        # def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, self.exp_name + "_" + name),
        )

    def summary_data(self, dataset):
        len_sent_list = []
        triplet_num = []

        print(dataset)
        print("sentence num %d" % len(len_sent_list))
        print("all triplet num %d" % sum(triplet_num))
        print("avg sentence length %f" % (sum(len_sent_list) / len(len_sent_list)))
        print("avg triplet num %f" % (sum(triplet_num) / len(triplet_num)))
        print(Counter(triplet_num))
        print("\n")

    def evaluation(self, loader) -> Tuple[float, str]:
        self.model.metrics.reset()
        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(loader):
                output = self.model(sample, is_train=False)
                self.model.run_metrics(output)

        result = self.model.get_metric()
        score = result["fscore"]
        log = (
            ", ".join(
                [
                    "%s: %.4f" % (name, value)
                    for name, value in result.items()
                    if not name.startswith("_")
                ]
            )
            + " ||"
        )
        return score, log

    def train(self):
        train_set = self.Dataset(self.hyper, self.hyper.train)
        train_loader = self.Loader(
            train_set,
            batch_size=self.hyper.batch_size_train,
            num_workers=8,
        )
        dev_set = self.Dataset(self.hyper, self.hyper.dev)
        dev_loader = self.Loader(
            dev_set,
            batch_size=self.hyper.batch_size_eval,
            num_workers=4,
        )
        test_set = self.Dataset(self.hyper, self.hyper.test)
        test_loader = self.Loader(
            test_set,
            batch_size=self.hyper.batch_size_eval,
            num_workers=4,
        )
        score = 0
        best_epoch = 0
        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(train_loader)

            for sample in pbar:
                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)

                loss = output["loss"]
                assert isinstance(loss, torch.Tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

                self.optimizer.step()

                pbar.set_description(output["description"](epoch, self.hyper.epoch_num))

            self.save_model(str(epoch))
            if epoch % self.hyper.print_epoch == 0 and epoch >= 2:
                new_score, log = self.evaluation(dev_loader)
                logging.info(log)
                if new_score >= score:
                    score = new_score
                    best_epoch = epoch
                    self.save_model("best")
        logging.info("best epoch: %d \t F1 = %.2f" % (best_epoch, score))
        self.load_model("best")
        new_score, log = self.evaluation(test_loader)
        logging.info(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        "-e",
        type=str,
        default="chinese_seq2umt",
        help="experiments/exp_name.json",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="train",
        help="preprocessing|train|evaluation|subevaluation|data_summary|model_summary",
    )
    args = parser.parse_args()

    exp_name = args.exp_name
    assert isinstance(exp_name, str)

    mode = args.mode
    assert isinstance(mode, str)

    config = Runner(exp_name=exp_name)
    config.run(mode=mode)
