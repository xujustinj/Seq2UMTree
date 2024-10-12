import logging
import os
from typing import Optional

import numpy as np
import torch
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from tqdm import tqdm

from seq2umt.data import Seq2UMTreeData, Seq2UMTreeDataset, Seq2UMTreeDataLoader
from seq2umt.config import Seq2UMTreeConfig
from seq2umt.models import Seq2UMTree
from seq2umt.preprocess import Seq2UMTreePreprocessor
from seq2umt.types import OptimizerName, SplitName

from util.device import get_device


class Runner:
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.model_dir = "saved_models"

        self.config = Seq2UMTreeConfig(os.path.join("experiments", self.exp_name + ".json"))
        self.device = get_device()

        logging.basicConfig(
            filename=os.path.join("experiments", self.exp_name + ".log"),
            filemode="w",
            format="%(asctime)s - %(message)s",
            level=logging.INFO,
        )

    def _get_dataloader(
            self,
            split: SplitName,
            batch_size: Optional[int] = 1,
            num_workers: int = 0,
            shuffle: Optional[bool] = None,
    ) -> tuple[Seq2UMTreeDataset, Seq2UMTreeDataLoader]:
        dataset = Seq2UMTreeDataset(config=self.config, dataset=split)
        dataloader = Seq2UMTreeDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
        return dataset, dataloader

    def _get_optimizer(self, model: Seq2UMTree) -> Optimizer:
        name: OptimizerName = self.config.optimizer
        if name == "adam":
            return Adam(model.parameters())
        elif name == "sgd":
            return SGD(model.parameters(), lr=0.5)

    def _get_model(self) -> Seq2UMTree:
        model = Seq2UMTree(self.config)
        model.to(device=self.device)
        return model

    def preprocessing(self):
        preprocessor = Seq2UMTreePreprocessor(self.config)
        preprocessor.gen_relation_vocab()
        preprocessor.gen_all_data()
        preprocessor.gen_vocab(min_freq=2)
        preprocessor.gen_schema()

    def run(self, mode: str):
        if mode == "preprocessing":
            self.preprocessing()

        elif mode == "train":
            self.train()

        elif mode == "evaluation":
            model = self._get_model()
            self.load_model(model, "best")
            dataset, loader = self._get_dataloader(
                split=self.config.test,
                batch_size=self.config.batch_size_eval,
                num_workers=8,
            )
            f1, log = self.evaluation(model=model, dataset=dataset, loader=loader)
            print(log)
            print("f1 = ", f1)

        elif mode == "model_summary":
            model = self._get_model()
            self.load_model(model, "best")
            parameter_num = np.sum([p.numel() for p in model.parameters()]).item()
            print(model)
            print(parameter_num)

        elif mode == "debug":
            loader = self._get_dataloader(
                split=self.config.dev,
                batch_size=self.config.batch_size_train,
                num_workers=0,
            )
            for sample in tqdm(loader):
                print(sample.__dict__)
                exit()

        else:
            raise ValueError("invalid mode")

    def model_path(self, name) -> str:
        return os.path.join(self.model_dir, f"{self.exp_name} {name}")

    def load_model(self, model: Seq2UMTree, name):
        model.load_state_dict(torch.load(self.model_path(name)))

    def save_model(self, model: Seq2UMTree, name):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(model.state_dict(), self.model_path(name))

    def evaluation(self, model: Seq2UMTree, dataset: Seq2UMTreeDataset, loader: Seq2UMTreeDataLoader) -> tuple[float, str]:
        model.metrics.reset()
        model.eval()

        schema_data = dataset.get_schema().to(device=self.device)
        with torch.no_grad():
            for sample in tqdm(loader):
                assert isinstance(sample, Seq2UMTreeData)
                sample = sample.to(device=self.device)
                output = model(sample, schema_data)
                model.run_metrics(output)

        result = model.get_metric()
        score = result["fscore"]
        log = (
            ", ".join(
                f"{name}: {value:.4f}"
                for name, value in result.items()
                if not name.startswith("_")
            )
            + " ||"
        )
        return score, log

    def train(self):
        model = self._get_model()
        optimizer = self._get_optimizer(model)
        train_dataset, train_loader = self._get_dataloader(
            split=self.config.train,
            batch_size=self.config.batch_size_train,
            num_workers=8,
            shuffle=True,
        )
        dev_dataset, dev_loader = self._get_dataloader(
            split=self.config.dev,
            batch_size=self.config.batch_size_eval,
            num_workers=4,
        )
        test_dataset, test_loader = self._get_dataloader(
            split=self.config.test,
            batch_size=self.config.batch_size_eval,
            num_workers=4,
        )
        score = -float("inf")
        best_epoch = -1
        num_epochs = self.config.epoch_num
        schema_data = train_dataset.get_schema().to(device=self.device)
        for epoch in range(1, 1+num_epochs):
            model.train()
            pbar = tqdm(train_loader)

            for sample in pbar:
                assert isinstance(sample, Seq2UMTreeData)

                optimizer.zero_grad()
                sample = sample.to(device=self.device)
                output = model(sample, schema_data)

                loss = output["loss"]
                assert isinstance(loss, torch.Tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

                optimizer.step()

                pbar.set_description(f"L: {float(loss.item()):.3f}, epoch: {epoch}/{num_epochs}")

            self.save_model(model, name=epoch)

            new_score, log = self.evaluation(model=model, dataset=dev_dataset, loader=dev_loader)
            logging.info(log)
            if new_score >= score:
                score = new_score
                best_epoch = epoch
                self.save_model(model, name="best")
        logging.info(f"best epoch: {best_epoch}\tvalidation F1 = {score:.3f}")
        self.load_model(model, name="best")
        new_score, log = self.evaluation(model=model, dataset=test_dataset, loader=test_loader)
        logging.info(log)


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        "-e",
        type=str,
        default="nyt_seq2umt_pos",
        help="experiments/{exp_name}.json",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="train",
        help="preprocessing|train|evaluation|model_summary",
    )
    args = parser.parse_args()

    exp_name = args.exp_name
    assert isinstance(exp_name, str)

    mode = args.mode
    assert isinstance(mode, str)

    def set_random_seeds(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    set_random_seeds(0)

    runner = Runner(exp_name=exp_name)
    runner.run(mode=mode)
