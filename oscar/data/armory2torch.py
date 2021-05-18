#
# Copyright (C) 2020 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

from importlib import import_module
from typing import Callable, Optional, Type, Union

from armory.data.datasets import ArmoryDataGenerator, SUPPORTED_DATASETS, ucf101_clean
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning import LightningDataModule

from oscar.defences.preprocessor.preprocessor_pytorch import PreprocessorPyTorch


SUPPORTED_DATASETS = dict(SUPPORTED_DATASETS)  # don't overwrite original
SUPPORTED_DATASETS.update(ucf101_clean=ucf101_clean)

DEFENCE_CLASS_TYPE = Union[str, Type[PreprocessorPyTorch]]


class Armory2TorchDataset(IterableDataset):
    def __init__(
        self,
        generator: Union[ArmoryDataGenerator, DataLoader],
        preprocessor_defence: Optional[PreprocessorPyTorch] = None,
    ):

        if not (
            isinstance(generator, ArmoryDataGenerator)
            or isinstance(generator, DataLoader)
        ):
            raise TypeError(generator)

        if not isinstance(preprocessor_defence, PreprocessorPyTorch):
            raise TypeError(preprocessor_defence)

        if isinstance(generator, ArmoryDataGenerator) and generator.batch_size != 1:
            raise ValueError(
                f"Only supports batch_size=1, got batch_size={generator.batch_size}"
            )

        if isinstance(generator, DataLoader) and preprocessor_defence is not None:
            raise ValueError("Only supports preprocessor_defence for numpy framework")

        self.generator = (
            generator if isinstance(generator, ArmoryDataGenerator) else iter(generator)
        )
        self.preprocessor_defence = preprocessor_defence

    def __len__(self):
        if isinstance(self.generator, ArmoryDataGenerator):
            return len(self.generator)

        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        if isinstance(self.generator, ArmoryDataGenerator):
            x, y = self.generator.get_batch()
        else:
            x, y = next(self.generator)

        assert x.shape[0] == y.shape[0] == 1

        if self.preprocessor_defence is not None:
            if isinstance(x, np.ndarray):
                x, y = self.preprocessor_defence(x, y)

                x_torch = torch.from_numpy(x[0])
                y_torch = torch.from_numpy(y)[0]

            elif isinstance(x, torch.Tensor):
                # this is specific to our preprocessor implementation
                x = x.to(self.preprocessor_defence.device)
                y = y.to(self.preprocessor_defence.device)

                with torch.no_grad():
                    x, y = self.preprocessor_defence.forward(x, y)

                x_torch = x[0]
                y_torch = y[0]

            else:
                raise TypeError(x)

        return x_torch, y_torch

    @classmethod
    def get_dataset(
        cls,
        dataset_name: str,
        split: str = "test",
        epochs: int = 1,
        shuffle_files: bool = False,
        armory_framework: str = "numpy",
        preprocessor_defence: Optional[PreprocessorPyTorch] = None,
    ) -> "Armory2TorchDataset":

        dataset_fn = SUPPORTED_DATASETS[dataset_name]

        dataset_fn_kwargs = dict(
            split=split,
            batch_size=1,
            epochs=epochs,
            shuffle_files=shuffle_files,
            framework=armory_framework,
        )
        if armory_framework != "numpy":
            dataset_fn_kwargs.update(preprocessing_fn=None)

        generator = dataset_fn(**dataset_fn_kwargs)

        return cls(generator, preprocessor_defence)

    @classmethod
    def get_dataloader(
        cls,
        dataset_name: str,
        split: str = "test",
        epochs: int = 1,
        batch_size: int = 1,
        shuffle_files: bool = False,
        armory_framework: str = "numpy",
        preprocessor_defence: Optional[PreprocessorPyTorch] = None,
    ) -> DataLoader:

        dataset = cls.get_dataset(
            dataset_name=dataset_name,
            split=split,
            epochs=epochs,
            shuffle_files=shuffle_files,
            armory_framework=armory_framework,
            preprocessor_defence=preprocessor_defence,
        )

        # XXX: num_workers=0 is a bottleneck, but necessary for defence
        return DataLoader(dataset, batch_size=batch_size, num_workers=0)

    @classmethod
    def from_scenario_config(
        cls, config: dict, epochs: int = 1, shuffle_files: bool = False
    ):
        dataset_config = config["dataset"]

        if dataset_config["module"] != "armory.data.datasets":
            raise ValueError(dataset_config["module"])

        dataset_name = dataset_config["name"]

        if "train_split" in dataset_config and "eval_split" in dataset_config:
            raise KeyError(
                "Only one of 'train_split' or 'eval_split' should be present"
            )
        elif "train_split" in dataset_config:
            split = dataset_config["train_split"]
        elif "eval_split" in dataset_config:
            split = dataset_config["eval_split"]
        else:
            split = "test"

        armory_framework = dataset_config.get("framework", "numpy")

        preprocessor_defence = None
        if "defense" in config and config["defense"] is not None:
            defence_config = config["defense"]
            defence_module = import_module(defence_config["module"])
            defence_class = getattr(defence_module, defence_config["name"])
            defence_kwargs = defence_config["kwargs"]
            preprocessor_defence = defence_class(**defence_kwargs)

        return cls.get_dataset(
            dataset_name=dataset_name,
            split=split,
            epochs=epochs,
            shuffle_files=shuffle_files,
            armory_framework=armory_framework,
            preprocessor_defence=preprocessor_defence,
        )


class ArmoryDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 16,
        num_epochs: int = 100,
        armory_framework: str = "numpy",
        preprocessor_defence_class: Optional[DEFENCE_CLASS_TYPE] = None,
        preprocessor_defence_kwargs: Optional[dict] = None,
    ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.armory_framework = armory_framework

        self.preprocessor_defence_class = preprocessor_defence_class
        self.preprocessor_defence_kwargs = preprocessor_defence_kwargs or dict()

    @property
    def required_trainer_kwargs(self) -> dict:
        inverse_num_epochs = 1.0 / self.num_epochs

        return dict(
            max_epochs=1,
            val_check_interval=inverse_num_epochs,
            limit_val_batches=inverse_num_epochs,
        )

    def setup(self, stage=None):
        # this is done on a per-GPU basis
        preprocessor_defence_class = self.preprocessor_defence_class
        if type(preprocessor_defence_class) is str:
            parts = preprocessor_defence_class.split(".")
            module = ".".join(parts[:-1])
            module = import_module(module)
            classname = parts[-1]
            preprocessor_defence_class = getattr(module, classname)

        self.preprocessor_defence = (
            preprocessor_defence_class(**self.preprocessor_defence_kwargs)
            if preprocessor_defence_class is not None
            else None
        )

    def train_dataloader(self) -> DataLoader:
        return Armory2TorchDataset.get_dataloader(
            dataset_name=self.dataset_name,
            split="train",
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            shuffle_files=True,
            armory_framework=self.armory_framework,
            preprocessor_defence=self.preprocessor_defence,
        )

    def val_dataloader(self) -> DataLoader:
        return Armory2TorchDataset.get_dataloader(
            dataset_name=self.dataset_name,
            split="test",  # XXX: should we be doing this?
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            shuffle_files=False,
            armory_framework=self.armory_framework,
            preprocessor_defence=self.preprocessor_defence,
        )

    def test_dataloader(self) -> DataLoader:
        return Armory2TorchDataset.get_dataloader(
            dataset_name=self.dataset_name,
            split="test",
            epochs=1,
            batch_size=self.batch_size,
            shuffle_files=False,
            armory_framework=self.armory_framework,
            preprocessor_defence=self.preprocessor_defence,
        )
