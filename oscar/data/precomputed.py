#
# Copyright (C) 2020 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule


logger = logging.getLogger(__name__)

_TRANSFORMS_DICT_TYPE = Optional[
    Dict[str, Optional[Callable[[torch.Tensor], torch.Tensor]]]
]


class PrecomputedDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        item_keys: Tuple[str] = ("x", "y"),
        transforms_dict: _TRANSFORMS_DICT_TYPE = None,
        mode: str = "r",
    ):

        self._hf = h5py.File(dataset_path, mode=mode, swmr=True)
        logger.info(f"Accessed precomputed data from `{dataset_path}`.")

        items_dset_name = "__items__"
        if items_dset_name not in self._hf:
            self._hf.create_dataset(
                items_dset_name, dtype=h5py.string_dtype(), shape=(0,), maxshape=(None,)
            )

        self._items_dset = self._hf.get(items_dset_name)
        self._items_list = self._items_dset[:].tolist() if mode == "r" else None

        self.item_keys = item_keys
        self.transforms_dict = (
            dict(transforms_dict) if transforms_dict is not None else dict()
        )

    @property
    def items(self) -> List[str]:
        if self._items_list is not None:
            return self._items_list

        else:
            return self._items_dset[:].tolist()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        item_name = self.items[idx]
        grp = self._hf.get(item_name)

        data = list()
        for k in self.item_keys:
            transform = self.transforms_dict.get(k, lambda v: v)
            if transform is None:
                transform = lambda v: v

            v = np.array(grp.get(k))
            v = torch.tensor(v)
            v = transform(v)

            data.append(v)

        return tuple(data)

    def add_item(self, item_name: str, **kwargs: Any) -> int:
        item_name = str(item_name)
        grp = self._hf.create_group(item_name)

        for k, v in kwargs.items():
            if k not in self.item_keys:
                raise KeyError(
                    f"Got unknown key '{k}'. Supported keys: {self.item_keys}"
                )

            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            grp.create_dataset(k, data=v)

        idx = len(self)
        self._items_dset.resize((idx + 1,))
        self._items_dset[idx] = item_name

        return idx

    def add_item_no_name(self, **kwargs: Any) -> int:
        idx = len(self)
        item_name = f"idx_{idx}"
        return self.add_item(item_name, **kwargs)


class PrecomputedLightningDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_path: str,
        test_dataset_path: str,
        val_dataset_path: Optional[str] = None,
        val_split_percent: Optional[float] = None,
        item_keys: Tuple[str] = ("x", "y"),
        transforms_dict: _TRANSFORMS_DICT_TYPE = None,
        dataloader_batch_size=64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if val_dataset_path is None and val_split_percent is None:
            val_dataset_path = test_dataset_path

        elif val_dataset_path is None and val_split_percent is not None:
            val_split_percent = float(val_split_percent)

            if val_split_percent <= 0.0 or val_split_percent >= 100.0:
                raise ValueError(f"val_split_percent = {val_split_percent}")

            val_split_percent = (
                val_split_percent
                if val_split_percent < 1.0
                else val_split_percent / 100.0
            )

        elif val_dataset_path is not None and val_split_percent is not None:
            raise ValueError(
                "Only one of `val_dataset_path` and `val_split_percent` should be set."
            )

        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.val_dataset_path = val_dataset_path
        self.val_split_percent = val_split_percent

        self.item_keys = item_keys
        self.transforms_dict = transforms_dict

        self.dataloader_batch_size = dataloader_batch_size

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_dataset = PrecomputedDataset(
                self.train_dataset_path,
                item_keys=self.item_keys,
                transforms_dict=self.transforms_dict,
            )

            if self.val_dataset_path is not None:
                val_dataset = PrecomputedDataset(
                    self.val_dataset_path,
                    item_keys=self.item_keys,
                    transforms_dict=self.transforms_dict,
                )

                self.train_split, self.val_split = train_dataset, val_dataset

            elif self.val_split_percent is not None:
                num_samples = len(train_dataset)
                val_len = int(self.val_split_percent * num_samples)
                train_len = num_samples - val_len

                logger.info(f"Train/val split: {train_len}, {val_len}")

                self.train_split, self.val_split = random_split(
                    train_dataset, [train_len, val_len]
                )

        if stage == "test" or stage is None:
            self.test_split = PrecomputedDataset(
                self.test_dataset_path,
                item_keys=self.item_keys,
                transforms_dict=self.transforms_dict,
            )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_split,
            shuffle=True,
            num_workers=0,
            batch_size=self.dataloader_batch_size,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.val_split,
            num_workers=0,
            batch_size=self.dataloader_batch_size,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.test_split,
            num_workers=0,
            batch_size=self.dataloader_batch_size,
        )
