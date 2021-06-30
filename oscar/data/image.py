#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#
import logging
import pickle
from pathlib import Path
from typing import Union, Optional, Callable, Tuple, List, Any, Dict

import torch
import torchvision
from torchvision.datasets import ImageFolder as _ImageFolder
from torchvision.datasets.folder import make_dataset as _make_dataset
from torchvision.datasets.folder import default_loader

logger = logging.getLogger(__name__)


class ImageFolder(_ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        # BEGIN: Monkey patch make_dataset
        torchvision.datasets.folder.make_dataset = ImageFolder.cached_make_dataset
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform,
                         loader=loader,
                         is_valid_file=None)
        torchvision.datasets.folder.make_dataset = _make_dataset
        # END: Monkey patch make_dataset

        # Filter samples
        logger.info("Filtering samples...")
        self.samples = self.filter_samples(self.samples,
                                           is_valid_file=is_valid_file,
                                           idx_to_class=self.classes)
        logger.info("done!")

    @staticmethod
    def cached_make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        cache = Path(directory) / "make_dataset.cache"

        if cache.exists():
            logger.info("Loading dataset cache: %s", cache)
            instances = pickle.load(cache.open('rb'))
        else:
            logger.info("Making dataset cache: %s", cache)
            instances = _make_dataset(directory, class_to_idx, extensions, is_valid_file)
            logger.info("Saving dataset cache: %s", cache)
            pickle.dump(instances, cache.open('wb'))

        return instances

    @staticmethod
    def filter_samples(images, is_valid_file=None, idx_to_class=None):
        # images = [(path, class_idx)]
        filtered_images = []

        for path, class_idx in images:
            if is_valid_file is not None and not is_valid_file(path):
                continue

            label = idx_to_class[class_idx]

            filtered_images.append((path, label))

        return filtered_images
