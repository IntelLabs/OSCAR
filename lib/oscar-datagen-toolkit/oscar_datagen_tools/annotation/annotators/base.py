#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import abc
import logging
from pathlib import Path
from typing import List

import coloredlogs
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from oscar_datagen_tools.common_utils import verify_image_filename

from ..utils import CategoriesHandler

__all__ = ["Annotator"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class Annotator(abc.ABC):
    def __init__(self, interval: int, categories_handler: CategoriesHandler) -> None:
        self.interval = interval
        self.categories_handler = categories_handler

        self._dict_annot = {}
        self._obj_id_count = 1

    def __pre_annotation__(self, category_mat_path: Path, data_path: Path, frame: int) -> ndimage:
        im = Image.open(category_mat_path)
        categories_mat = np.array(im)
        return categories_mat

    def annotate_sensor(self, categories_mat_paths: List[Path], data_paths: List[Path]) -> bool:
        categories_mat_paths.sort()
        data_paths.sort()

        for idx, (data_path, categories_mat_path) in tqdm(
            enumerate(zip(data_paths, categories_mat_paths)),
            total=len(data_paths),
            desc="Camera annotation progress",
        ):
            if idx % self.interval != 0:
                continue

            # verify paths
            if not verify_image_filename(data_path.name):
                return False
            if not verify_image_filename(categories_mat_path.name):
                return False

            # Note: Adding 1 to the index idx to match the frame count in the oscar_data_saver.py tool
            categories_mat = self.__pre_annotation__(categories_mat_path, data_path, idx + 1)
            masks = self.categories_handler.extract_masks(categories_mat)

            for mask in masks:
                if not self.__parse_mask__(data_path, mask):
                    logger.error(f"Mask parse failed for {data_path}")
                    return False

            if not self.__post_annotation__(data_path, len(masks)):
                logger.error(f"Post-annotation failed for {data_path}")
                return False

        return True

    def __parse_mask__(self, data_path: Path, mask: tuple) -> bool:
        # Map UE object IDs to IDs starting from 1
        if mask.obj_id not in self._dict_annot.keys():
            self._dict_annot[mask.obj_id] = self._obj_id_count
            self._obj_id_count += 1

        return True

    def __post_annotation__(self, data_path: Path, num_annotations: int) -> bool:
        logger.debug(f"Created {num_annotations} to {data_path}")
        return True

    @abc.abstractmethod
    def store_annotations(self, filename: str) -> bool:
        pass
