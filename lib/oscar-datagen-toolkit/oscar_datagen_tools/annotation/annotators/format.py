#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import abc
import logging
from pathlib import Path

import coloredlogs
import imageio
import numpy as np
import pycocotools.mask as rletools

__all__ = ["Format", "PNG", "Text"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class Format(abc.ABC):
    def __init__(self, output) -> None:
        self.output = output

    @abc.abstractmethod
    def process(self, mask: tuple, image_id: str, object_id: int) -> None:
        pass

    @abc.abstractmethod
    def save(self) -> None:
        pass


class PNG(Format):
    def __init__(self, output) -> None:
        super().__init__(output)

        self._filename = ""
        self._mots_mat = None

    def process(self, mask: tuple, image_id: str, object_id: int) -> None:
        height, width = mask.binary_mask.shape
        self._filename = f"{image_id}.png"

        if self._mots_mat is None:
            # Create image annotations with 16 bit png channel
            self._mots_mat = np.zeros((height, width), dtype=np.uint16)

        idx = np.where(mask.binary_mask == 1)
        self._mots_mat[idx] = object_id

    def save(self) -> None:
        if self._mots_mat is None:
            return

        unique, counts = np.unique(self._mots_mat, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        logger.debug(f"Frequencies: {frequencies}")

        if not self.output.exists():
            self.output.mkdir()

        # Write annotation images to instances/ folder
        path = self.output / self._filename
        imageio.imwrite(path, self._mots_mat.astype(np.uint16))

        # cleanup
        self._mots_mat = None
        self._filename = ""


class Text(Format):
    def __init__(self, output) -> None:
        super().__init__(output)

        self.output = self.output / Path("instances.txt")
        self._annot_lines = []

        # empty the text file if it exists
        if self.output.exists():
            with open(self.output, "r+") as ann_file:
                ann_file.truncate(0)

    def process(self, mask: tuple, image_id: str, object_id: int) -> None:
        height, width = mask.binary_mask.shape
        category_id = mask.category_info["id"]

        rle = rletools.encode(np.asfortranarray(mask.binary_mask))["counts"]
        rle = str(rle, "utf-8")
        logger.debug(f"RLE: {rle}")

        ann_str = f"{image_id} {object_id} {category_id} {height} {width} {rle} \n"
        self._annot_lines.append(ann_str)

    def save(self) -> None:
        with open(self.output, "a") as ann_file:
            ann_file.writelines(self._annot_lines)

        # cleanup
        self._annot_lines = []
