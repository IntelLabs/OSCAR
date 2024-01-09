#!/usr/bin/env python3
#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import argparse
import logging
from pathlib import Path
from typing import List

import coloredlogs
import fire

from oscar_datagen_tools.annotation.annotators import COCO, MOTS
from oscar_datagen_tools.annotation.annotators import PNG as MOTSPng
from oscar_datagen_tools.annotation.annotators import Annotator
from oscar_datagen_tools.annotation.annotators import Text as MOTSText
from oscar_datagen_tools.annotation.utils import (
    CARLA_CATEGORIES,
    CATEGORIES_KEY,
    SENSORS_KEY,
    CategoriesHandler,
    verify_dataset_path,
)

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class AnnotatorController:
    def __init__(
        self,
        annotator: Annotator,
        dataset_path: Path,
        exclude_dirs: List[Path],
        sensor: str,
        output: str,
    ) -> None:
        self.annotator = annotator
        self.dataset_path = dataset_path
        self.exclude_dirs = exclude_dirs
        self.dataset = None
        self.sensor = sensor
        self.output = output

    def prepare(self) -> bool:
        logger.info(f"Dataset path: {self.dataset_path}")

        self.dataset = verify_dataset_path(self.dataset_path, self.exclude_dirs)
        if self.dataset is None:
            return False

        return True

    def annotate(self) -> True:
        if self.sensor not in self.dataset[SENSORS_KEY]:
            logger.error(f"No data found for sensor: {self.sensor}")
            return False

        if CATEGORIES_KEY not in self.dataset[SENSORS_KEY]:
            logger.error(f"No data found for sensor: {CATEGORIES_KEY}")
            return False

        runs = self.dataset[SENSORS_KEY][self.sensor].keys()
        for run in runs:
            logger.info(f"Annotating run {run}")
            sensor_paths = self.dataset[SENSORS_KEY][self.sensor][run]
            categories_paths = self.dataset[SENSORS_KEY][CATEGORIES_KEY][run]
            self.annotator.annotate_sensor(categories_paths, sensor_paths)

        filename = self.dataset_path / self.output
        self.annotator.store_annotations(filename)

        return True


class CLI:
    """COCO converter."""

    def __init__(
        self,
        dataset_parent_dir: str = "outputs",
        exclude_dirs: List[str] = [],
        output: str = "out.json",
        interval: int = 1,
        categories: List[str] = ["Pedestrian", "Vehicle", "TrafficLight"],
        sensor: str = "rgb",
        binary_fill_holes: bool = False,
    ) -> None:
        self.dataset_parent_dir = Path(dataset_parent_dir)
        self.exclude_dirs = exclude_dirs
        self.output = output
        self.interval = interval
        self.categories = categories
        self.sensor = sensor
        self.binary_fill_holes = binary_fill_holes

        for i, exclude_dir in enumerate(self.exclude_dirs):
            self.exclude_dirs[i] = Path(exclude_dir)

        self._categories_handler = CategoriesHandler(self.categories, self.binary_fill_holes)

    def list(self) -> None:
        """List CARLA semantic categories."""
        logger.info(f"CARLA categories: {CARLA_CATEGORIES}")

    def __annotate__(self, annotator: Annotator) -> None:
        """Trigger the annotation process."""
        controller = AnnotatorController(
            annotator, self.dataset_parent_dir, self.exclude_dirs, self.sensor, self.output
        )

        if not controller.prepare():
            raise Exception("Annotator controller preparation failed.")

        if not controller.annotate():
            raise Exception("Annotate process failed.")

    def kwcoco(self) -> None:
        """COCO compatible annotation."""
        annotator = COCO(self.interval, self._categories_handler, self.sensor)
        self.__annotate__(annotator)

    def mots_txt(self) -> None:
        """MOTS compatible annotation, generating a text file."""
        annot_format = MOTSText(self.dataset_parent_dir)
        annotator = MOTS(self.interval, self._categories_handler, self.sensor, annot_format)
        self.__annotate__(annotator)

    def mots_png(self) -> None:
        """MOTS compatible annotation, generating PNG images."""
        annot_format = MOTSPng(self.dataset_parent_dir / "instances")
        annotator = MOTS(self.interval, self._categories_handler, self.sensor, annot_format)
        self.__annotate__(annotator)


def main() -> None:
    cli = fire.Fire(CLI)


if __name__ == "__main__":
    main()
