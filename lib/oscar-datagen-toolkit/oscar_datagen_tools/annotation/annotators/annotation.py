#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import json
import logging
from pathlib import Path
from typing import List

import coloredlogs
from pycococreatortools import pycococreatortools
from scipy import ndimage

from oscar_datagen_tools.common_utils import get_image_id, get_sensor_type

from ..utils import INFO, LICENSES, CategoriesHandler, calc_image_id
from .base import Annotator
from .format import Format

__all__ = ["COCO", "MOTS"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

TOLERANCE = 2


class COCO(Annotator):
    def __init__(self, interval: int, categories_handler: CategoriesHandler) -> None:
        super().__init__(interval, categories_handler)

        self._annot_id = 1

        self._annotation_obj = {}
        self._annotation_obj["annotations"] = []
        self._annotation_obj["videos"] = []
        self._annotation_obj["images"] = []
        self._annotation_obj["info"] = INFO
        self._annotation_obj["licenses"] = LICENSES

    def __pre_annotation__(self, category_mat_path: Path, data_path: Path, frame: int) -> ndimage:
        categories_mat = super().__pre_annotation__(category_mat_path, data_path, frame)

        camera_type_name = get_sensor_type(data_path.parent)
        image_id = calc_image_id(data_path)
        height, width, _ = categories_mat.shape

        image_info = pycococreatortools.create_image_info(
            image_id, data_path.name, (width, height)
        )
        image_info["video_id"] = camera_type_name
        image_info["frame_index"] = frame
        self._annotation_obj["images"].append(image_info)

        return categories_mat

    def annotate_sensor(self, category_mat_paths: List[Path], data_paths: List[Path]) -> bool:
        if not super().annotate_sensor(category_mat_paths, data_paths):
            return False

        self._annotation_obj["categories"] = self.categories_handler.categories

        return True

    def __parse_mask__(self, data_path: Path, mask: tuple) -> bool:
        height, width = mask.binary_mask.shape
        image_id = calc_image_id(data_path)
        annotation_info = pycococreatortools.create_annotation_info(
            self._annot_id,
            image_id,
            mask.category_info,
            mask.binary_mask,
            (width, height),
            tolerance=TOLERANCE,
        )

        if annotation_info is not None:
            if not super().__parse_mask__(data_path, mask):
                return False

            annotation_info["track_id"] = str(self._dict_annot[mask.obj_id])
            self._annotation_obj["annotations"].append(annotation_info)

        self._annot_id = self._annot_id + 1
        return True

    def __post_annotation__(self, data_path: Path, num_annotations: int) -> bool:
        if not super().__post_annotation__(data_path, num_annotations):
            return False

        if num_annotations == 0:
            # remove image information since there were not annotations
            self._annotation_obj["images"].pop()

        return True

    def store_annotations(self, filename: str) -> bool:
        with open(filename, "w+") as json_file:
            logger.info(f"Saving output file: {filename}")
            json.dump(self._annotation_obj, json_file, indent=4)

        # re-init class to clean up it's attributes and leave the
        # class ready to use in another annotation process.
        self.__init__(self.interval, self.categories_handler)

        return True


OBJECT_SCALE_FACTOR = 1000


class MOTS(Annotator):
    def __init__(
        self, interval: int, categories_handler: CategoriesHandler, format: Format
    ) -> None:
        super().__init__(interval, categories_handler)

        self.format = format

        self._annotations = []

    def annotate_sensor(self, category_mat_paths: List[Path], data_paths: List[Path]) -> bool:
        if not super().annotate_sensor(category_mat_paths, data_paths):
            return False

        self._annotations = [(category["name"]) for category in self.categories_handler.categories]
        return True

    def __parse_mask__(self, data_path: Path, mask: tuple) -> bool:
        if not super().__parse_mask__(data_path, mask):
            return False

        assert OBJECT_SCALE_FACTOR > int(self._dict_annot[mask.obj_id])
        object_id = int(mask.category_info["id"]) * OBJECT_SCALE_FACTOR + int(
            self._dict_annot[mask.obj_id]
        )
        image_id = get_image_id(data_path)
        self.format.process(mask, image_id, object_id)
        return True

    def __post_annotation__(self, data_path: Path, num_annotations: int) -> bool:
        if not super().__post_annotation__(data_path, num_annotations):
            return False

        self.format.save()
        return True

    def store_annotations(self, filename: str) -> bool:
        with open(filename, "w+") as categories_file:
            self._annotations = [annotation + "\n" for annotation in self._annotations]
            categories_file.writelines(self._annotations)

        # re-init class to clean up it's attributes and leave the
        # class ready to use in another annotation process.
        self.__init__(self.interval, self.categories_handler, self.format)

        return True
