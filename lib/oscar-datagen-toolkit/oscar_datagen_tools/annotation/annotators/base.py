#
# Copyright (C) 2024 Intel Corporation
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

from oscar_datagen_tools.common_utils import (
    build_projection_matrix,
    filename_decode,
    get_patch_in_camera,
    load_dynamic_metadata,
    load_sensors_static_metadata,
    verify_image_filename,
)

from ..utils import CategoriesHandler

__all__ = ["Annotator"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class Annotator(abc.ABC):
    def __init__(
        self,
        categories_handler: CategoriesHandler,
        sensor_type: str,
        dataset_path: Path,
        interval: int = 1,
    ) -> None:
        self.interval = interval
        self.categories_handler = categories_handler
        self.sensor_type = sensor_type
        self.dataset_path = dataset_path

        assert self.dataset_path.exists()

        self._dict_annot = {}
        self._obj_id_count = 1
        self._patch_mask = None

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
            desc="Run annotation progress",
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
            self._patch_mask = None

            for mask in masks:
                category_name = self.categories_handler.get_category_name_from_id(
                    mask.category_info["id"]
                )
                # Verify if the mask is related with a patch. If that is
                # the case, generate the patch mask and metadata, but skip
                # its annotation.
                if category_name == "Unlabeled":
                    self.__build_patch_mask__(mask.binary_mask)
                    continue

                if not self.__parse_mask__(data_path, mask):
                    logger.error(f"Mask parse failed for {data_path}")
                    return False

            self.__save_patch_mask__(data_path)
            self.__extract_patch_metadata__(data_path)

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

    def __build_patch_mask__(self, binary_mask: ndimage):
        if self._patch_mask is None:
            height, width = binary_mask.shape
            self._patch_mask = np.zeros((height, width), dtype=np.uint8)

        idx = np.where(binary_mask == 1)
        self._patch_mask[idx] = 255

    def __save_patch_mask__(self, path: Path):
        if self._patch_mask is None:
            return

        patch_path = self.dataset_path / "foreground_mask"
        if not patch_path.exists():
            patch_path.mkdir(parents=True)

        img_patch_mask = Image.fromarray(self._patch_mask)
        rgb_img_patch_mask = img_patch_mask.convert("RGB")
        rgb_img_patch_mask.save(patch_path / path.name)

    def __extract_patch_metadata__(self, path: Path):
        if self._patch_mask is None:
            return

        # get the static metadata for the given sensor
        sensor_id, frame_num = filename_decode(path.name)
        static_metadata_path = self.dataset_path / "metadata" / "metadata_static.json"
        assert static_metadata_path.exists()
        sensor_static_metadata = load_sensors_static_metadata(static_metadata_path, sensor_id)
        assert sensor_static_metadata, f"Did not find static data for sensor {sensor_id}"

        # get the dynamic metadata for the given sensor and patches
        dynamic_metadata_path = self.dataset_path / "metadata" / f"{frame_num}.json"
        assert dynamic_metadata_path.exists()
        dynamic_metadata = load_dynamic_metadata(dynamic_metadata_path, sensor_id)
        assert dynamic_metadata, f"Did not find dynamic data for sensor {sensor_id}"

        # get patch corners
        _, corners = get_patch_in_camera(sensor_static_metadata, dynamic_metadata)

        # post-processing the corner's points
        # 1. Clip the points
        # 2. Round its values
        image_width = sensor_static_metadata["image_size_x"]
        image_height = sensor_static_metadata["image_size_y"]
        processed_corners = []
        for corner in corners:
            x, y = corner
            x = int(round(min(max(x, 0), image_width)))
            y = int(round(min(max(y, 0), image_height)))

            processed_corners.append([x, y])

        # store the corners coordinates
        patch_metadata_path = self.dataset_path / "patch_metadata"
        if not patch_metadata_path.exists():
            patch_metadata_path.mkdir(parents=True)

        patch_metadata_raw_coords_path = patch_metadata_path / f"{path.stem}_raw_coords.npy"
        np.save(patch_metadata_raw_coords_path, np.array(corners))

        patch_metadata_coords_path = patch_metadata_path / f"{path.stem}_coords.npy"
        np.save(patch_metadata_coords_path, np.array(processed_corners))

        if self.sensor_type == "depth":
            # load depth image
            depth_image = Image.open(path)
            depth_image = np.array(depth_image).astype(np.float32)

            # convert depth image to a grayscale one
            # formula taken from: https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
            R = depth_image[:, :, 0]
            G = depth_image[:, :, 1]
            B = depth_image[:, :, 2]
            normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
            in_meters = 1000 * normalized

            # extract average depth of patch
            idx = np.where(self._patch_mask == 255)
            depth_avg = np.average(in_meters[idx])

            patch_metadata_depth_avg_path = patch_metadata_path / f"{path.stem}_avg_depth.npy"
            np.save(patch_metadata_depth_avg_path, np.array(depth_avg))

    def __post_annotation__(self, data_path: Path, num_annotations: int) -> bool:
        logger.debug(f"Created {num_annotations} to {data_path}")
        return True

    @abc.abstractmethod
    def store_annotations(self, filename: str) -> bool:
        pass
