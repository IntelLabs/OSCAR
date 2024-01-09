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

from ..utils import (
    CategoriesHandler,
    build_projection_matrix,
    filename_decode,
    get_image_point,
    load_dynamic_metadata,
    load_sensors_static_metadata,
)

__all__ = ["Annotator"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class Annotator(abc.ABC):
    def __init__(
        self, interval: int, categories_handler: CategoriesHandler, sensor_type: str
    ) -> None:
        self.interval = interval
        self.categories_handler = categories_handler
        self.sensor_type = sensor_type

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

        # get dataset dir from image path that has the following format: dataset_dir/<modality>/<collection_timestamp>/00000x.png
        base_path = path.parent.parent.parent
        assert base_path.exists()

        patch_path = base_path / "foreground_mask" / path.parent.name
        if not patch_path.exists():
            patch_path.mkdir(parents=True)

        img_patch_mask = Image.fromarray(self._patch_mask)
        rgb_img_patch_mask = img_patch_mask.convert("RGB")
        rgb_img_patch_mask.save(patch_path / path.name)

    def __extract_patch_metadata__(self, path: Path):
        if self._patch_mask is None:
            return

        # get the static metadata for the given sensor
        camera_id, frame_num = filename_decode(path.name)
        run_id = path.parent.name
        static_metadata_path = (
            path.parent.parent.parent / "metadata" / run_id / "metadata_static.json"
        )
        assert static_metadata_path.exists()
        sensor_static_metadata = load_sensors_static_metadata(
            static_metadata_path, self.sensor_type, camera_id
        )

        # calculate camera matrix
        image_width = sensor_static_metadata["image_size_x"]
        image_height = sensor_static_metadata["image_size_y"]
        fov = sensor_static_metadata["fov"]
        k = build_projection_matrix(image_width, image_height, fov)

        # get the dynamic metadata for the given sensor and patches
        dynamic_metadata_path = (
            path.parent.parent.parent / "metadata" / run_id / f"{frame_num}.json"
        )
        assert dynamic_metadata_path.exists()
        dynamic_metadata = load_dynamic_metadata(
            dynamic_metadata_path, self.sensor_type, camera_id
        )

        # map from 2D to 3D
        w2c = np.array(dynamic_metadata["sensor"]["actorsnapshot_transform_get_inverse_matrix"])
        corners = []
        for patch_key in dynamic_metadata["patches"].keys():
            patch_corners = dynamic_metadata["patches"][patch_key]["corners"]
            for corner_key in patch_corners.keys():
                location = patch_corners[corner_key]["actor_location"]
                # point -> [x, y]
                point = get_image_point(location, k, w2c)
                corner = [int(point[0]), int(point[1])]

                # verify that the mapped point is inside the camera view
                if 0 <= corner[0] <= image_width and 0 <= corner[1] <= image_height:
                    corners.append(corner)
                else:
                    corners = []
                    break

            # stop iterating when a patch that is in the camera view is found
            if len(corners) > 0:
                break

        # store the corners coordinates
        patch_metadata_path = path.parent.parent.parent / "patch_metadata" / run_id
        if not patch_metadata_path.exists():
            patch_metadata_path.mkdir(parents=True)

        patch_metadata_coords_path = patch_metadata_path / f"{path.stem}_coords.npy"
        np.save(patch_metadata_coords_path, np.array(corners))

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
