#
# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import datetime
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import coloredlogs
import cv2
import numpy as np
from scipy import ndimage

__all__ = ["CategoriesHandler", "INFO", "CARLA_CATEGORIES", "LICENSES"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


INFO = {
    "description": "Carla Dataset",
    "url": "https://github.com/carla-simulator/carla",
    "version": "0.9.12",
    "year": 2021,
    "contributor": "",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [{"id": 1, "name": "", "url": ""}]

# Define Carla semantic categories - as per Carla v0.9.13
# https://carla.readthedocs.io/en/0.9.13/ref_sensors/#semantic-segmentation-camera
CARLA_CATEGORIES = [
    "Unlabeled",
    "Building",
    "Fence",
    "Other",
    "Pedestrian",
    "Pole",
    "RoadLine",
    "Road",
    "Sidewalk",
    "Vegetation",
    "Vehicle",
    "Wall",
    "TrafficSign",
    "Sky",
    "Ground",
    "Bridge",
    "RailTrack",
    "GuardRail",
    "TrafficLight",
    "Static",
    "Dynamic",
    "Water",
    "Terrain",
]


@dataclass
class Mask:
    binary_mask: ndimage
    category_info: Dict[str, Any]
    obj_id: int


class CategoriesHandler:
    def __init__(self, user_categories: List[str], closing: bool):
        self.closing = closing

        self.categories = []
        self.category_ids = {}
        self.__verify_categories__(user_categories)

    def __verify_categories__(self, user_categories: List[str]):
        """Verify that the especified categories are included in CARLA.

        Returns
        -------
        categories: Dict[str, Any]
            Dictionary with the verified categories along with their ids.
        """
        # start dataset's ids from 1
        idx = 1
        for category in user_categories:
            if category not in CARLA_CATEGORIES:
                print(f"Skip category {category} as it is not a CARLA semantic category")
                continue

            # map the CARLA category ids with the dataset category ids
            # <CARLA Index>: <Dataset Index>
            self.category_ids[CARLA_CATEGORIES.index(category)] = idx
            self.categories.append({"supercategory": category, "id": idx, "name": category})
            idx += 1

        if len(self.categories) == 0:
            raise Exception("No valid categories found")

        logger.info(f"Categories: {self.categories}")

    def __process_binary_mask__(
        self, obj_binary: ndimage, img_category: ndimage, frame_category: int
    ):
        """Post-processing of the extracted binary mask.

        Parameters
        ----------
        obj_binary: numpy.ndarray
            Binary mask extracted.
        img_category: numpy.ndarray
            Plane with the scene's categories.
        frame_category: int
            Mask's category.
        Returns
        -------
        binary_mask: numpy.ndarray
            Processed binary mask.
        """
        binary_mask = obj_binary
        contours, _ = cv2.findContours(obj_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # TBD: Support partial occlusion using IoU of segmentation mask area and 3d bounding boxes from carla projected into 2d space
        if frame_category == CARLA_CATEGORIES.index("TrafficLight"):
            # Treat TrafficLight differently as we want to only extract lights (do not include poles)
            for contour in contours:
                # Valid polygons have >=6 coordinates (3 points)
                if contour.size >= 6:
                    binary_mask = np.zeros_like(img_category)
                    binary_mask = cv2.fillPoly(binary_mask, pts=[contour], color=(1, 1, 1))

        return binary_mask

    def extract_masks(self, instance_segmentation_im: ndimage):
        """Extract binary masks from CARLA's instance segmentation sensor outputs.

        Parameters
        ----------
        instance_segmentation_im : numpy.ndarray
            3-channel instance segmentation image.
        Returns
        -------
        masks: List[Mask]
        """

        masks = []
        img_category = instance_segmentation_im[:, :, 0]

        # R = Category id, G = first byte of Object ID, B = second byte of object ID
        # Merge G and B values to resolve Object ID and create 2D array with only Object IDs
        obj_ids = (
            instance_segmentation_im[:, :, 2].astype(np.uint32) << 8
        ) + instance_segmentation_im[:, :, 1].astype(np.uint32)

        # List of all unique Object IDs
        all_obj_ids = list(np.unique(obj_ids))

        for obj_id in all_obj_ids:
            frame_categories = np.unique(img_category[np.where(obj_ids == obj_id)])
            frame_categories = [
                frame_category
                for frame_category in frame_categories
                if frame_category in self.category_ids.keys()
            ]

            # If there are not categories extracted from the instance segmentation
            # will continue with the next object id.
            if not frame_categories:
                continue

            is_crowd = 0

            for frame_category in frame_categories:
                category_id = self.category_ids[frame_category]
                category_info = {"id": category_id, "is_crowd": is_crowd}

                extract_obj_binary = np.zeros_like(img_category)
                extract_obj_binary[
                    np.where((img_category == frame_category) & (obj_ids == obj_id))
                ] = 1
                if self.closing:
                    extract_obj_binary = ndimage.binary_fill_holes(extract_obj_binary).astype(
                        np.uint8
                    )

                if np.sum(extract_obj_binary) > 0:
                    binary_mask = self.__process_binary_mask__(
                        extract_obj_binary, img_category, frame_category
                    )
                    mask = Mask(
                        binary_mask=binary_mask, category_info=category_info, obj_id=obj_id
                    )
                    masks.append(mask)

        return masks
