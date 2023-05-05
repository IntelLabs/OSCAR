#
# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List

import coloredlogs

from oscar_datagen_tools.common_utils import (
    get_camera_id,
    get_sensor_type,
    verify_camera_name,
    verify_sensor_name,
)

__all__ = ["calc_image_id", "verify_dataset_path"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


def verify_dataset_path(path: Path, exclude_dirs: List[Path]) -> Dict[str, Any]:
    """Verify that the especified dataset path contains the expected elements.

    Parameters
    ----------
    path : Path
        Path to the dataset.
    exclude_dirs : List[Path]
        List of paths to exclude.
    Returns
    -------
    dataset_content: Dict[str, Any]
        Dictionary with the parsed dataset's paths.
    """
    if not path.exists():
        logger.error(f"Path {path} does not exist.")
        return None

    dataset_content = {}
    dataset_content["cameras"] = []

    # iterate over the cameras
    for camera in path.iterdir():
        if not verify_camera_name(camera.name):
            continue

        camera_id = get_camera_id(camera)
        camera_content = {}
        camera_content["id"] = camera_id
        camera_content["sensors"] = {}
        dataset_content["cameras"].append(camera_content)

        # iterate over the sensors
        for sensor in camera.iterdir():
            if not verify_sensor_name(sensor.name):
                continue

            sensor_name = get_sensor_type(sensor)
            camera_content["sensors"][sensor_name] = []

            # iterate over the sensor's data
            for image_path in sensor.iterdir():
                if not image_path.with_suffix(".png") or image_path in exclude_dirs:
                    continue

                camera_content["sensors"][sensor_name].append(image_path)

            if len(camera_content["sensors"][sensor_name]) == 0:
                logger.warning(f"No data found for {camera_id} camera, {sensor_name} sensor")

        if not camera_content["sensors"]:
            logger.warning(f"No sensors found for {camera_id} camera")

    if len(dataset_content["cameras"]) == 0:
        logger.warning(f"No cameras found at {path}")

    return dataset_content


def calc_image_id(path: Path) -> int:
    """Calculates the image ID form its filename.

    Parameters
    ----------
    path : Path
        Path to the image.
    Returns
    -------
    image_id: int
        Image's ID number.
    """
    image_id = int(hashlib.sha256(path.name.encode("utf-8")).hexdigest(), 16) % 10**8
    return image_id
