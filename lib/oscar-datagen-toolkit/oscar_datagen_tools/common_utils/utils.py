#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import re
from pathlib import Path

import coloredlogs

__all__ = [
    "format_camera_name",
    "format_sensor_name",
    "generate_image_filename",
    "get_camera_id",
    "get_image_id",
    "get_sensor_type",
    "verify_run_name",
    "verify_image_filename",
]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

# regex patterns
run_pattern = re.compile(
    r"[0-9]+-[0-9]+-[0-9]+_[0-9]+-[0-9]+-[0-9]+-[0-9]+", re.IGNORECASE
)  # Y-m-d_H-M-S-f
filename_pattern = re.compile(
    r"route[0-9]+_[0-9]*\.[0-9]+z_[0-9]*\.[0-9]+deg\.[0-9]+\.png", re.IGNORECASE
)  # route<CAMERA_ID>_<Z_LOCATION>z_<PITCH>deg.0000000n.png

# camera constants
CAMERA_ID = -1

# unformatted sensor directory name: camera.sensor.<TYPE>.<ID>
SENSOR_TYPE = -2
SENSOR_ID = -1


def format_camera_name(camera_id: int) -> str:
    """Adjust the format of the camera name, where the expected format is: sensor.camera.<ID>

    Parameters
    ----------
    camera_id : id
        Cameras's unique identifier.
    Returns
    -------
    camera_name: str
        Camera name.
    """
    return f"sensor.camera.{camera_id}"


def format_sensor_name(name: str) -> str:
    """Adjust the format of the sensor name, where the expected format is: <TYPE>

    Parameters
    ----------
    name : id
        Sensor's name
    Returns
    -------
    sensor_name: str
        Sensor's formatted name.
    """
    sensor_type = name.split(".")[SENSOR_TYPE]
    return sensor_type


def generate_image_filename(camera_id: int, z_location: float, pitch: float, index: int) -> str:
    """Creates an image filename with the format: 0000000n.png.

    Parameters
    ----------
    camera_id : int
        Camera ID.
    z_location: float
        Camera's elevation
    pitch: float
        Camera's pitch angle
    index: int
        Frame index
    Returns
    -------
    image_name: str
        Image's formatted name.
    """
    return f"route{camera_id}_{z_location:.2f}z_{pitch:.2f}deg.{index:08}.png"


def get_camera_id(path: Path) -> int:
    """Gets the camera ID from it's name.

    Parameters
    ----------
    path : Path
        Path to sensor's directory.
    Returns
    -------
    camera_id: int
        Camera's identifier.
    """
    return int(path.name.split(".")[CAMERA_ID])


def get_image_id(path: Path) -> int:
    """Gets the image ID from it's filename.

    Parameters
    ----------
    path : Path
        Path to image's file.
    Returns
    -------
    image_id: str
        Image's identifier.
    """

    # The expected filename follows the format "route<CAMERA_ID>_<Z_LOCATION>z_<PITCH>deg.0000000n.png"
    return path.stem


def get_sensor_type(path: Path) -> str:
    """Gets the sensor' type name.

    Parameters
    ----------
    path : Path
        Path to sensor's directory.
    Returns
    -------
    sensor_type: str
        Sensor's type name.
    """
    return path.name


def verify_run_name(name: str) -> bool:
    """Verify if the run name follows the
    format: Y-m-d_H-M-S-f.

    Parameters
    ----------
    name : str
        Run's name.
    Returns
    -------
    result: bool
        True if the name is correct, False otherwise.
    """
    if not run_pattern.match(name):
        logger.warning(f"Run name {name} does not follow the format: Y-m-d_H-M-S-f")
        return False

    return True


def verify_image_filename(filename: str) -> bool:
    """Verify if the image name follows the
    format: route<CAMERA_ID>_<Z_LOCATION>z_<PITCH>deg.0000000n.png.

    Parameters
    ----------
    name : str
        Image's name.
    Returns
    -------
    result: bool
        True if the name is correct, False otherwise.
    """
    if not filename_pattern.match(filename):
        logger.warning(
            f"Filename {filename} does not follow the format: route<CAMERA_ID>_<Z_LOCATION>z_<PITCH>deg.0000000n.png"
        )
        return False

    return True
