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
    "verify_camera_name",
    "verify_image_filename",
    "verify_sensor_name",
]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

# regex patterns
camera_pattern = re.compile(r"sensor\.camera\.[0-9]+", re.IGNORECASE)  # sensor.camera.<ID>
filename_pattern = re.compile(r"[0-9]+\.png", re.IGNORECASE)  # 0000000n.png
sensor_name_pattern = re.compile(r"[A-Za-z_]+\.[0-9]+", re.IGNORECASE)  # <TYPE>.<ID>

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
    """Adjust the format of the sensor name, where the expected format is: <TYPE>.<ID>

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
    sensor_id = name.split(".")[SENSOR_ID]
    return f"{sensor_type}.{sensor_id}"


def generate_image_filename(index: int) -> str:
    """Creates an image filename with the format: 0000000n.png.

    Parameters
    ----------
    name : index
        Image index.
    Returns
    -------
    image_name: str
        Image's formatted name.
    """
    return f"{index:08}"


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
    """Gets the image ID from it's name.

    Parameters
    ----------
    path : Path
        Path to image's file.
    Returns
    -------
    image_id: int
        Image's identifier.
    """

    # The expected filename follows the format "000000n.png"
    return int(path.stem)


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
    return path.name.split(".")[SENSOR_TYPE]


def verify_camera_name(name: str) -> bool:
    """Verify if the camera name follows the
    format: sensor.camera.<ID>.

    Parameters
    ----------
    name : str
        Camera's name.
    Returns
    -------
    result: bool
        True if the name is correct, False otherwise.
    """
    if not camera_pattern.match(name):
        logger.warning(f"Camera name {name} does not follow the format: sensor.camera.<ID>")
        return False

    return True


def verify_image_filename(filename: str) -> bool:
    """Verify if the image name follows the
    format: 0000000n.png.

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
        logger.warning(f"Filename {filename} does not follow the format: 000000n.png")
        return False

    return True


def verify_sensor_name(name: str) -> bool:
    """Verify if the sensor name follows the
    format: <TYPE>.<ID>.

    Parameters
    ----------
    name : str
        Sensor's name.
    Returns
    -------
    result: bool
        True if the name is correct, False otherwise.
    """
    if not sensor_name_pattern.match(name):
        logger.warning(f"Sensor name {name} does not follow the format: <TYPE>.<ID>")
        return False

    return True
