#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import glob
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import coloredlogs
import numpy as np

from oscar_datagen_tools.common_utils import (
    get_camera_id,
    get_sensor_type,
    verify_image_filename,
    verify_run_name,
)

__all__ = [
    "build_projection_matrix",
    "calc_image_id",
    "filename_decode",
    "get_image_point",
    "load_dynamic_metadata",
    "load_sensors_static_metadata",
    "verify_dataset_path",
    "get_relative_path",
    "SENSORS_KEY",
    "CATEGORIES_KEY",
]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

# regex patterns
image_path_pattern = re.compile(
    r"[A-Za-z0-9_]+/[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]+-[0-9]+-[0-9]+-[0-9]+/route[0-9]+_[0-9]*\.[0-9]+z_[0-9]*\.[0-9]+deg\.[0-9]+\.png",
    re.IGNORECASE,
)  # path patter similar to: rgb/2023-07-12_19-53-06-033373/route244542543218049066126211912380029915139_31.42z_281.95deg.00000004.png

SENSORS_KEY = "sensors"
CATEGORIES_KEY = "instance_segmentation"


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

    # Get the paths of those images that located 2 levels deeper.
    # The expected path structure is: /<SENSOR>/<RUN_ID>/<IMAGE>.png
    path_pattern = path / "*/*/*.png"
    image_paths = glob.glob(str(path_pattern))

    dataset_content = {}
    dataset_content[SENSORS_KEY] = {}

    for image_path in image_paths:
        # get the 3 latest levels of the path
        relative_path = get_relative_path(Path(image_path), parent_level=3)
        # verify the format
        if not image_path_pattern.match(str(relative_path)):
            continue

        # the parent's parent is expected to be the sensor
        sensor = relative_path.parent.parent.name
        # the parent is expected to be the run
        run = relative_path.parent.name

        if sensor not in dataset_content[SENSORS_KEY]:
            dataset_content[SENSORS_KEY][sensor] = {}

        if run not in dataset_content[SENSORS_KEY][sensor]:
            dataset_content[SENSORS_KEY][sensor][run] = []

        dataset_content[SENSORS_KEY][sensor][run].append(Path(image_path))

    if not bool(dataset_content[SENSORS_KEY]):
        logger.warning(f"No sensors found at {path}")

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


def get_relative_path(path: Path, parent_level: int) -> Path:
    """Extracts a relative path from an absolute path.

    Parameters
    ----------
    path : Path
        Path to the image.
    parent_level : int
        Parent levels to extract.
    Returns
    -------
    relative_path: Path
        Relative path to image.
    """
    path_parents = list(path.parents)[:parent_level]
    return path.relative_to(*path_parents)


def filename_decode(name: str) -> Tuple[str, str]:
    """Decode the image filename to extract the camera ID and the frame number. The expected
    filename format is the following: route<CAMERA_ID>_<ELEVATION>z_<ANGLE>deg.0000000n.

    Parameters
    ----------
    name : str
        Image filename.
    Returns
    -------
    (camera_id, frame_num) : Tuple[str, str]
    """
    name_substring = name.split("_")

    # get the camera id
    assert "route" in name_substring[0]
    camera_id = name_substring[0]
    camera_id = camera_id.replace("route", "")

    # get frame number
    frame = name_substring[-1].split(".")
    frame = frame[-2]

    return camera_id, frame


def load_sensors_static_metadata(path: Path, sensor_type: str, camera_id: str) -> dict:
    """Load the static metadata for an specific camera and sensor type.

    Parameters
    ----------
    path : Path
        Path to the static metadata.
    sensor_type : str
        Sensor type (rgb, depth, ...)
    camera_id:
        Camera identifier.
    Returns
    -------
    sensor_metadata : dict
    """
    sensor = None
    with open(path) as metadata_file:
        data = json.load(metadata_file)
        sensors = data["cameras"][camera_id]["sensors"]

        for key in sensors.keys():
            if sensor_type in sensors[key]["bp_type"]:
                sensor = sensors[key]
                break

    return sensor


def load_dynamic_metadata(path: Path, sensor_type: str, camera_id: str) -> dict:
    """Load the dynamic metadata for an specific camera and sensor type.

    Parameters
    ----------
    path : Path
        Path to the static metadata.
    sensor_type : str
        Sensor type (rgb, depth, ...)
    camera_id:
        Camera identifier.
    Returns
    -------
    metadata : dict
    """
    metadata = {}

    with open(path) as metadata_file:
        data = json.load(metadata_file)
        sensors = data["cameras"][camera_id]["sensors"]

        for key in sensors.keys():
            if sensor_type in sensors[key]["type"]:
                sensor = sensors[key]
                metadata["sensor"] = sensor
                break

        metadata["patches"] = data["patches"]

    return metadata


def build_projection_matrix(w, h, fov):
    """Code taken from: https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/"""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    """Code taken from: https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/"""
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc[0], loc[1], loc[2], 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]
