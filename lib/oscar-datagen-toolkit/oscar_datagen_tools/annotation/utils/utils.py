#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import glob
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import coloredlogs

from oscar_datagen_tools.common_utils import (
    get_camera_id,
    get_sensor_type,
    verify_image_filename,
    verify_run_name,
)

__all__ = [
    "calc_image_id",
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
