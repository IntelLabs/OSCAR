# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import json
import logging
import multiprocessing
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import coloredlogs
import cv2
import numpy as np
from carla import Location, Transform, WorldSnapshot
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from scipy import ndimage

if TYPE_CHECKING:
    from ..actors import Actor

__all__ = [
    "actors_still_alive",
    "handle_keyboard_interrupt",
    "save_static_metadata",
    "save_dynamic_metadata",
    "is_jsonable",
    "is_equal",
    "replace_modality_in_path",
    "transform",
    "generate_uuid",
    "load_image",
]

RETRY = 5

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


def handle_keyboard_interrupt(func: Callable[[Any], Any]) -> Any:
    """Decorator to handle data collection loop.

    Args:
        func (function): Function to be wrapped by the decorator.

    Returns:
        None
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.info("Cancelled by user. Bye!")
            return False
        except Exception as error:
            logger.exception(error)
            return False

    return wrapper


def actors_still_alive(actors: list[Actor]):
    """Verify that all the actors are still alive in the running simulation.

    Args:
        actors (list): List of all simulation's actors.

    Returns:
        bool: True if all the actors are alive or False if at least one is not alive.
    """
    alive_actors = 0
    for actor in actors:
        if not actor.is_alive:
            logger.info(f"Actor {actor.name} is dead!")
        else:
            alive_actors += 1

    if alive_actors == len(actors):
        return True
    else:
        return False


def __save_metadata_file__(
    metadata: dict, filename: str = "metadata_static.json", path: Path = None
):
    """Auxiliary function used to save the collected metadata.

    Args:
        metadata (dict): Dictionary structure with the simulation's metadata.
        filename (str): Name of the metadata's file.
        path (Path): Path of metadata's file.

    Returns:
        None
    """
    if path is None:
        logger.error("Metadata path is None!")
        return

    if not path.exists():
        path.mkdir(parents=True)

    metadata_file = path / filename
    with open(metadata_file, "w") as json_file:
        json.dump(metadata, json_file, indent=1)


def save_static_metadata(actors: list[Actor], path: Path):
    """Collects the static metadata information for all the simulation's actors.

    Args:
        actors (list): List of all the simulation's actors.
        path (Path): Static metadata's path.

    Returns:
        None
    """
    # build the metadata
    metadata_static = defaultdict(dict)
    for actor in actors:
        actor_metadata = actor.get_static_metadata()
        # here is assumed that the root key correspond to the actor's category
        category = next(iter(actor_metadata))

        # include actor's metadata
        metadata_static[category].update(actor_metadata[category])

    __save_metadata_file__(metadata_static, path=path)


def save_dynamic_metadata(
    actors: list[Actor], local_frame_num: int, snapshot: WorldSnapshot, path: Path
):
    """Collects the dynamic metadata information for all the simulation's actors.

    Args:
        actors (list): List of all the simulation's actors.
        local_frame_num (int): Frame that is been processed.
        snapshot (carla.WorldSnapshot): CARLA's simulation's snapshot.
        path (Path): Dynamic metadata's path.

    Returns:
        None
    """
    # build the metadata
    metadata_dynamic = defaultdict(dict)
    metadata_dynamic["frame_id"] = local_frame_num
    metadata_dynamic["timestamp"] = snapshot.timestamp.elapsed_seconds
    for actor in actors:
        actor_metadata = actor.get_dynamic_metadata()
        # here is assumed that the root key correspond to the actor's category
        category = next(iter(actor_metadata))

        # include actor's metadata
        metadata_dynamic[category].update(actor_metadata[category])

    __save_metadata_file__(metadata_dynamic, filename=f"{local_frame_num:08}.json", path=path)


def is_jsonable(value: Any):
    """Evaluates if some value is JSON serializable.

    Args:
        value (Any): Value to evaluate.

    Returns:
        bool: True if the value is JSON serializable or False otherwise.
    """
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False


def is_equal(transform_a: Transform, transform_b: Transform, threshold: float = 0.5) -> bool:
    """Compare two different Transforms to verify if they are equal.

    Args:
        transform_a (carla.Transform): First transform.
        transform_b (carla.Transform): Second transform.
        threshold (float): Tolerance value.

    Returns:
        bool: True if the transformers are equal or Flase otherwise.
    """

    # If any of the parameters differ more than the threshold value the
    # transformers are considered different.
    if (
        abs(transform_a.location.x - transform_b.location.x) > threshold or
        abs(transform_a.location.y - transform_b.location.y) > threshold or
        abs(transform_a.location.z - transform_b.location.z) > threshold or
        abs(transform_a.rotation.pitch - transform_b.rotation.pitch) > threshold or
        abs(transform_a.rotation.yaw - transform_b.rotation.yaw) > threshold or
        abs(transform_a.rotation.roll - transform_b.rotation.roll) > threshold
    ):
        return False

    return True


def replace_modality_in_path(path: Path, modality: str) -> str:
    """Replace the given modality in the path. Is expected that the modality path contains the
    'modality_type' tag.

    Args:
        path (Path): Path to modify.
        modality (str): Modality type to include in the path.

    Returns:
        str: Path' string with the desired modality.
    """
    return str(path).format(modality_type=modality)


def transform(object_transform: Transform, center_transform: Transform) -> Transform:
    """Transform an object location and rotation using another Transform object as reference.

    Args:
        object_transform (Transform): Object location and rotation.
        center_transform (Transform): Reference transform.

    Returns:
        Transform: new transform.
    """
    # get a 3D affine transformation matrix
    matrix_transform = center_transform.get_matrix()

    # extract the center and object point's locations
    center = np.array(
        [
            [center_transform.location.x],
            [center_transform.location.y],
            [center_transform.location.z],
            [1],
        ]
    )
    point = np.array(
        [
            [object_transform.location.x],
            [object_transform.location.y],
            [object_transform.location.z],
            [1],
        ]
    )

    # calculate delta axis from center point
    center_result = np.matmul(matrix_transform, center)
    center_result = center_result.T[0]
    delta_x = center_result[0] - center_transform.location.x
    delta_y = center_result[1] - center_transform.location.y
    delta_z = center_result[2] - center_transform.location.z

    # apply transform to object point
    result = np.matmul(matrix_transform, point)
    result = result.T[0]

    # build a new transform
    x = result[0] - delta_x
    y = result[1] - delta_y
    z = result[2] - delta_z
    new_transform = Transform(location=Location(x=x, y=y, z=z), rotation=object_transform.rotation)

    return new_transform


def generate_uuid(seed: int = None) -> int:
    """Generate a UUID identifier in a deterministic way by specifying an integer value randomly
    generated with numpy.

    Args:
        seed : seed to reset the state of the RNG.
    Returns:
        UUID identifier.
    """
    if seed:
        np.random.seed(seed)

    # Generate four 32-bit integers and combine them to form a 128-bit integer
    random_bits = [np.random.randint(0, 2**32) for _ in range(4)]
    random_uuid_int = sum(bit << (32 * i) for i, bit in enumerate(random_bits))

    return int(uuid.UUID(int=random_uuid_int))


def load_image(path: Path) -> ndimage:
    """Loads an image with OpenCV.

    Parameters:
        path : Path
            Path to image.
    Returns
        Loaded image.
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
