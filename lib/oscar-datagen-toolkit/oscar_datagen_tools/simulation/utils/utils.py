# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import coloredlogs
from carla import Transform, WorldSnapshot
from hydra.core.hydra_config import HydraConfig

if TYPE_CHECKING:
    from ..actors import Actor

__all__ = [
    "actors_still_alive",
    "run_loop",
    "run_request",
    "run_spawn",
    "save_static_metadata",
    "save_dynamic_metadata",
    "is_jsonable",
    "is_equal",
    "replace_modality_in_path",
]

RETRY = 5

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


def run_request(func: Callable[[Any], Any]) -> bool:
    """Decorator to handle the requests to the CARLA server.

    Args:
        func (function): Function to be wrapped by the decorator.

    Returns:
        bool: True if the function was executed correctly or False otherwise.
    """

    def wrapper(*args):
        instance = args[0]
        retry = instance.client_params.retry

        for i in range(retry):
            try:
                func(*args)
                return True
            except RuntimeError as error:
                logger.exception(error)
                logger.error(f"CARLA connection failed on attempt {i + 1} of {retry}")
                time.sleep(5)

        return False

    return wrapper


def run_spawn(func: Callable[[Any], Any]) -> bool:
    """Decorator to handle the spawn of actors.

    Args:
        func (function): Function to be wrapped by the decorator.

    Returns:
        bool: True if the function was executed correctly or False otherwise.
    """

    def wrapper(*args):
        instance = args[0]

        for i in range(RETRY):
            try:
                return func(*args)
            except RuntimeError as error:
                logger.exception(error)
                logger.debug(
                    f"Spawn process failed with actor {instance} on attempt {i + 1} of {RETRY}"
                )

                # reset actor's transforms
                # Note: the most common reason for this process to fail is when
                # traying to spwan in a location too close to another actor.
                instance.transform = None
                instance.destination_transform = None

        return False

    return wrapper


def run_loop(func: Callable[[Any], Any]) -> Any:
    """Decorator to handle data collection loop.

    Args:
        func (function): Function to be wrapped by the decorator.

    Returns:
        None
    """

    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.info("Cancelled by user. Bye!")
        except Exception as error:
            logger.exception(error)

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
        if not actor.is_alive():
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

    metadata_static_file = path / filename
    with open(metadata_static_file, "w") as json_file:
        json.dump(metadata, json_file, indent=1)


def save_static_metadata(actors: list[Actor], path: Path):
    """Collects the static metadata information for all the simulation's actors.

    Args:
        actors (list): List of all the simulation's actors.
        path (Path): Static metadata's path.

    Returns:
        None
    """
    metadata_static = {}
    metadata_static["actors"] = {}
    metadata_static["cameras"] = {}
    metadata_static["controllers"] = {}

    for actor in actors:
        metadata, key_value = actor.get_static_metadata()
        metadata_static[key_value][actor.id] = metadata

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
    metadata_dynamic = {}
    metadata_dynamic["frame_id"] = local_frame_num
    metadata_dynamic["timestamp"] = snapshot.timestamp.elapsed_seconds
    metadata_dynamic["actors"] = {}
    metadata_dynamic["cameras"] = {}
    metadata_dynamic["controllers"] = {}

    for actor in actors:
        metadata, key_value = actor.get_dynamic_metadata()
        metadata_dynamic[key_value][actor.id] = metadata

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
