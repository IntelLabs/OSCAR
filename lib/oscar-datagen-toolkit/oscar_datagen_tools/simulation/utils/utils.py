# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import coloredlogs
from carla import WorldSnapshot, Transform
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
            except RuntimeError:
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
            except RuntimeError:
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


def __save_metadata_file__(metadata: dict, filename: str = "metadata_static.json"):
    """Auxiliary function used to save the collected metadata.

    Args:
        metadata (dict): Dictionary structure with the simulation's metadata.
        filename (str): Name of the metadata's file.

    Returns:
        None
    """
    metadata_path = Path(HydraConfig.get().run.dir) / "metadata"
    if not metadata_path.exists():
        metadata_path.mkdir()

    metadata_static_file = metadata_path / filename
    with open(metadata_static_file, "w") as json_file:
        json.dump(metadata, json_file, indent=1)


def save_static_metadata(actors: list[Actor]):
    """Collects the static metadata information for all the simulation's actors.

    Args:
        actors (list): List of all the simulation's actors.

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

    __save_metadata_file__(metadata_static)


def save_dynamic_metadata(actors: list[Actor], local_frame_num: int, snapshot: WorldSnapshot):
    """Collects the dynamic metadata information for all the simulation's actors.

    Args:
        actors (list): List of all the simulation's actors.
        local_frame_num (int): Frame that is been processed.
        snapshot (carla.WorldSnapshot): CARLA's simulation's snapshot.

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

    __save_metadata_file__(metadata_dynamic, f"{local_frame_num:08}.json")


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

    # If any of the paramenters differ more than the threshold value the
    # transformers are considered different.
    if (abs(transform_a.location.x - transform_b.location.x) > threshold or
        abs(transform_a.location.y - transform_b.location.y) > threshold or
        abs(transform_a.location.z - transform_b.location.z) > threshold or
        abs(transform_a.rotation.pitch - transform_b.rotation.pitch) > threshold or
        abs(transform_a.rotation.yaw - transform_b.rotation.yaw) > threshold or
        abs(transform_a.rotation.roll - transform_b.rotation.roll) > threshold):
        return False
    
    return True
