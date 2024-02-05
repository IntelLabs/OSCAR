#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import math
import time
import weakref

import carla
import coloredlogs
from carla import AttachmentType, Location, SensorData, Transform
from hydra.utils import instantiate
from numpy import random

from oscar_datagen_tools.common_utils import format_sensor_name

from ..context import Context
from ..parameters import MotionParameters
from ..utils import generate_uuid, replace_modality_in_path
from .base import Actor
from .controller_actors import SensorController

__all__ = ["Camera", "Sensor"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class Sensor(Actor):

    KEY_VALUE = "sensors"

    def __init__(
        self,
        name: str,
        **kwargs,
    ) -> None:
        super().__init__(f"sensor.{name}", **kwargs)

        self.id = generate_uuid()
        self.data = None

    def __post_spawn__(self) -> bool:
        self.name = format_sensor_name(self.name)
        return True

    @staticmethod
    def __on_listen__(weak_self, data: SensorData) -> None:
        self = weak_self()
        self.data = data

    def start_listening(self) -> None:
        # reset data
        self.data = None
        # register listening callback
        weak_self = weakref.ref(self)
        self.carla_actor.listen(lambda data: weak_self().__on_listen__(weak_self, data))

    def stop_listening(self) -> None:
        self.carla_actor.stop()

    def wait_sensors_data(self, frame: int, timeout: int):
        # wait for sensor data
        start_time = time.time()
        while self.data is None or self.data.frame != frame:
            logger.debug(f"Waiting for sensor {self.name}'s data...")

            # check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                logger.warning(f"Timeout reached in sensor {self.name}")
                break

    def save_to_disk(self, path: str) -> None:
        if self.data is None:
            logger.error("Sensor's data is None!")
            return

        path = replace_modality_in_path(path, modality=self.name)
        self.data.save_to_disk(path)
        self.data = None


class Camera:
    def __new__(
        cls,
        modalities: list[str],
        **kwargs,
    ) -> None:
        # check if the user included an instance segmentation
        if "instance_segmentation" not in modalities:
            logger.warning(
                "Camera does not have a segmentation sensor. One will be added automatically."
            )
            modalities.append("instance_segmentation")

        # verify and init sensors
        sensors = []
        for modality in modalities:
            new_kwargs = instantiate(kwargs)
            sensors.append(Sensor(f"camera.{modality}", **new_kwargs))
        return sensors
