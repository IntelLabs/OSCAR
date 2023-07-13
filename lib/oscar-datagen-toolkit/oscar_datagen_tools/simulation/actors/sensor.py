#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import abc
import logging
import math
import time
import uuid
from pathlib import Path
from typing import List

import carla
import coloredlogs
from carla import AttachmentType, Location, SensorData, Transform
from numpy import random

from oscar_datagen_tools.common_utils import format_camera_name, format_sensor_name

from ..blueprints import InstanceSegmentation as BPInstanceSegmentation
from ..blueprints import Sensor as BPSensor
from ..context import Context
from ..parameters import MotionParameters
from ..utils import is_jsonable, replace_modality_in_path
from .base import Actor
from .controller_actors import SensorController

__all__ = ["Camera", "ISensor", "Sensor"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class ISensor(abc.ABC):
    @abc.abstractmethod
    def start_listening(self) -> None:
        pass

    @abc.abstractmethod
    def stop_listening(self) -> None:
        pass

    @abc.abstractmethod
    def wait_sensors_data(self, frame: int, timeout: int):
        pass

    @abc.abstractmethod
    def step(self, local_time_step: int) -> None:
        pass

    @abc.abstractmethod
    def save_to_disk(self, path: Path) -> None:
        pass


class Sensor(Actor, ISensor):

    KEY_VALUE = "sensors"

    def __init__(
        self,
        blueprint: BPSensor = None,
        context: Context = Context(),
        motion_params: MotionParameters = None,
        transform: Transform = None,
        destination_transform: Transform = None,
        attachments: List[carla.Actor] = [],
        attachment_type: AttachmentType = AttachmentType.Rigid,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            context,
            transform,
            destination_transform,
            attachments,
            attachment_type,
            *args,
            **kwargs,
        )

        self.motion_params = motion_params or MotionParameters()
        self.blueprint = blueprint
        self.data = None

        self.controller = SensorController(
            transform=self.transform,
            destination_transform=self.destination_transform,
            motion_params=motion_params,
        )
        self.controller.parent = self
        self.attachments.append(self.controller)

    def __pre_spawn__(self) -> bool:
        if not super().__pre_spawn__():
            return False

        assert isinstance(self.blueprint, BPSensor)
        return True

    def __post_spawn__(self) -> bool:
        if not super().__post_spawn__():
            return False

        self.name = format_sensor_name(self.name)

        return True

    def __on_listen__(self, data: SensorData) -> None:
        logger.debug(f"Received new data for sensor: {self.name}")
        self.data = data

    def start_listening(self) -> None:
        assert self.carla_actor is not None
        # register listening callback
        self.carla_actor.listen(lambda data: self.__on_listen__(data))

    def stop_listening(self) -> None:
        assert self.carla_actor is not None
        self.carla_actor.stop()

    def wait_sensors_data(self, frame: int, timeout: int):
        # wait for sensor data
        start_time = time.time()
        while self.data is None or self.data.frame != frame:
            logger.debug(f"Waiting for sensor {self.name}'s data...")
            time.sleep(0.05)

            # check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                logger.warning(f"Timeout reached in sensor {self.name}")
                break

    def step(self) -> carla.command.ApplyTransform:
        # If the sensor is attached to another actor
        # this step won't be performed
        if self.parent is not None:
            return

        # update sensor's location
        command = None
        new_sensor_transform = self.controller.step()
        if new_sensor_transform is not None and self.carla_actor is not None:
            self.transform = new_sensor_transform
            command = carla.command.ApplyTransform(self.carla_actor, self.transform)

        return command

    def save_to_disk(self, path: str) -> None:
        if self.data is None:
            logger.error("Sensor's data is None!")
            return

        self.data.save_to_disk(path)
        self.data = None

    def get_static_metadata(self):
        sensor_data = {}

        for attribute, value in self.blueprint.__dict__.items():
            if is_jsonable(value):
                sensor_data[attribute] = value

        return (sensor_data, self.KEY_VALUE)

    def get_dynamic_metadata(self):
        sensor_data, _ = super().get_dynamic_metadata()

        return (sensor_data, self.KEY_VALUE)


class Camera(Actor, ISensor):

    KEY_VALUE = "cameras"

    def __init__(
        self,
        context: Context = Context(),
        motion_params: MotionParameters = None,
        transform: Transform = None,
        destination_transform: Transform = None,
        sensors_blueprints: List[BPSensor] = [],
        attachments: List[carla.Actor] = [],
        attachment_type: AttachmentType = AttachmentType.Rigid,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            context,
            transform,
            destination_transform,
            attachments,
            attachment_type,
            *args,
            **kwargs,
        )

        self.id = int(uuid.uuid1())
        self.carla_actor = None
        self.parent = None
        self.motion_params = motion_params or MotionParameters()

        assert len(sensors_blueprints) > 0, "The camera must have at least one sensor!"

        # check if the user included an instance segmentation
        segmentations = list(
            filter(lambda bp: isinstance(bp, BPInstanceSegmentation), sensors_blueprints)
        )

        if len(segmentations) == 0:
            logger.warning(
                "Camera does not have a segmentation sensor. One will be added automatically."
            )
            segmentation_blueprint = BPInstanceSegmentation(*args, **kwargs)
            sensors_blueprints.append(segmentation_blueprint)

        # verify and init sensors
        self.sensors = []
        for sensor_blueprint in sensors_blueprints:
            assert isinstance(sensor_blueprint, BPSensor)

            # set general camera's attributes
            sensor_blueprint.__dict__.update(kwargs)

            sensor = Sensor(
                sensor_blueprint,
                context,
                motion_params,
                transform,
                destination_transform,
                attachments,
                attachment_type,
                *args,
                **kwargs,
            )
            self.sensors.append(sensor)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

        if not hasattr(self, "sensors"):
            return

        # forward the camera's parent to its sensors
        for sensor in self.sensors:
            sensor.parent = value

    @property
    def carla_actor(self):
        if not hasattr(self, "sensors"):
            return None

        if len(self.sensors) > 0 and self._carla_actor is None:
            # randomly select one of the sensors to return
            # its spawn CARLA's actor
            sensor = random.choice(self.sensors)
            self._carla_actor = sensor.carla_actor

        return self._carla_actor

    @carla_actor.setter
    def carla_actor(self, value):
        self._carla_actor = value

    def __pre_spawn__(self) -> bool:
        if not super().__pre_spawn__():
            return False

        assert len(self.sensors) > 0
        return True

    def __post_spawn__(self) -> bool:
        self.name = format_camera_name(self.id)
        return True

    def spawn(self) -> bool:
        if not self.__pre_spawn__():
            logger.error("Pre-spawn process failed.")
            return False

        for sensor in self.sensors:
            # update sensors's transforms
            sensor.transform = self.transform
            sensor.destination_transform = self.destination_transform

            if not sensor.spawn():
                return False

        if not self.__post_spawn__():
            logger.error("Post-spawn process failed.")
            return False

        return True

    def start_listening(self) -> None:
        for sensor in self.sensors:
            sensor.start_listening()

    def stop_listening(self) -> None:
        for sensor in self.sensors:
            sensor.stop_listening()

    def wait_sensors_data(self, frame: int, timeout: int):
        for sensor in self.sensors:
            sensor.wait_sensors_data(frame, timeout)

    def step(self) -> None:
        commands = []
        for sensor in self.sensors:
            command = sensor.step()

            if command is not None:
                commands.append(command)

        # update camera position
        self.context.apply_batch_sync(commands)
        self.transform = self.sensors[0].transform

    def save_to_disk(self, path: Path) -> None:
        # check if the camera sensors are at the same position
        res = all(
            sensor.get_transform() == self.sensors[0].get_transform() for sensor in self.sensors
        )
        if not res:
            logger.warning(
                "Not all the camera's sensors have the same position. Data of frame won't be stored."
            )
            return

        for sensor in self.sensors:
            # specify intermediate directories for each of the sensors.
            sensor_path = replace_modality_in_path(path, modality=sensor.name)
            sensor.save_to_disk(sensor_path)

    def is_alive(self) -> bool:
        alive_sensors = 0
        for sensor in self.sensors:
            if not sensor.is_alive():
                logger.info(f"Sensor {sensor.name} is dead!")
            else:
                alive_sensors += 1

        if alive_sensors == len(self.sensors):
            return True
        else:
            return False

    def destroy(self) -> bool:
        for sensor in self.sensors:
            if not sensor.destroy():
                return False

        return True

    def get_static_metadata(self):
        camera_data = {}
        camera_data["sensors"] = {}

        for sensor in self.sensors:
            metadata, key_value = sensor.get_static_metadata()
            camera_data[key_value][sensor.carla_actor.id] = metadata

        return (camera_data, self.KEY_VALUE)

    def get_dynamic_metadata(self):
        camera_data = {}
        camera_data["sensors"] = {}

        for sensor in self.sensors:
            metadata, key_value = sensor.get_dynamic_metadata()
            camera_data[key_value][sensor.carla_actor.id] = metadata

        return (camera_data, self.KEY_VALUE)
