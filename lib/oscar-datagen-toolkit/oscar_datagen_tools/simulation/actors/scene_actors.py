#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
from typing import List

import carla
import coloredlogs
import numpy as np
from numpy import random
from carla import AttachmentType, Transform

from ..context import Context
from .base import Actor
from .sensor import Sensor
from .controller_actors import WalkerController
from ...common_utils import get_world_point

__all__ = ["Walker", "Vehicle"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class SceneActor(Actor):
    KEY_VALUE = "actors"

    def __init__(self,
        *args,
        spawn_in_view_of_sensor: Sensor | list[Sensor] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._spawn_in_view_of_sensor = spawn_in_view_of_sensor[0] if isinstance(spawn_in_view_of_sensor, list) else spawn_in_view_of_sensor

    # FIXME: This feels like the wrong abstraction. The problem is we need to select different
    #        waypoints depending upon the kind of actor (vehicles -> roads, walkers -> sidewalks).
    def _random_waypoint_in_view_of_sensor(self, **waypoint_kwargs):
        sensor = self._spawn_in_view_of_sensor

        # Chose random image coordinate and project it to world coordinates using sensor transform
        x = random.uniform(0, sensor.image_size_x)
        y = random.uniform(0, sensor.image_size_y)
        x, y = get_world_point(
            np.array([[x, y]]),
            sensor.transform,
            sensor.image_size_x,
            sensor.image_size_y,
            sensor.fov
        )[0]

        # Find nearest waypoint and move actor there
        location = carla.Location(x=x, y=y, z=0)
        waypoint = self.context.world.get_map().get_waypoint(location, **waypoint_kwargs)
        transform = waypoint.transform
        transform.location.z = 0.6 # spawn point height

        return transform

    def get_static_metadata(self, **actor_data):
        actor_data["3D_BoundingBox"] = {}
        bbox = self.carla_actor.bounding_box
        actor_data["3D_BoundingBox"]["location"] = (
            bbox.location.x,
            bbox.location.y,
            bbox.location.z,
        )
        actor_data["3D_BoundingBox"]["rotation"] = (
            bbox.rotation.pitch,
            bbox.rotation.yaw,
            bbox.rotation.roll,
        )
        actor_data["3D_BoundingBox"]["extent"] = (bbox.extent.x, bbox.extent.y, bbox.extent.z)

        return super().get_static_metadata(**actor_data)


class Walker(SceneActor):
    def __init__(
        self,
        name: str = "*",
        **kwargs
    ) -> None:
        super().__init__(f"walker.{name}", **kwargs)

    def __pre_spawn__(self) -> bool:
        # Replace transform with transform in view of sensor
        if self._spawn_in_view_of_sensor:
            self.transform = self._random_waypoint_in_view_of_sensor(lane_type=carla.LaneType.Sidewalk)

        if self.transform is None:
            self.transform = self.context.get_spawn_point(from_navigation=True)

        return True

    def get_dynamic_metadata(self):
        actor_data = {}
        actor_data["bones"] = []
        bones = self.carla_actor.get_bones().bone_transforms
        for bone in bones:
            bone_data = {}
            bone_data["name"] = bone.name
            bone_data["world"] = {}
            bone_data["world"]["location"] = (
                bone.world.location.x,
                bone.world.location.y,
                bone.world.location.z,
            )
            bone_data["world"]["rotation"] = (
                bone.world.rotation.pitch,
                bone.world.rotation.yaw,
                bone.world.rotation.roll,
            )
            bone_data["component"] = {}
            bone_data["component"]["location"] = (
                bone.component.location.x,
                bone.component.location.y,
                bone.component.location.z,
            )
            bone_data["component"]["rotation"] = (
                bone.component.rotation.pitch,
                bone.component.rotation.yaw,
                bone.component.rotation.roll,
            )
            bone_data["relative"] = {}
            bone_data["relative"]["location"] = (
                bone.relative.location.x,
                bone.relative.location.y,
                bone.relative.location.z,
            )
            bone_data["relative"]["rotation"] = (
                bone.relative.rotation.pitch,
                bone.relative.rotation.yaw,
                bone.relative.rotation.roll,
            )
            actor_data["bones"].append(bone_data)

        return super().get_dynamic_metadata(**actor_data)


class Vehicle(SceneActor):
    def __init__(self,
        name: str = "*",
        color: str = "{},{},{}",
        **kwargs
    ) -> None:
        # Allow random color generation
        if color:
            kwargs["color"] = color.format(*random.randint(0, 255, 3))

        super().__init__(f"vehicle.{name}", **kwargs)

    def __pre_spawn__(self) -> bool:
        # Replace transform with transform in view of sensor
        if self._spawn_in_view_of_sensor:
            self.transform = self._random_waypoint_in_view_of_sensor(lane_type=carla.LaneType.Driving)

        if self.transform is None:
            self.transform = self.context.get_spawn_point(from_navigation=False)

        return True
