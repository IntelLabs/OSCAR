#
# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import logging
from itertools import cycle
from typing import List

import carla
import coloredlogs
from carla import AttachmentType, LaneType, Transform

from ..blueprints import WalkerAIController as BPWalkerAIController
from ..context import Context
from .base import Actor

__all__ = ["Controller", "WalkerController", "SensorController"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    logger.warning("CARLA's agents API is not available.")

KEY_VALUE = "controllers"


class Controller(Actor):
    def __init__(
        self,
        context: Context = Context(),
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

    def get_static_metadata(self):
        actor_data = {}
        actor_data["type"] = self.carla_actor.type_id

        return (actor_data, KEY_VALUE)

    def get_dynamic_metadata(self):
        actor_data, _ = super().get_dynamic_metadata()

        return (actor_data, KEY_VALUE)


class WalkerController(Controller):
    def __init__(
        self,
        context: Context = Context(),
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

        self.blueprint = BPWalkerAIController(*args, **kwargs)

    def __pre_spawn__(self):
        self.transform = Transform()

        if self.destination_transform is None:
            self.destination_transform = self.context.get_spawn_point(from_navigation=True)
            logger.debug(
                f"Selected random destination point (none provided) x: {self.destination_transform.location.x}, "
                f"y: {self.destination_transform.location.y}, z: {self.destination_transform.location.z}"
            )

        assert isinstance(self.blueprint, BPWalkerAIController)
        return True

    def __post_spawn__(self) -> bool:
        if not super().__post_spawn__():
            return False

        # setup walker's controller
        self.carla_actor.start()
        self.carla_actor.go_to_location(self.destination_transform.location)
        self.carla_actor.set_max_speed(self.blueprint.max_speed)

        return True

    def destroy(self) -> bool:
        self.carla_actor.stop()
        if not super().destroy():
            return False

        return True

    def get_static_metadata(self):
        actor_data, _ = super().get_static_metadata()
        actor_data["max_speed"] = self.blueprint.max_speed

        return (actor_data, KEY_VALUE)


class SensorController(Controller):
    TYPE = "controller.sensor"

    def __init__(
        self,
        context: Context = Context(),
        transform: Transform = None,
        destination_transform: Transform = None,
        attachments: List[carla.Actor] = [],
        attachment_type: AttachmentType = AttachmentType.Rigid,
        sampling_resolution: int = 2,
        lane_type: LaneType = LaneType.Driving,
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

        self.sampling_resolution = sampling_resolution
        self.lane_type = lane_type

    def spawn(self) -> bool:
        if not self.__pre_spawn__():
            logger.error("Pre-spawn process failed.")
            return False

        sim_map = self.context.world.get_map()

        # Get the nearest waypoints in the center of a Driving lane.
        start_waypoint = sim_map.get_waypoint(
            location=self.transform.location,
            project_to_road=True,
            lane_type=self.lane_type,
        )
        end_waypoint = sim_map.get_waypoint(
            location=self.destination_transform.location,
            project_to_road=True,
            lane_type=self.lane_type,
        )

        grp = GlobalRoutePlanner(sim_map, self.sampling_resolution)
        self.route = grp.trace_route(
            start_waypoint.transform.location, end_waypoint.transform.location
        )
        self.route_iter = cycle(self.route)

        if not self.__post_spawn__():
            logger.error("Post-spawn process failed.")
            return False

        return True

    def __post_spawn__(self) -> bool:
        if self.parent is not None:
            self.id = self.parent.id
            self.name = f"{self.TYPE}.{self.parent.id}"
            return True
        else:
            logger.error("Sensor controller must be attached to a sensor!")
            return False

    def step(self) -> carla.Transform:
        waypoint = next(self.route_iter, None)

        # build the next sensor position based on the route's waypoint
        # and sensor Z position and rotation.
        if self.parent is not None:
            sensor_transform = self.transform
            waypoint_transform = waypoint[0].transform

            next_location = carla.Location(
                x=waypoint_transform.location.x,
                y=waypoint_transform.location.y,
                z=sensor_transform.location.z,
            )

            next_pitch = waypoint_transform.rotation.pitch + sensor_transform.rotation.pitch
            next_yaw = waypoint_transform.rotation.yaw + sensor_transform.rotation.yaw
            next_roll = waypoint_transform.rotation.roll + sensor_transform.rotation.roll
            next_rotation = carla.Rotation(
                pitch=next_pitch,
                yaw=next_yaw,
                roll=next_roll,
            )

            next_transform = carla.Transform(location=next_location, rotation=next_rotation)
            return next_transform
        else:
            return None

    def is_alive(self) -> bool:
        return True

    def destroy(self) -> bool:
        self.route = []
        return True

    def get_static_metadata(self):
        actor_data = {}
        actor_data["type"] = self.TYPE
        actor_data["sampling_resolution"] = self.sampling_resolution

        return (actor_data, KEY_VALUE)

    def get_dynamic_metadata(self):
        actor_data = {}
        actor_data["type"] = self.TYPE

        return (actor_data, KEY_VALUE)
