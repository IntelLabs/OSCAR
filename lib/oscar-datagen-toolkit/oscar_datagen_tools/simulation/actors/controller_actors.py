#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
from itertools import cycle
from typing import List

import carla
import coloredlogs
from carla import AttachmentType, LaneType, Rotation, Transform

from ..blueprints import WalkerAIController as BPWalkerAIController
from ..context import Context
from ..parameters import MotionParameters
from ..utils import is_equal
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
        attachments: list[carla.Actor] = None,
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
        attachments: list[carla.Actor] = None,
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
        attachments: list[carla.Actor] = None,
        attachment_type: AttachmentType = AttachmentType.Rigid,
        motion_params: MotionParameters = None,
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

        self._motion_params = motion_params
        self._rotation = self.transform.rotation if self.transform else Rotation()

    @property
    def id(self):
        if self.parent is not None:
            return self.parent.id

        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def name(self):
        return f"{self.TYPE}.{self.id}"

    @name.setter
    def name(self, value):
        # ignore new values to name attribute. This attribute depends on the
        # actor's id
        pass

    def __pre_spawn__(self) -> bool:
        assert self.parent is not None

        # set the sensor's position
        self.transform = self.parent.transform
        self.destination_transform = self.parent.destination_transform
        self._rotation = self.transform.rotation

        return True

    def spawn(self) -> bool:
        if not self.__pre_spawn__():
            logger.error("Pre-spawn process failed.")
            return False

        sim_map = self.context.world.get_map()

        # Get the nearest waypoints in the center of a Driving lane.
        start_waypoint = sim_map.get_waypoint(
            location=self.transform.location,
            project_to_road=True,
            lane_type=self._motion_params.lane_type,
        )
        end_waypoint = sim_map.get_waypoint(
            location=self.destination_transform.location,
            project_to_road=True,
            lane_type=self._motion_params.lane_type,
        )

        grp = GlobalRoutePlanner(sim_map, self._motion_params.sampling_resolution)

        # build list of route waypoints. Add random points between the start and end points
        waypoints = [start_waypoint]
        points = self._motion_params.points
        for point in points:
            waypoint = sim_map.get_waypoint(
                location=point.location,
                project_to_road=True,
                lane_type=self._motion_params.lane_type,
            )
            waypoints.append(waypoint)
        waypoints.append(end_waypoint)

        # trace the route following the waypoints
        self.route = []
        for waypoint, next_waypoint in zip(waypoints, waypoints[1:]):
            subroute = grp.trace_route(
                waypoint.transform.location, next_waypoint.transform.location
            )

            # merge the different subroutes.
            # there is an overlap between the last element of one subroute and the first
            # element of the next subrout.
            index = 1 if len(self.route) > 0 else 0
            self.route += subroute[index:]

        if not self.__post_spawn__():
            logger.error("Post-spawn process failed.")
            return False

        return True

    def __post_spawn__(self) -> bool:
        # validate generated route
        # remove consecutive points that are too close
        # NOTE: the reason to do this filtering to the generated route is due to the
        # agents.navigation.global_route_planner module that sometimes generates two
        # consecutive points too close to each other. A more robust fix is by reviewn
        # the mentioned module.
        indexes_to_remove = []
        for index, (waypoint, next_waypoint) in enumerate(zip(self.route, self.route[1:])):
            waypoint_transform = waypoint[0].transform
            next_waypoint_transform = next_waypoint[0].transform

            if is_equal(waypoint_transform, next_waypoint_transform):
                # index + 1 -> corresponds to the next_waypoint index
                indexes_to_remove = [index + 1] + indexes_to_remove

        # remove the waypoints identified as duplications
        for index in indexes_to_remove:
            self.route.pop(index)

        self.route_iter = cycle(self.route)

        return True

    def spawn_command(self, parent: carla.Actor = None) -> carla.Command:
        # SensorController is not spawned via CARLA
        return None

    def step(self) -> carla.Transform:
        waypoint = next(self.route_iter, None)

        # build the next sensor position based on the route's waypoint
        # and sensor Z position and rotation.
        if self.parent is not None:
            sensor_transform = self.parent.transform
            waypoint_transform = waypoint[0].transform

            # the motion parameter's elevation value overrides the transformer's Z value
            elevation = (
                sensor_transform.location.z
                if self._motion_params.elevation == 0
                else self._motion_params.elevation
            )

            next_location = carla.Location(
                x=waypoint_transform.location.x,
                y=waypoint_transform.location.y,
                z=elevation,
            )

            next_pitch = waypoint_transform.rotation.pitch + self._rotation.pitch
            next_yaw = waypoint_transform.rotation.yaw + self._rotation.yaw
            next_roll = waypoint_transform.rotation.roll + self._rotation.roll
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
        actor_data["sampling_resolution"] = self._motion_params.sampling_resolution

        return (actor_data, KEY_VALUE)

    def get_dynamic_metadata(self):
        actor_data = {}
        actor_data["type"] = self.TYPE

        return (actor_data, KEY_VALUE)
