#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
from itertools import cycle
from typing import List

import carla
import coloredlogs
from numpy import random
from carla import AttachmentType, LaneType, Rotation, Transform

from ..context import Context
from ..parameters import MotionParameters
from ..utils import is_equal
from .base import Actor

__all__ = ["Controller", "VehicleController", "WalkerController", "SensorController"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    logger.warning("CARLA's agents API is not available.")


class Controller(Actor):
    KEY_VALUE = "controllers"

    def __pre_spawn__(self) -> bool:
        # Controllers are always attached to other actors and so should have an alive parent.
        return self.parent is not None and self.parent.is_alive

    @property
    def is_alive(self):
        return self.parent.is_alive

    def step(self) -> carla.command.ApplyTransform:
        return None

    def get_dynamic_metadata(self, **metadata) -> dict[dict[str, Any], str]:
        # Controllers have no dynamic metadata.
        return {self.KEY_VALUE: {}}


class VehicleController(Controller):
    TYPE = "controller.vehicle"

    def __init__(self, **kwargs):
        super().__init__(None, **kwargs)

    @property
    def id(self):
        return self.parent.id

    @property
    def name(self):
        return f"{self.TYPE}.{self.id}"

    def __post_spawn__(self) -> bool:
        # setup vehicle
        self.parent.carla_actor.set_autopilot(True, self.context.simulation_params.traffic_manager_port)
        self.parent.carla_actor.set_light_state(carla.VehicleLightState.NONE)

        return True


class WalkerController(Controller):
    def __init__(
        self,
        max_speed: float = None,
        **kwargs
    ) -> None:
        super().__init__("controller.ai.walker", **kwargs)

        if max_speed is None:
            max_speed = 1 + random.random() # between 1-2 m/s
        self.max_speed = max_speed

    @property
    def is_alive(self):
        # Since this controller is an Actor, we also need to return our actor's aliveness
        if self._carla_actor is None:
            return False
        return self._carla_actor.is_alive and self.parent.is_alive

    def __pre_spawn__(self) -> bool:
        # FIXME: Should these just go in __init__?
        # If we have no destination, then randomly generate one from navigation.
        if self.transform is None:
            self.transform = self.context.get_spawn_point(from_navigation=True)
        return super().__pre_spawn__()

    def __post_spawn__(self) -> bool:
        # setup walker's controller
        self.carla_actor.start()
        self.carla_actor.go_to_location(self.transform.location)
        self.carla_actor.set_max_speed(self.max_speed)

        return True

    def destroy(self) -> bool:
        self.carla_actor.stop()
        return super().destroy()

    def get_static_metadata(self):
        return super().get_static_metadata(max_speed=self.max_speed)


class SensorController(Controller):
    TYPE = "controller.sensor"

    def __init__(
        self,
        motion_params: MotionParameters = None,
        **kwargs,
    ) -> None:
        super().__init__(None, **kwargs)

        self._motion_params = motion_params

    @property
    def id(self):
        return self.parent.id

    @property
    def name(self):
        return f"{self.TYPE}.{self.id}"

    def __post_spawn__(self) -> bool:
        sim_map = self.context.world.get_map()

        # Get the nearest waypoints in the center of a Driving lane.
        start_waypoint = sim_map.get_waypoint(
            location=self.parent.transform.location,
            project_to_road=True,
            lane_type=self._motion_params.lane_type,
        )
        end_waypoint = sim_map.get_waypoint(
            location=self.transform.location,
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

    def step(self) -> carla.Transform:
        waypoint = next(self.route_iter, None)

        # build the next sensor position based on the route's waypoint
        # and sensor Z position and rotation.
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

        next_pitch = waypoint_transform.rotation.pitch + self.transform.rotation.pitch
        next_yaw = waypoint_transform.rotation.yaw + self.transform.rotation.yaw
        next_roll = waypoint_transform.rotation.roll + self.transform.rotation.roll
        next_rotation = carla.Rotation(
            pitch=next_pitch,
            yaw=next_yaw,
            roll=next_roll,
        )

        next_transform = carla.Transform(location=next_location, rotation=next_rotation)

        # update sensor transform
        self.parent.transform = next_transform

        # build the apply transform command
        command = carla.command.ApplyTransform(self.parent.carla_actor, next_transform)
        return command

    def destroy(self) -> bool:
        self.route = []  # FIXME: why is this necessary?
        return super().destroy()

    def get_static_metadata(self):
        return super().get_static_metadata(
            blueprint_id=self.TYPE, sampling_resolution=self._motion_params.sampling_resolution
        )
