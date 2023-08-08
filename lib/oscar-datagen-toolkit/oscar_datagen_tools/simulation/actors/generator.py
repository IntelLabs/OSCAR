# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
from typing import List

import coloredlogs
from numpy import random

from ..context import Context
from .base import Actor
from .controller_actors import WalkerController
from .scene_actors import Vehicle, Walker

__all__ = ["ActorsGenerator"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class ActorsGenerator:
    def __init__(
        self, context: Context = Context(), number_of_vehicles: int = 0, number_of_walkers: int = 0
    ):
        self.context = context
        self.number_of_vehicles = number_of_vehicles
        self.number_of_walkers = number_of_walkers

    def generate_vehicles(self) -> List[Actor]:
        number_of_spawn_points = self.context.get_number_of_spawn_points()
        if number_of_spawn_points < self.number_of_vehicles:
            logger.warning(
                f"Requested {self.number_of_vehicles} vehicles, but could only find {number_of_spawn_points} spawn points"
            )
            self.number_of_vehicles = number_of_spawn_points

        vehicles = []
        for _ in range(self.number_of_vehicles):
            R = random.randint(0, 255)
            G = random.randint(0, 255)
            B = random.randint(0, 255)
            color = f"{R},{G},{B}"
            new_vehicle = Vehicle(color=color, role_name="autopilot")
            vehicles.append(new_vehicle)

        return vehicles

    def generate_walkers(self) -> List[Actor]:
        walkers = []
        for _ in range(self.number_of_walkers):
            speed = 1 + random.random()  # Between 1 and 2 m/s
            destination = self.context.get_spawn_point(from_navigation=True)

            new_walker_controller = WalkerController(
                max_speed=speed, destination_transform=destination
            )
            attachments = [new_walker_controller]
            new_walker = Walker(is_invincible=False, attachments=attachments)
            walkers.append(new_walker)

        return walkers
