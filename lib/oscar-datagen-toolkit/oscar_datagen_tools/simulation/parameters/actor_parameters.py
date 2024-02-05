#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from dataclasses import dataclass
from typing import List, Tuple

from carla import LaneType, Rotation, Transform
from numpy import random

from ..context import Context

__all__ = ["MotionParameters"]


@dataclass
class MotionParameters:
    sampling_resolution: int = 2
    lane_type: LaneType = LaneType.Driving
    num_waypoints: int = 0
    elevation: float = 0
    context: Context = Context()

    def __post_init__(self) -> None:
        self._points = []

    @property
    def points(self) -> List[Transform]:
        if len(self._points) == 0:
            for _ in range(self.num_waypoints):
                random_point = self.context.get_spawn_point()
                self._points.append(random_point)

        return self._points
