#
# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from dataclasses import dataclass

from carla import LaneType

__all__ = ["MotionParameters"]


@dataclass
class MotionParameters:
    sampling_resolution: int = 2
    lane_type: LaneType = LaneType.Driving
