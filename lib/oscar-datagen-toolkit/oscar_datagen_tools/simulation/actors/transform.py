# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import logging
from typing import Tuple, Union

import carla
import coloredlogs
from numpy import random

__all__ = ["Location", "Rotation"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class Location(carla.Location):
    def __init__(
        self,
        x: Union[float, Tuple[float]] = 0.0,
        y: Union[float, Tuple[float]] = 0.0,
        z: Union[float, Tuple[float]] = 0.0,
    ) -> None:
        # process location coordinates
        self._x = x
        self._y = y
        self._z = z

        if isinstance(x, (list, tuple)):
            self._x = random.uniform(x[0], x[-1])

        if isinstance(y, (list, tuple)):
            self._y = random.uniform(y[0], y[-1])

        if isinstance(z, (list, tuple)):
            self._z = random.uniform(z[0], z[-1])

        # init base class
        super().__init__(x=self._x, y=self._y, z=self._z)


class Rotation(carla.Rotation):
    def __init__(
        self,
        pitch: Union[float, Tuple[float]] = 0.0,
        yaw: Union[float, Tuple[float]] = 0.0,
        roll: Union[float, Tuple[float]] = 0.0,
    ) -> None:
        # process rotation values
        self._pitch = pitch
        self._yaw = yaw
        self._roll = roll

        if isinstance(pitch, (list, tuple)):
            self._pitch = random.uniform(pitch[0], pitch[-1])

        if isinstance(yaw, (list, tuple)):
            self._yaw = random.uniform(yaw[0], yaw[-1])

        if isinstance(roll, (list, tuple)):
            self._roll = random.uniform(roll[0], roll[-1])

        # init base class
        super().__init__(pitch=self._pitch, yaw=self._yaw, roll=self._roll)
