#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List

import carla
import coloredlogs
from carla import AttachmentType, Location, Transform
from scipy import ndimage

from oscar_datagen_tools.common_utils import format_patch_name

from ..context import Context
from ..utils import generate_uuid, load_image, transform
from .base import Actor

__all__ = ["Patch"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

OBJECT_NAME_KEY = "StaticMeshActor"


@dataclass
class TextureROI:
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0


class Patch(Actor):
    KEY_VALUE = "patches"

    def __init__(
        self,
        width: float,
        height: float,
        texture: ndimage | str = None,
        texture_roi: TextureROI = None,
        **kwargs,
    ) -> None:
        # Construct blueprint id using specified format
        _width = str(width).replace(".", "_")
        _height = str(height).replace(".", "_")
        super().__init__(f"static.prop.patch-{_width}x{_height}", **kwargs)

        self.id = generate_uuid()
        self.object_name = ""
        self.element_names = []
        self.texture_roi = texture_roi
        self.texture = texture

        self.corners = self.__extract_corners__(width, height)

    @property
    def texture(self):
        return self._texture

    @texture.setter
    def texture(self, value):
        self._texture = value
        self.__apply_texture__()

    def __apply_texture__(self):
        if not self.object_name:
            return

        if self._texture is None:
            return

        if isinstance(self._texture, str):
            self._texture = load_image(self._texture)
        height, width, _ = self._texture.shape

        # Use a ROI.
        if self.texture_roi:
            x1 = int(self.texture_roi.x * width)
            x2 = int((self.texture_roi.width * width) + x1)
            y1 = int(self.texture_roi.y * height)
            y2 = int((self.texture_roi.height * height) + y1)
            self._texture = self._texture[y1:y2, x1:x2, :]
            height, width, _ = self._texture.shape

        # Instantiate a carla.TextureColor object and populate
        # the pixels with data from the modified image
        texture = carla.TextureColor(width, height)
        for column in range(0, width):
            for row in range(0, height):
                r = int(self._texture[row, column, 0])
                g = int(self._texture[row, column, 1])
                b = int(self._texture[row, column, 2])
                a = 255
                texture.set(column, row, carla.Color(r, g, b, a))

        # Apply the texture to the building asset
        self.context.world.apply_color_texture_to_object(
            self.object_name, carla.MaterialParameter.Diffuse, texture
        )

    def __pre_spawn__(self) -> bool:
        # get object names before spawn patch
        self.element_names = set(
            filter(lambda k: OBJECT_NAME_KEY in k, self.context.world.get_names_of_all_objects())
        )

        return True

    def __post_spawn__(self) -> bool:
        self.name = format_patch_name(self.id)

        # Find object name in the simulator
        new_element_names = set(
            filter(lambda k: OBJECT_NAME_KEY in k, self.context.world.get_names_of_all_objects())
        )

        _object_name = new_element_names - self.element_names
        if len(_object_name) > 0:
            self.object_name = _object_name.pop()
        else:
            logger.warning(f"No object name generated for patch: {self.name}")

        # Apply new texture
        self.__apply_texture__()

        return True

    def __extract_corners__(self, width: float, height: float):
        # set patch corner locations
        # (x, y)
        upper_left = (
            self.transform.location.x - (width / 2),
            self.transform.location.y - (height / 2),
        )
        upper_right = (
            self.transform.location.x + (width / 2),
            self.transform.location.y - (height / 2),
        )
        bottom_right = (
            self.transform.location.x + (width / 2),
            self.transform.location.y + (height / 2),
        )
        bottom_left = (
            self.transform.location.x - (width / 2),
            self.transform.location.y + (height / 2),
        )

        # [upper-left, upper-right, bottom-right, bottom-left]
        corners = []
        corners.append(
            Transform(
                location=Location(x=upper_left[0], y=upper_left[1], z=self.transform.location.z),
                rotation=self.transform.rotation,
            )
        )
        corners.append(
            Transform(
                location=Location(x=upper_right[0], y=upper_right[1], z=self.transform.location.z),
                rotation=self.transform.rotation,
            )
        )
        corners.append(
            Transform(
                location=Location(
                    x=bottom_right[0], y=bottom_right[1], z=self.transform.location.z
                ),
                rotation=self.transform.rotation,
            )
        )
        corners.append(
            Transform(
                location=Location(x=bottom_left[0], y=bottom_left[1], z=self.transform.location.z),
                rotation=self.transform.rotation,
            )
        )

        # transform corners
        for i, corner in enumerate(corners):
            corners[i] = transform(corner, self.transform)

        return corners

    def get_dynamic_metadata(self):
        actor_data = {}
        actor_data["corners"] = {}
        actor_data["corners"]["upper_left"] = {}
        actor_data["corners"]["upper_left"]["actor_location"] = (
            self.corners[0].location.x,
            self.corners[0].location.y,
            self.corners[0].location.z,
        )
        actor_data["corners"]["upper_left"]["actor_rotation"] = (
            self.corners[0].rotation.pitch,
            self.corners[0].rotation.yaw,
            self.corners[0].rotation.roll,
        )

        actor_data["corners"]["upper_right"] = {}
        actor_data["corners"]["upper_right"]["actor_location"] = (
            self.corners[1].location.x,
            self.corners[1].location.y,
            self.corners[1].location.z,
        )
        actor_data["corners"]["upper_right"]["actor_rotation"] = (
            self.corners[1].rotation.pitch,
            self.corners[1].rotation.yaw,
            self.corners[1].rotation.roll,
        )

        actor_data["corners"]["bottom_right"] = {}
        actor_data["corners"]["bottom_right"]["actor_location"] = (
            self.corners[2].location.x,
            self.corners[2].location.y,
            self.corners[2].location.z,
        )
        actor_data["corners"]["bottom_right"]["actor_rotation"] = (
            self.corners[2].rotation.pitch,
            self.corners[2].rotation.yaw,
            self.corners[2].rotation.roll,
        )

        actor_data["corners"]["bottom_left"] = {}
        actor_data["corners"]["bottom_left"]["actor_location"] = (
            self.corners[3].location.x,
            self.corners[3].location.y,
            self.corners[3].location.z,
        )
        actor_data["corners"]["bottom_left"]["actor_rotation"] = (
            self.corners[3].rotation.pitch,
            self.corners[3].rotation.yaw,
            self.corners[3].rotation.roll,
        )

        return super().get_dynamic_metadata(**actor_data)
