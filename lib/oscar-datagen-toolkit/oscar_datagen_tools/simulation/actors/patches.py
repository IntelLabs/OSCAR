#
# Copyright (C) 2023 Intel Corporation
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
import cv2
from carla import AttachmentType, Location, Transform

from oscar_datagen_tools.common_utils import format_patch_name

from ..blueprints import BigPatch as BPBigPatch
from ..blueprints import HugePatch as BPHugePatch
from ..blueprints import MediumPatch as BPMediumPatch
from ..blueprints import Patch as BPPatch
from ..context import Context
from ..utils import generate_uuid, get_number_of_tiles, transform
from .base import Actor

__all__ = ["BasePatch", "Patch", "ScalePatch"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

KEY_VALUE = "patches"
OBJECT_NAME_KEY = "StaticMeshActor"


@dataclass
class TextureROI:
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0


class BasePatch(Actor):
    def __init__(
        self,
        blueprint: BPPatch = None,
        context: Context = Context(),
        transform: Transform = None,
        destination_transform: Transform = None,
        attachments: list[carla.Actor] = None,
        attachment_type: AttachmentType = AttachmentType.Rigid,
        texture_path: str = "",
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

        self.blueprint = blueprint or BPPatch()
        self.texture_path = texture_path

    def get_static_metadata(self):
        actor_data = {}
        actor_data["type"] = self.carla_actor.type_id

        return (actor_data, KEY_VALUE)

    def get_dynamic_metadata(self):
        actor_data, _ = super().get_dynamic_metadata()

        return (actor_data, KEY_VALUE)


class Patch(BasePatch):
    def __init__(
        self,
        blueprint: BPPatch = None,
        context: Context = Context(),
        transform: Transform = None,
        destination_transform: Transform = None,
        attachments: list[carla.Actor] = None,
        attachment_type: AttachmentType = AttachmentType.Rigid,
        texture_path: str = "",
        texture_roi: TextureROI = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            blueprint,
            context,
            transform,
            destination_transform,
            attachments,
            attachment_type,
            texture_path,
            *args,
            **kwargs,
        )

        self.object_name = ""
        self.element_names = []
        self.texture_roi = texture_roi

    def __apply_texture__(self):
        if not self.texture_path or not self.object_name:
            return

        image = cv2.imread(self.texture_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # Use a ROI.
        if self.texture_roi:
            x1 = int(self.texture_roi.x * width)
            x2 = int((self.texture_roi.width * width) + x1)
            y1 = int(self.texture_roi.y * height)
            y2 = int((self.texture_roi.height * height) + y1)
            image = image[y1:y2, x1:x2, :]
            height, width, _ = image.shape

        # Instantiate a carla.TextureColor object and populate
        # the pixels with data from the modified image
        texture = carla.TextureColor(width, height)
        for column in range(0, width):
            for row in range(0, height):
                r = int(image[row, column, 0])
                g = int(image[row, column, 1])
                b = int(image[row, column, 2])
                a = 255
                texture.set(column, row, carla.Color(r, g, b, a))

        # Apply the texture to the building asset
        self.context.world.apply_color_texture_to_object(
            self.object_name, carla.MaterialParameter.Diffuse, texture
        )

    def __pre_spawn__(self) -> bool:
        if not super().__pre_spawn__():
            return False

        assert isinstance(self.blueprint, BPPatch)

        # get object names before spawn patch
        self.element_names = set(
            filter(lambda k: OBJECT_NAME_KEY in k, self.context.world.get_names_of_all_objects())
        )

        return True

    def __post_spawn__(self) -> bool:
        if not super().__post_spawn__():
            return False

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


class ScalePatch(BasePatch):
    def __init__(
        self,
        context: Context = Context(),
        transform: Transform = None,
        destination_transform: Transform = None,
        attachments: list[carla.Actor] = None,
        attachment_type: AttachmentType = AttachmentType.Rigid,
        texture_path: str = "",
        width: float = 1.0,
        height: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        # determine which patch fits better for the given width and height
        blueprint = None
        blueprints = [BPHugePatch(), BPBigPatch(), BPMediumPatch()]
        for possible_blueprint in blueprints:
            scale_x = width / possible_blueprint.approx_width
            scale_y = height / possible_blueprint.approx_height

            if scale_x >= 1.0 and scale_y >= 1.0:
                blueprint = possible_blueprint
                logger.debug(f"Selected patch blueprint: {blueprint}")
                break

        assert blueprint, f"No patch blueprint fits given width={width} and height={height}"

        self.scale_x = scale_x
        self.scale_y = scale_y
        logger.debug(f"Scale X: {self.scale_x}, scale Y: {self.scale_y}")

        super().__init__(
            blueprint,
            context,
            transform,
            destination_transform,
            attachments,
            attachment_type,
            texture_path,
            *args,
            **kwargs,
        )

        self.id = generate_uuid()
        self.width = width
        self.height = height
        self.subpatches = []

    def __pre_spawn__(self) -> bool:
        if not super().__pre_spawn__():
            return False

        # determine how may patches along x and its step size
        self._columns = get_number_of_tiles(self.scale_x)
        size_in_x = self.scale_x * self.blueprint.approx_width
        self._step_x = size_in_x / self._columns

        # determine how many patches along y and its step size
        self._rows = get_number_of_tiles(self.scale_y)
        size_in_y = self.scale_y * self.blueprint.approx_height
        self._step_y = size_in_y / self._rows

        # get the transform information of the upper-left patch
        subpatch_x = (
            self.transform.location.x + (self.blueprint.approx_width / 2) - (size_in_x / 2)
        )
        subpatch_y = (
            self.transform.location.y + (self.blueprint.approx_height / 2) - (size_in_y / 2)
        )
        subpatch_z = self.transform.location.z

        subpatch_location = Location(x=subpatch_x, y=subpatch_y, z=subpatch_z)
        self._subpatch_transform = Transform(
            location=subpatch_location, rotation=self.transform.rotation
        )

        # set patch corner locations
        # (x, y)
        upper_left = (
            self.transform.location.x - (self.width / 2),
            self.transform.location.y - (self.height / 2),
        )
        upper_right = (
            self.transform.location.x + (self.width / 2),
            self.transform.location.y - (self.height / 2),
        )
        bottom_right = (
            self.transform.location.x + (self.width / 2),
            self.transform.location.y + (self.height / 2),
        )
        bottom_left = (
            self.transform.location.x - (self.width / 2),
            self.transform.location.y + (self.height / 2),
        )

        # [upper-left, upper-right, bottom-right, bottom-left]
        self.corners = []
        self.corners.append(
            Transform(
                location=Location(x=upper_left[0], y=upper_left[1], z=self.transform.location.z),
                rotation=self.transform.rotation,
            )
        )
        self.corners.append(
            Transform(
                location=Location(x=upper_right[0], y=upper_right[1], z=self.transform.location.z),
                rotation=self.transform.rotation,
            )
        )
        self.corners.append(
            Transform(
                location=Location(
                    x=bottom_right[0], y=bottom_right[1], z=self.transform.location.z
                ),
                rotation=self.transform.rotation,
            )
        )
        self.corners.append(
            Transform(
                location=Location(x=bottom_left[0], y=bottom_left[1], z=self.transform.location.z),
                rotation=self.transform.rotation,
            )
        )

        # transform corners
        for i, corner in enumerate(self.corners):
            self.corners[i] = transform(corner, self.transform)

        return True

    def __post_spawn__(self) -> bool:
        self.name = format_patch_name(self.id)
        return True

    def __generate_tile__(self) -> bool:
        reference_location_x = self._subpatch_transform.location.x
        roi_width = 1.0 / self.scale_x
        roi_height = 1.0 / self.scale_y

        for row in range(self._rows):
            for column in range(self._columns):
                # calculate texture ROI location
                roi = TextureROI(
                    x=(1 / self._columns) * column,
                    y=(1 / self._rows) * row,
                    width=roi_width,
                    height=roi_height,
                )

                # apply transformation to subpatch location
                next_transform = transform(self._subpatch_transform, self.transform)

                # create sub-patch
                subpatch = Patch(
                    context=self.context,
                    blueprint=self.blueprint,
                    transform=next_transform,
                    destination_transform=self.destination_transform,
                    attachment_type=self.attachment_type,
                    texture_path=self.texture_path,
                    texture_roi=roi,
                )

                self.subpatches.append(subpatch)

                # update transform in x
                self._subpatch_transform.location.x += self._step_x

            # update transform in y
            self._subpatch_transform.location.x = reference_location_x
            self._subpatch_transform.location.y += self._step_y

        return True

    def spawn(self) -> bool:
        if not self.__pre_spawn__():
            logger.error("Pre-spawn process failed.")
            return False

        # Spawn all sub-patches
        if not self.__generate_tile__():
            logger.error("Patch tile generation failed.")
            return False

        for subpatch in self.subpatches:
            if not subpatch.spawn():
                return False

            self.carla_actor = subpatch.carla_actor

        if not self.__post_spawn__():
            logger.error("Post-spawn process failed.")
            return False

        return True

    def spawn_command(self, parent: carla.Actor = None) -> carla.Command:
        if not self.__pre_spawn__():
            logger.error("Pre-spawn process failed.")
            return False

        # generate all subpatches
        if not self.__generate_tile__():
            logger.error("Patch tile generation failed.")
            return False

        command = None
        for subpatch in self.subpatches:
            subpatch_command = subpatch.spawn_command()
            if not command:
                command = subpatch_command
            else:
                command.then(subpatch_command)

        return command

    def get_static_metadata(self):
        scale_patch_data = {}
        scale_patch_data["subpatches"] = {}
        scale_patch_data["size"] = (self.width, self.height)

        for subpatch in self.attachments:
            metadata, _ = subpatch.get_static_metadata()
            scale_patch_data["subpatches"][subpatch.carla_actor.id] = metadata

        return (scale_patch_data, KEY_VALUE)

    def get_dynamic_metadata(self):
        scale_patch_data = {}
        scale_patch_data["subpatches"] = {}

        scale_patch_data["actor_location"] = (
            self.transform.location.x,
            self.transform.location.y,
            self.transform.location.z,
        )
        scale_patch_data["actor_rotation"] = (
            self.transform.rotation.pitch,
            self.transform.rotation.yaw,
            self.transform.rotation.roll,
        )
        scale_patch_data["actor_transform_get_matrix"] = self.transform.get_matrix()
        scale_patch_data[
            "actor_transform_get_inverse_matrix"
        ] = self.transform.get_inverse_matrix()

        scale_patch_data["corners"] = {}
        scale_patch_data["corners"]["upper_left"] = {}
        scale_patch_data["corners"]["upper_left"]["actor_location"] = (
            self.corners[0].location.x,
            self.corners[0].location.y,
            self.corners[0].location.z,
        )
        scale_patch_data["corners"]["upper_left"]["actor_rotation"] = (
            self.corners[0].rotation.pitch,
            self.corners[0].rotation.yaw,
            self.corners[0].rotation.roll,
        )

        scale_patch_data["corners"]["upper_right"] = {}
        scale_patch_data["corners"]["upper_right"]["actor_location"] = (
            self.corners[1].location.x,
            self.corners[1].location.y,
            self.corners[1].location.z,
        )
        scale_patch_data["corners"]["upper_right"]["actor_rotation"] = (
            self.corners[1].rotation.pitch,
            self.corners[1].rotation.yaw,
            self.corners[1].rotation.roll,
        )

        scale_patch_data["corners"]["bottom_right"] = {}
        scale_patch_data["corners"]["bottom_right"]["actor_location"] = (
            self.corners[2].location.x,
            self.corners[2].location.y,
            self.corners[2].location.z,
        )
        scale_patch_data["corners"]["bottom_right"]["actor_rotation"] = (
            self.corners[2].rotation.pitch,
            self.corners[2].rotation.yaw,
            self.corners[2].rotation.roll,
        )

        scale_patch_data["corners"]["bottom_left"] = {}
        scale_patch_data["corners"]["bottom_left"]["actor_location"] = (
            self.corners[3].location.x,
            self.corners[3].location.y,
            self.corners[3].location.z,
        )
        scale_patch_data["corners"]["bottom_left"]["actor_rotation"] = (
            self.corners[3].rotation.pitch,
            self.corners[3].rotation.yaw,
            self.corners[3].rotation.roll,
        )

        for subpatch in self.attachments:
            metadata, _ = subpatch.get_dynamic_metadata()
            scale_patch_data["subpatches"][subpatch.carla_actor.id] = metadata

        return (scale_patch_data, KEY_VALUE)
