#
# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import logging
from typing import List

import carla
import coloredlogs
from carla import AttachmentType, Transform

from ..blueprints import Vehicle as BPVehicle
from ..blueprints import Walker as BPWalker
from ..context import Context
from .base import Actor

__all__ = ["Walker", "Vehicle"]

KEY_VALUE = "actors"

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class SceneActor(Actor):
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

        bbox = self.carla_actor.bounding_box
        actor_data["3D_BoundingBox"] = {}
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

        return (actor_data, KEY_VALUE)

    def get_dynamic_metadata(self):
        actor_data, _ = super().get_dynamic_metadata()

        return (actor_data, KEY_VALUE)


class Walker(SceneActor):
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

        self.blueprint = BPWalker(*args, **kwargs)

    def __pre_spawn__(self):
        assert self.context is not None
        assert self.context.world is not None

        # verify transforms
        if self.transform is None:
            self.transform = self.context.get_spawn_point(from_navigation=True)
            logger.debug(
                f"Selected random location (none provided) x: {self.transform.location.x}, "
                f"y: {self.transform.location.y}, z: {self.transform.location.z} "
                f"roll: {self.transform.rotation.roll}, pitch: {self.transform.rotation.pitch}, yaw: {self.transform.rotation.yaw}"
            )
        if self.destination_transform is None:
            self.destination_transform = self.transform
            logger.debug(
                f"Selected random destination location (none provided) x: {self.destination_transform.location.x}, "
                f"y: {self.destination_transform.location.y}, z: {self.destination_transform.location.z}"
            )

        assert isinstance(self.blueprint, BPWalker)
        return True

    def get_dynamic_metadata(self):
        actor_data, key_value = super().get_dynamic_metadata()

        bones = self.carla_actor.get_bones().bone_transforms
        actor_data["bones"] = []
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

        return (actor_data, key_value)


class Vehicle(SceneActor):
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

        self.blueprint = BPVehicle(*args, **kwargs)

    def __pre_spawn__(self):
        if not super().__pre_spawn__():
            return False

        assert isinstance(self.blueprint, BPVehicle)
        return True

    def __post_spawn__(self) -> bool:
        if not super().__post_spawn__():
            return False

        # setup vehicle
        self.carla_actor.set_autopilot(True, self.context.simulation_params.traffic_manager_port)
        self.carla_actor.set_light_state(carla.VehicleLightState.NONE)

        return True
