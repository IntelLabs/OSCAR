#
# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import logging
from typing import Any, Dict, List

import carla
import coloredlogs
from carla import AttachmentType, Transform

from ..context import Context
from ..utils import run_spawn

__all__ = ["Actor"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class Actor:
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
        self.carla_actor = None
        self.blueprint = None
        self.context = context
        self.transform = transform
        self.destination_transform = destination_transform
        self.id = 0
        self.name = ""
        self.parent = None
        self.attachments = attachments
        self.attachment_type = attachment_type

        for attachment in self.attachments:
            attachment.parent = self

    def get_transform(self):
        if self.carla_actor is not None:
            return self.carla_actor.get_transform()

        return self.transform

    def set_transform(self, value):
        if self.carla_actor is not None:
            self.carla_actor.set_transform(value)

    def __pre_spawn__(self) -> bool:
        assert self.context is not None
        assert self.context.world is not None

        # verify transforms
        if self.transform is None:
            self.transform = self.context.get_spawn_point()
            if self.transform is None:
                logger.error(
                    f"None of the {self.context.get_number_of_spawn_points()} spawn points available"
                )
                return False

            logger.debug(
                f"Selected random spawn point (none provided) x: {self.transform.location.x}, "
                f"y: {self.transform.location.y}, z: {self.transform.location.z} "
                f"roll: {self.transform.rotation.roll}, pitch: {self.transform.rotation.pitch}, yaw: {self.transform.rotation.yaw}"
            )
        if self.destination_transform is None:
            self.destination_transform = self.transform
            logger.debug(
                f"Selected random destination point (none provided) x: {self.destination_transform.location.x}, "
                f"y: {self.destination_transform.location.y}, z: {self.destination_transform.location.z}"
            )

        return True

    def __post_spawn__(self) -> bool:
        self.id = self.carla_actor.id
        self.name = f"{self.carla_actor.type_id}.{self.carla_actor.id}"
        return True

    @run_spawn
    def spawn(self) -> bool:
        if not self.__pre_spawn__():
            logger.error("Pre-spawn process failed.")
            return False

        # setup actor's blueprint
        assert self.blueprint is not None
        blueprint_lib = self.context.world.get_blueprint_library()
        if not self.blueprint.setup(blueprint_lib):
            logger.error("Blueprint setup failed.")
            return False

        # spawn actor
        attach_to = self.parent.carla_actor if self.parent is not None else None
        self.carla_actor = self.context.world.spawn_actor(
            self.blueprint.carla_blueprint,
            self.transform,
            attach_to=attach_to,
            attachment_type=self.attachment_type,
        )
        self.context.world.tick()

        # spawn attachments
        for attachment in self.attachments:
            if not attachment.spawn():
                return False

        if not self.__post_spawn__():
            logger.error("Post-spawn process failed.")
            return False

        return True

    def is_alive(self) -> bool:
        assert self.carla_actor is not None
        return self.carla_actor.is_alive

    def destroy(self) -> bool:
        return self.carla_actor.destroy()

    def get_static_metadata(self) -> Dict[Dict[str, Any], str]:
        return ({}, "")

    def get_dynamic_metadata(self) -> Dict[Dict[str, Any], str]:
        actor_data = {}
        actor_data["type"] = self.carla_actor.type_id

        actorsnapshot = self.context.world.get_snapshot().find(self.carla_actor.id)
        actorsnapshot_transform = actorsnapshot.get_transform()
        actor_data["actorsnapshot_location"] = (
            actorsnapshot_transform.location.x,
            actorsnapshot_transform.location.y,
            actorsnapshot_transform.location.z,
        )
        actor_data["actorsnapshot_rotation"] = (
            actorsnapshot_transform.rotation.pitch,
            actorsnapshot_transform.rotation.yaw,
            actorsnapshot_transform.rotation.roll,
        )
        actor_data["actorsnapshot_transform_get_matrix"] = actorsnapshot_transform.get_matrix()
        actor_data[
            "actorsnapshot_transform_get_inverse_matrix"
        ] = actorsnapshot_transform.get_inverse_matrix()

        actor_transform = self.carla_actor.get_transform()
        actor_data["actor_location"] = (
            actor_transform.location.x,
            actor_transform.location.y,
            actor_transform.location.z,
        )
        actor_data["actor_rotation"] = (
            actor_transform.rotation.pitch,
            actor_transform.rotation.yaw,
            actor_transform.rotation.roll,
        )
        actor_data["actor_transform_get_matrix"] = actor_transform.get_matrix()
        actor_data["actor_transform_get_inverse_matrix"] = actor_transform.get_inverse_matrix()

        return (actor_data, "")
