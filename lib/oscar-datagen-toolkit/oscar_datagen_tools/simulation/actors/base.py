#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
from typing import Any, Dict, List

import carla
import coloredlogs
from numpy import random
from carla import Transform, ActorAttributeType

from ..context import Context
from ..utils import is_jsonable

__all__ = ["Actor"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class Actor:
    KEY_VALUE = "actors"

    def __init__(
        self,
        blueprint: carla.ActorBlueprint | str = None,
        *,
        context: Context = Context(),
        transform: Transform = None,
        attachments: list[carla.Actor] = None,
        **blueprint_attributes,
    ) -> None:
        self._carla_actor = None
        self.blueprint = blueprint
        self.context = context
        self.transform = transform
        self._id = None
        self._name = None
        self._parent = None
        self.attachments = attachments or []

        for attachment in self.attachments:
            attachment.parent = self

        if isinstance(blueprint, str):
            blueprints = context.blueprint_library.filter(blueprint)
            if len(blueprints) == 0:
                raise AttributeError(f"Could not find \"{blueprint}\" blueprint in CARLA.")
            self.blueprint = random.choice(blueprints)

        # set blueprint's attributes
        if self.blueprint is not None:
            for attribute, value in blueprint_attributes.items():
                if self.blueprint.has_attribute(attribute):
                    self.blueprint.set_attribute(attribute, str(value))

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        if self._parent is not None:
            raise AttributeError("Actors can have only one parent!")
        self._parent = value

    def __getattr__(self, name):
        # Check if blueprint has attribute
        if self.blueprint is None or not self.blueprint.has_attribute(name):
            return object.__getattribute__(self, name)

        value = self.blueprint.get_attribute(name)

        if value.type == ActorAttributeType.Bool:
            return value.as_bool()

        if value.type == ActorAttributeType.Int:
            return value.as_int()

        if value.type == ActorAttributeType.Float:
            return value.as_float()

        if value.type == ActorAttributeType.String:
            return value.as_str()

        if value.type == ActorAttributeType.RGBColor:
            return value.as_color()

        return value

    def get_transform(self):
        if self.carla_actor is not None:
            return self.carla_actor.get_transform()

        return self.transform

    def set_transform(self, value):
        if self.carla_actor is not None:
            self.carla_actor.set_transform(value)

    @property
    def id(self):
        if self._id:
            return self._id
        return self.carla_actor.id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def name(self):
        if self._name:
            return self._name
        return f"{self.carla_actor.type_id}.{self.carla_actor.id}"

    @name.setter
    def name(self, value):
        self._name = value

    def spawn(self) -> bool:
        ret = True
        if not getattr(self, "__pre_spawn__", lambda: True)():
            logger.error(f"Pre-spawn process failed: {self}")
            ret = False

        # if __pre_spawn__ failed, force an spawn error
        command = self._spawn_command() if ret else carla.command.SpawnActor()
        response = yield command

        # only log error if we haven't already errored and we have a blueprint
        # some actors don't have blueprints, which causes an error when spawning.
        if ret and response.has_error() and self.blueprint is not None:
            logging.warning(f"{self}: {response.error}")
            ret = False

        # get_actor returns None when actor_id cannot be found
        self._carla_actor = self.context.world.get_actor(response.actor_id)

        # only call __post_spawn__ if __pre_spawn__ and response didn't fail
        # this means code in __post_spawn__ can rely upon there being a carla_actor
        if ret and not getattr(self, "__post_spawn__", lambda: True)():
            logger.error(f"Post-spawn process failed: {self}")
            ret = False

        return ret

    def _spawn_command(self) -> carla.Command:
        if self.blueprint is None:
            # This will cause an "invalid actor description" error, but we just
            # ignore it in spawn.
            return carla.command.SpawnActor()
        
        if self.transform is None:
            raise AttributeError("CARLA actor must have a transform assigned!")

        if self.parent and self.parent.is_alive:
            # NOTE: You would think passing parent=None to this command would
            #       work, but it causes Python to crash.
            return carla.command.SpawnActor(
                self.blueprint, self.transform, self.parent.carla_actor
            )
        else:
            return carla.command.SpawnActor(
                self.blueprint, self.transform
            )

    @property
    def carla_actor(self) -> carla.Actor:
        assert self._carla_actor is not None and self._carla_actor.is_alive, "Calls to carla_actor should be protected by a check on is_alive!"
        return self._carla_actor

    @property
    def is_alive(self) -> bool:
        if self._carla_actor is None:
            return False
        return self._carla_actor.is_alive

    def destroy(self) -> carla.Command:
        actor_id = 0
        if self._carla_actor is not None:
            actor_id = self._carla_actor.id
            self._carla_actor = None

        return carla.command.DestroyActor(actor_id)

    def get_static_metadata(self, **extra_metadata) -> dict[dict[str, Any], str]:
        actor_data = dict(extra_metadata)

        if self.blueprint:
            actor_data["blueprint_id"] = self.blueprint.id
            for attribute in iter(self.blueprint):
                name = attribute.id
                value = getattr(self, name)
                if is_jsonable(value):
                    actor_data[name] = value

        return {self.KEY_VALUE: {self.id: actor_data}}

    def get_dynamic_metadata(self, **extra_metadata) -> dict[dict[str, Any], str]:
        actor_data = dict(extra_metadata)

        if self.blueprint:
            actor_data["blueprint_id"] = self.blueprint.id

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

        return {self.KEY_VALUE: {self.id: actor_data}}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.blueprint} @ {self.transform})"
