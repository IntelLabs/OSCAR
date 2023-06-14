# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from dataclasses import dataclass

from carla import BlueprintLibrary
from numpy import random

__all__ = ["Blueprint"]


@dataclass
class Blueprint:
    # Blueprint library
    bp_type: str = ""
    carla_blueprint: BlueprintLibrary = None

    # Blueprint attributes
    role_name: str = ""

    def setup(self, blueprint_lib: BlueprintLibrary = None) -> bool:
        # get actor's blueprint
        blueprints = blueprint_lib.filter(self.bp_type)
        self.carla_blueprint = random.choice(blueprints)

        # set blueprint's attributes
        for attribute, value in self.__dict__.items():
            if self.carla_blueprint.has_attribute(attribute):
                self.carla_blueprint.set_attribute(attribute, str(value))

        return True
