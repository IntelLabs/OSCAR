# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from dataclasses import dataclass

from .base import Blueprint

__all__ = ["Walker", "WalkerAIController", "Vehicle"]


@dataclass
class Walker(Blueprint):
    # Blueprint library
    bp_type: str = "walker.pedestrian.*"

    # Blueprint attributes
    is_invincible: bool = False
    speed: float = 0.0


@dataclass
class WalkerAIController(Blueprint):
    # Blueprint library
    bp_type: str = "controller.ai.walker"

    # Blueprint attributes
    max_speed: float = 0.0


@dataclass
class Vehicle(Blueprint):
    # Blueprint library
    bp_type: str = "vehicle.*.*"
    color: str = "255,0,0"
    sticky_control: bool = False
    terramechanics: bool = False
