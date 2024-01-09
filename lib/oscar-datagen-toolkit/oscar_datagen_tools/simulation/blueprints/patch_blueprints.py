# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from dataclasses import dataclass

from .base import Blueprint

__all__ = ["MediumPatch", "BigPatch", "HugePatch", "Patch"]

# The Unreal Editor works in units of centimeters, while CARLA works
# in units of meters so the units must be converted. Ensure to divide
# the Unreal Editor coordinates by 100 before using in the CARLA simulator.
# https://carla.readthedocs.io/en/latest/tuto_G_pedestrian_bones/#spawning-a-pedestrian-in-the-carla-simulator
CONVERSION_SCALE = 100


@dataclass
class Patch(Blueprint):
    # Blueprint library
    bp_type: str = "static.prop.*"

    # The approx_width and approx_height are in centimeters
    approx_width: float = 1.0
    approx_height: float = 1.0


@dataclass
class MediumPatch(Patch):
    # Blueprint library
    bp_type: str = "static.prop.mediumpatch"
    approx_width: float = 175 / CONVERSION_SCALE
    approx_height: float = 175 / CONVERSION_SCALE


@dataclass
class BigPatch(Patch):
    # Blueprint library
    bp_type: str = "static.prop.bigpatch"
    approx_width: float = 475 / CONVERSION_SCALE
    approx_height: float = 300 / CONVERSION_SCALE


@dataclass
class HugePatch(Patch):
    # Blueprint library
    bp_type: str = "static.prop.hugepatch"
    approx_width: float = 850 / CONVERSION_SCALE
    approx_height: float = 725 / CONVERSION_SCALE
