# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from dataclasses import dataclass

from .base import Blueprint

__all__ = ["Sensor", "RGB", "InstanceSegmentation", "Depth"]


# Default values taken from: https://github.com/carla-simulator/carla/blob/f14acb257ebf44c302b225b02080ac5f0eedcf7f/Docs/ref_sensors.md
@dataclass
class Sensor(Blueprint):
    # Blueprint library
    bp_type: str = "sensor.*"

    # Blueprint attributes
    speed: float = 0.5
    fov: float = 90.0
    image_size_x: int = 800
    image_size_y: int = 600
    lens_circle_falloff: float = 5.0
    lens_circle_multiplier: float = 0.0
    lens_k: float = -1.0
    lens_kcube: float = 0.0
    lens_x_size: float = 0.08
    lens_y_size: float = 0.08
    sensor_tick: float = 0.0


@dataclass
class RGB(Sensor):
    # Blueprint library
    bp_type: str = "sensor.camera.rgb"

    # Blueprint attributes
    black_clip: float = 0.0
    white_clip: float = 0.04
    blade_count: int = 5
    bloom_intensity: float = 0.675
    blur_amount: float = 1.0
    blur_radius: float = 0.0
    calibration_constant: float = 16.0
    chromatic_aberration_intensity: float = 0.0
    chromatic_aberration_offset: float = 0.0
    enable_postprocess_effects: bool = True
    exposure_compensation: float = 0.75
    exposure_max_bright: float = 12.0
    exposure_min_bright: float = 10.0
    exposure_mode: str = "histogram"
    exposure_speed_down: float = 1.0
    exposure_speed_up: float = 3.0
    focal_distance: float = 1000.0
    fstop: float = 1.4
    min_fstop: float = 1.2
    gamma: float = 2.2
    iso: float = 100.0
    motion_blur_intensity: float = 0.45
    motion_blur_max_distortion: float = 0.35
    motion_blur_min_object_screen_size: float = 0.1
    shoulder: float = 0.26
    shutter_speed: float = 200.0
    slope: float = 0.88
    temp: float = 6500.0
    tint: float = 0.0
    toe: float = 0.55


@dataclass
class InstanceSegmentation(Sensor):
    # Blueprint library
    bp_type: str = "sensor.camera.instance_segmentation"


@dataclass
class Depth(Sensor):
    # Blueprint library
    bp_type: str = "sensor.camera.depth"
