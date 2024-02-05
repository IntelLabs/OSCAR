# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
from dataclasses import dataclass
from pathlib import Path

import coloredlogs
import cv2
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

from ..common_utils import (
    filename_decode,
    get_patch_in_camera,
    load_dynamic_metadata,
    load_sensors_static_metadata,
)

__all__ = ["CarlaTextureRenderer"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


@dataclass
class CarlaImageInfo:
    frame_num: int = None
    sensor_id: int = None
    patch_id: int = None
    texture: ndimage = None
    width: int = None
    height: int = None


def get_frame_info(item_path: Path, root_metadata: Path):
    # 1. Load static metadata
    sensor_id, frame_num = filename_decode(item_path.name)
    run_id = item_path.parent.name
    static_metadata_path = root_metadata / run_id / "metadata_static.json"
    assert static_metadata_path.exists()
    sensor_static_metadata = load_sensors_static_metadata(static_metadata_path, sensor_id)
    logger.info(f"Sensor ID: {sensor_id}")

    # 2. Load dynamic metadata
    dynamic_metadata_path = root_metadata / run_id / f"{frame_num}.json"
    assert dynamic_metadata_path.exists()
    dynamic_metadata = load_dynamic_metadata(dynamic_metadata_path, sensor_id)

    # 3. Get patch
    patch_id, _ = get_patch_in_camera(sensor_static_metadata, dynamic_metadata)
    logger.info(f"Patch ID: {patch_id}")

    width = sensor_static_metadata["image_size_x"]
    height = sensor_static_metadata["image_size_y"]

    sample_info = CarlaImageInfo(
        frame_num=int(frame_num),
        sensor_id=int(sensor_id),
        patch_id=int(patch_id),
        width=width,
        height=height,
    )

    return sample_info


class CarlaTextureRenderer:
    def __init__(
        self,
        image_path,
        root_metadata,
        controller,
        num_insertion_ticks: int = 2,
    ):
        self.info = get_frame_info(image_path, root_metadata)
        self.controller = controller
        self.num_insertion_ticks = num_insertion_ticks

    def to_tensor(self, ndarray, device=None):
        # HWC -> CHW
        # return torch.tensor(ndarray[:, :, ::-1].copy(), device=device).permute(2, 0, 1)
        return torch.tensor(ndarray.copy(), device=device).permute(2, 0, 1)

    def to_opencv(self, tensor):
        # CHW -> HWC
        return tensor.cpu().permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy().copy()

    # Get a tensor, return a tensor.
    def __call__(self, texture: torch.Tensor):
        device = texture.device
        texture = self.to_opencv(texture)

        self.info.texture = texture
        samples_info = [self.info]

        # Look for the patch
        patch_ids_to_filter = [sample.patch_id for sample in samples_info]
        patches = [actor for actor in self.controller.actors if actor.id in patch_ids_to_filter]
        patches = sorted(patches, key=lambda patch: patch_ids_to_filter.index(patch.id))

        # Set the perturbation on the patch
        for patch, info in zip(patches, samples_info):
            patch.texture = info.texture

        # tick the simulation
        for _ in tqdm(range(self.num_insertion_ticks), desc="Texture rendering ticks"):
            self.controller.context.world.tick()

        # Collect the data
        sensor_ids_to_filter = [sample.sensor_id for sample in samples_info]
        sensors = [
            sensor for sensor in self.controller.sensors if sensor.id in sensor_ids_to_filter
        ]
        sensors = sorted(sensors, key=lambda camera: sensor_ids_to_filter.index(camera.id))

        data = []
        for sensor, info in zip(sensors, samples_info):
            sensor_data = sensor.data
            data.append(bytes(sensor_data.raw_data))

        # image post-processing
        tmp_image = np.frombuffer(data[0], dtype=np.dtype("uint8"))
        tmp_image = np.reshape(tmp_image, (self.info.height, self.info.width, 4))
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)

        image = self.to_tensor(tmp_image, device=device)
        return image
