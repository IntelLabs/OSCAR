# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import coloredlogs
import hydra
import yaml
from hydra import compose, initialize_config_module
from omegaconf import DictConfig
from torchvision.datasets import CocoDetection

from .. import AnnotatorController
from ..annotation.annotators import COCO
from ..annotation.utils import CategoriesHandler
from ..common_utils import connect_to_carla_server
from .carla_renderer import CarlaTextureRenderer

__all__ = ["CarlaDataset"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class CarlaDataset(CocoDetection):
    def __init__(
        self,
        simulation_run: str,
        annFile: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        modality: str = "rgb",
        num_insertion_ticks: int = 2,
        **kwargs,
    ):
        # Verify simulation path
        simulation_run = Path(simulation_run).resolve()
        if not simulation_run.exists():
            raise AttributeError(f"Simulation run {simulation_run} does not exist!")

        # verify the annotation file
        if not annFile:
            logger.info("No annotation file provided, generating one with the annotator tool...")
            categories_handler = CategoriesHandler()
            # assumed COCO compatible format
            annotator = COCO(
                dataset_path=simulation_run,
                sensor_type=modality,
                categories_handler=categories_handler,
            )
            annotator_controller = AnnotatorController(
                annotator=annotator,
                dataset_path=simulation_run,
                sensor=modality,
                output="kwcoco_annotations.json",
            )

            if not annotator_controller.prepare():
                raise Exception("Annotator controller preparation failed.")

            if not annotator_controller.annotate():
                raise Exception("Annotate process failed.")

            # set new annotation file path
            annFile = simulation_run / "kwcoco_annotations.json"

        # MART may already use Hydra.
        hydra.core.global_hydra.GlobalHydra.instance().clear()

        # get hydra config name and overrides from hydra.yaml file
        config_name = "collector"
        hydra_overrides = []
        hydra_info_path = simulation_run / ".hydra" / "hydra.yaml"
        if hydra_info_path.exists():
            with open(hydra_info_path, 'r') as file:
                hydra_data = yaml.safe_load(file)

                # get config name
                try:
                    config_name = hydra_data["hydra"]["job"]["config_name"]
                except KeyError:
                    logger.warning(
                        f"config_name not found in {hydra_info_path}, using default value: {config_name}"
                    )

                # get overrides
                try:
                    hydra_overrides.extend(hydra_data["hydra"]["overrides"]["task"])
                except KeyError:
                    logger.warning(
                        f"Overrides not found in {hydra_info_path}, using default value: {hydra_overrides}"
                    )

        # load hydra config info
        cfg = None
        with initialize_config_module(
            version_base="1.2", config_module="oscar_datagen_tools.configs"
        ):
            cfg = compose(config_name=config_name, overrides=hydra_overrides)

        if cfg is None:
            raise AttributeError("Simulation configuration information is None!")

        # setup CARLA simulation
        self.num_insertion_ticks = num_insertion_ticks
        self.root_metadata = simulation_run / "metadata"
        assert self.root_metadata.exists()

        logger.info("Start CARLA simulation...")
        cfg.paths.data_dir = None  # We do not want to store data
        self.controller = connect_to_carla_server(cfg)
        if not self.controller:
            sys.exit(1)

        if not self.controller.collect():
            sys.exit(1)

        logger.info("CARLA simulation is ready")

        # CocoDetection init
        root_modality = simulation_run / modality
        super().__init__(
            root=root_modality,
            annFile=annFile,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

    def _load_target(self, id: int) -> List[Any]:
        annotations = super()._load_target(id)
        file_name = self.coco.loadImgs(id)[0]["file_name"]

        return {"image_id": id, "file_name": file_name, "annotations": annotations}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = super().__getitem__(index)

        image_path = Path(target["file_name"])
        renderer = CarlaTextureRenderer(
            image_path, self.root_metadata, self.controller, self.num_insertion_ticks
        )
        target["renderer"] = renderer

        return image, target

    def __del__(self):
        if self.controller:
            self.controller.destroy()
