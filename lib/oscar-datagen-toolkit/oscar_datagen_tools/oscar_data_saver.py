#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import os
import sys
from pathlib import Path

import carla
import coloredlogs
import hydra
from numpy import random
from omegaconf import DictConfig, OmegaConf

from oscar_datagen_tools.common_utils import connect_to_carla_server

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

# check if PROJECT_ROOT env variable
default_project_root = Path(os.path.expanduser("~")) / "oscar_data"
if "PROJECT_ROOT" not in os.environ:
    # if PROJECT_ROOT does not exist, set the HOME directory
    os.environ["PROJECT_ROOT"] = str(default_project_root)


def collect_from_sim(cfg):
    logger.info("Start CARLA simulation...")
    controller = connect_to_carla_server(cfg)
    if not controller:
        sys.exit(1)

    logger.info(f"Start collection data to {cfg.paths.data_dir}")
    if not controller.collect():
        sys.exit(1)

    if not controller.destroy():
        sys.exit(1)


@hydra.main(version_base="1.2", config_path="configs", config_name="collector")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    collect_from_sim(cfg)


if __name__ == "__main__":
    main()
