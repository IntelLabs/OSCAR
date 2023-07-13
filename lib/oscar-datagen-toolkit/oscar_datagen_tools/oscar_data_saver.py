#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import os
from pathlib import Path
from typing import List

import coloredlogs
import hydra
from numpy import random
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from oscar_datagen_tools.common_utils import generate_image_filename
from oscar_datagen_tools.simulation import Context, SyncMode
from oscar_datagen_tools.simulation.actors import Actor, ActorsGenerator, ISensor
from oscar_datagen_tools.simulation.utils import (
    actors_still_alive,
    replace_modality_in_path,
    run_loop,
    save_dynamic_metadata,
    save_static_metadata,
)

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

# check if PROJECT_ROOT env variable
default_project_root = Path(os.path.expanduser("~")) / "oscar_data"
if "PROJECT_ROOT" not in os.environ:
    # if PROJECT_ROOT does not exist, set the HOME directory
    os.environ["PROJECT_ROOT"] = default_project_root


class CollectorController:
    def __init__(
        self,
        actors: List[Actor],
        context: Context = Context(),
        actors_generator: ActorsGenerator = None,
        max_frames: int = 10,
        output_dir: Path = None,
    ):
        self.context = context
        self.actors_generator = actors_generator
        self.max_frames = max_frames
        self.actors = actors
        self.sensors = []
        self.output_dir = output_dir

    def prepare_simulation(self) -> bool:
        assert self.context is not None

        # setup simulator
        if not self.context.verify_connection():
            return False

        if not self.context.setup_simulator():
            return False

        # set random seed before generate actors
        random.seed(self.context.client_params.seed)

        if self.actors_generator is not None:
            # add auto generated actors
            vehicles = self.actors_generator.generate_vehicles()
            walkers = self.actors_generator.generate_walkers()
            self.actors = self.actors + vehicles + walkers

        # setup actors
        for actor in self.actors:
            # add the actor's attachments to the actor's list
            self.actors += actor.attachments

        # set random seed before spawn the actors
        random.seed(self.context.client_params.seed)

        # spawn actors
        for actor in tqdm(self.actors, desc="Actors's spawn progress"):
            # Just spawn the root actors. They will spawn
            # its possible attachments
            if actor.parent is not None:
                continue

            if not actor.spawn():
                return False

        self.sensors = [actor for actor in self.actors if isinstance(actor, ISensor)]

        # setup initial position from those sensors that use the SensorController
        for sensor in self.sensors:
            sensor.step()

        # save static metadata
        metadata_path = Path(replace_modality_in_path(self.output_dir, modality="metadata"))
        save_static_metadata(self.actors, path=metadata_path)

        return True

    @run_loop
    def collect(self) -> None:
        assert self.context is not None

        with SyncMode(self.context, self.sensors) as sync_mode:
            # warm up time before start collecting
            for _ in tqdm(range(self.context.simulation_params.warmup), desc="Warmup counter"):
                self.context.world.tick()

            local_frame_num = 0
            while True:
                local_frame_num += 1
                logger.info("Begin sync_mode.tick")

                if local_frame_num > self.max_frames:
                    logger.info(f"Max frames, {self.max_frames}, reached. Exiting...")
                    break

                # Advance the simulation and wait for the data.
                snapshot = sync_mode.tick(self.context.sync_params.timeout)
                logger.info(f"Snapshot: {snapshot}, Local frame: {local_frame_num}")

                # Break if any vehicle in our actor list has disappeared
                if not actors_still_alive(self.actors):
                    logger.info("Actor(s) dead")
                    break

                assert self.max_frames >= local_frame_num

                # save dynamic metadata
                metadata_path = Path(
                    replace_modality_in_path(self.output_dir, modality="metadata")
                )
                save_dynamic_metadata(self.actors, local_frame_num, snapshot, path=metadata_path)

                for sensor_idx, sensor in enumerate(self.sensors):
                    filename = generate_image_filename(
                        sensor.id,
                        sensor.transform.location.z,
                        sensor.transform.rotation.pitch
                        % 360,  # keep the angle as a positive value between 0 a 360
                        local_frame_num,
                    )
                    path = self.output_dir / filename

                    sensor.save_to_disk(path)
                    sensor.step()

    def destroy(self):
        self.context.traffic_manager.shut_down()

        for actor in self.actors:
            actor.destroy()


@hydra.main(version_base="1.2", config_path="configs", config_name="collector")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    # First instantiation of the Context Singleton object
    context = hydra.utils.instantiate(cfg.context, reinit=True)

    # set random seed before instantiate the spawn actors
    logger.info(f"Random seed: {context.client_params.seed}")
    random.seed(context.client_params.seed)

    # Convert all to primitive containers when instantiating the actors, specially
    # for the Blueprint's Sensor data type where the Camera's init method verify
    # this instance type.
    actors = hydra.utils.instantiate(cfg.spawn_actors, _convert_="all")
    actors_generator = (
        hydra.utils.instantiate(cfg.actors_generator) if "actors_generator" in cfg else None
    )
    controller = CollectorController(
        actors=actors,
        actors_generator=actors_generator,
        max_frames=cfg.max_frames,
        output_dir=Path(cfg.paths.data_dir),
    )

    if not controller.prepare_simulation():
        logger.error("Error while preparing the simulator.")
        return

    logger.info(f"Start collection data to {cfg.paths.data_dir}")
    controller.collect()
    controller.destroy()


if __name__ == "__main__":
    main()
