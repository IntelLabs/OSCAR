#
# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import logging
from pathlib import Path
from typing import List

import coloredlogs
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from oscar_datagen_tools.common_utils import generate_image_filename
from oscar_datagen_tools.simulation import Context, SyncMode
from oscar_datagen_tools.simulation.actors import Actor, ActorsGenerator, ISensor
from oscar_datagen_tools.simulation.utils import (
    actors_still_alive,
    run_loop,
    save_dynamic_metadata,
    save_static_metadata,
)

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class CollectorController:
    def __init__(
        self,
        actors: List[Actor],
        context: Context = Context(),
        actors_generator: ActorsGenerator = None,
        max_frames: int = 10,
    ):
        self.context = context
        self.actors_generator = actors_generator
        self.max_frames = max_frames
        self.actors = actors
        self.sensors = []

    def prepare_simulation(self) -> bool:
        assert self.context is not None

        # setup simulator
        if not self.context.verify_connection():
            return False

        if not self.context.setup_simulator():
            return False

        if self.actors_generator is not None:
            # add auto generated actors
            vehicles = self.actors_generator.generate_vehicles()
            walkers = self.actors_generator.generate_walkers()
            self.actors = self.actors + vehicles + walkers

        # setup actors
        for actor in self.actors:
            # add the actor's attachments to the actor's list
            self.actors += actor.attachments

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
        save_static_metadata(self.actors)

        return True

    @run_loop
    def collect(self) -> None:
        assert self.context is not None

        with SyncMode(self.context, self.sensors) as sync_mode:
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
                save_dynamic_metadata(self.actors, local_frame_num, snapshot)

                for sensor_idx, sensor in enumerate(self.sensors):
                    # calculate an id to the sensor's data, that does not repeat
                    # among the different sensors, starting from 1.
                    image_id = (sensor_idx * self.max_frames) + local_frame_num
                    filename = Path(sensor.name) / generate_image_filename(image_id)
                    path = Path(HydraConfig.get().run.dir) / filename

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
    hydra.utils.instantiate(cfg.context, reinit=True)

    # Convert all to primitive containers when instantiating the actors, specially
    # for the Blueprint's Sensor data type where the Camera's init method verify
    # this instance type.
    actors = hydra.utils.instantiate(cfg.spawn_actors, _convert_="all")
    actors_generator = (
        hydra.utils.instantiate(cfg.actors_generator) if "actors_generator" in cfg else None
    )
    controller = CollectorController(
        actors=actors, actors_generator=actors_generator, max_frames=cfg.max_frames
    )

    if not controller.prepare_simulation():
        logger.error("Error while preparing the simulator.")
        return

    controller.collect()
    controller.destroy()


if __name__ == "__main__":
    main()
