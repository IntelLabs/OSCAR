# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
import os
from pathlib import Path

import coloredlogs
from numpy import random
from tqdm import tqdm
from itertools import chain

from oscar_datagen_tools.common_utils import generate_image_filename

from . import Context, SyncMode
from .actors import Actor, Controller, Patch, Sensor
from .utils import (
    actors_still_alive,
    handle_keyboard_interrupt,
    replace_modality_in_path,
    save_dynamic_metadata,
    save_static_metadata,
)

__all__ = ["CollectorController"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class CollectorController:
    def __init__(
        self,
        actors: list[Actor] = None,
        context: Context = Context(),
        max_frames: int = 10,
        output_dir: Path = None,
    ):
        self.context = context
        self.max_frames = max_frames
        self.actors = actors
        self.output_dir = output_dir

        # collection stages flags
        self._setup_complete = False
        self._preparation_complete = False

        self._sync_mode = None

    @property
    def actors(self):
        return self._actors or []

    @actors.setter
    def actors(self, value):
        self._actors = value

        # internal sensors list
        self.sensors = [actor for actor in self.actors if isinstance(actor, Sensor)]

        # set a list of movable actors
        self.controllers = [actor for actor in self.actors if isinstance(actor, Controller)]

    def setup_simulation(self) -> bool:
        assert self.context is not None

        # setup simulator
        if not self.context.verify_connection():
            return False

        if not self.context.setup_simulator():
            return False

        self._setup_complete = True

        return True

    def prepare_simulation(self) -> bool:
        assert self.context is not None

        # Verify if dependent processes are done
        if self._preparation_complete:
            logger.warning("The actors were already spawned.")

        if not self._setup_complete:
            if not self.setup_simulation():
                logger.error("Error while setting up the simulator.")
                return False

        # init sync mode
        self._sync_mode = SyncMode(self.context)
        self._sync_mode.__enter__()

        def count_actors(actors):
            count = len(actors)
            for actor in actors:
                count += count_actors(actor.attachments)
            return count

        def batch_actors(actors):
            # Spawn patches one at a time
            patches = [actor for actor in actors if isinstance(actor, Patch)]
            for patch in patches:
                yield [patch]

            # Batch spawn non-patch actors
            yield [actor for actor in actors if not isinstance(actor, Patch)]

        # set random seed before spawn a batch of actors
        random.seed(self.context.client_params.seed)

        # Recursively spawn actors
        actors = self.actors
        total_actors = count_actors(self.actors)
        spawned_actors = []
        with tqdm(total=total_actors, desc="Actor's spawn") as pbar:
            while actors:
                for actor_batch in batch_actors(actors):
                    spawned_batch = self.context.batch_spawn(actor_batch)
                    spawned_actors += spawned_batch
                    pbar.update(len(spawned_batch))

                    # next loop we spawn any attachments
                    actors = list(chain.from_iterable(actor.attachments for actor in spawned_batch))

        # Keep only those actors that are alive
        assert all([actor.is_alive for actor in spawned_actors])
        self.actors = spawned_actors

        # setup initial position from those sensors that use the SensorController
        self.context.apply_batch_sensor_step(self.controllers)

        # save static metadata
        if self.output_dir:
            metadata_path = Path(replace_modality_in_path(self.output_dir, modality="metadata"))
            save_static_metadata(self.actors, path=metadata_path)

        self._preparation_complete = True

        return True

    @handle_keyboard_interrupt
    def collect(self, max_frame: int = None) -> bool:
        assert self.context is not None

        # Verify that the simulation is already setup
        if not self._setup_complete:
            if not self.setup_simulation():
                logger.error("Error while setting up the simulator.")
                return False

        # Verify if dependent processes are done
        if not self._preparation_complete:
            if not self.prepare_simulation():
                logger.error("Error while preparing the simulator.")
                return False

        # warm up time before start collecting
        for _ in tqdm(range(self.context.simulation_params.warmup), desc="Warmup ticks"):
            self.context.world.tick()

        # Prepare sensors to start receiving data
        self._sync_mode.prepare_sensors(self.sensors)

        local_frame_num = 0
        local_max_frames = max_frame if max_frame else self.max_frames
        with tqdm(total=local_max_frames, desc="Collection progress") as pbar:
            while True:
                local_frame_num += 1
                logger.debug("Begin sync_mode.tick")

                if local_frame_num > local_max_frames:
                    logger.debug(f"Max frames, {local_max_frames}, reached. Exiting...")
                    return True

                # Advance the simulation and wait for the data.
                snapshot = self._sync_mode.tick(self.context.sync_params.timeout)
                logger.debug(f"Snapshot: {snapshot}, Local frame: {local_frame_num}")

                # Break if any vehicle in our actor list has disappeared
                if not actors_still_alive(self.actors):
                    logger.error("Actors were found dead!")
                    return False

                assert local_max_frames >= local_frame_num

                if self.output_dir:
                    # save dynamic metadata
                    metadata_path = Path(
                        replace_modality_in_path(self.output_dir, modality="metadata")
                    )
                    save_dynamic_metadata(self.actors, local_frame_num, snapshot, path=metadata_path)

                    for sensor in self.sensors:
                        filename = generate_image_filename(
                            sensor.id,
                            sensor.transform.location.z,
                            sensor.transform.rotation.pitch
                            % 360,  # keep the angle as a positive value between 0 a 360
                            local_frame_num,
                        )

                        path = self.output_dir / filename
                        if not path.exists():
                            sensor.save_to_disk(path)

                # perform the sensor step
                self.context.apply_batch_sensor_step(self.controllers)
                pbar.update(1)

    def destroy(self) -> bool:
        if not self._preparation_complete:
            logger.warning("The simulation does not have actors yet!")
            return

        self._sync_mode.__exit__()

        commands = [actor.destroy() for actor in self.actors]
        self.context.apply_batch(commands)

        self._preparation_complete = False
        return True
