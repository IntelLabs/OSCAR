# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
from collections import Counter
from itertools import cycle
from typing import TYPE_CHECKING, List

import carla
import coloredlogs
from carla import Client, Transform, WeatherParameters, WorldSettings
from numpy import random
from packaging import version

if TYPE_CHECKING:
    from .actors import Actor, Controller
    from .parameters import ClientParameters, SimulationParameters, SyncParameters

__all__ = ["Context"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class Context(Client):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(
        self,
        client_params: ClientParameters = None,
        simulation_params: SimulationParameters = None,
        sync_params: SyncParameters = None,
        weather_params: WeatherParameters = None,
        reinit: bool = False,
    ) -> None:
        # check if the singleton was already initialized
        if not reinit:
            return

        # setup CARLA client
        super().__init__(client_params.host, client_params.port)
        self.set_timeout(client_params.timeout)

        # setup weather
        self.weather = weather_params

        # set parameters
        self.client_params = client_params
        self.simulation_params = simulation_params
        self.sync_params = sync_params

        self.world = None
        self.sensors = []

        # private
        self._spawn_points = []
        self._blueprint_library = None

    @property
    def blueprint_library(self):
        return self._blueprint_library

    def verify_connection(self) -> bool:
        # TODO: Add a verification if the CARLA server is up and running
        client_version = self.get_client_version().split("-")[0]
        server_version = self.get_server_version().split("-")[0]
        logger.info(f"CARLA client version: {client_version}")
        logger.info(f"CARLA server version: {server_version}")

        assert client_version == server_version, "CARLA server and client should have same version"
        assert version.parse(client_version) >= version.parse(
            "0.9.13"
        ), "CARLA version needs be >= '0.9.13'"

        return True

    def setup_simulator(self) -> bool:
        # setup simulation's world
        self.world = self.get_world()

        # change map if necessary
        if self.world.get_map().name != self.simulation_params.townmap:
            logger.info(f"Loading map {self.simulation_params.townmap}.")
            self.world = self.load_world(self.simulation_params.townmap)

        # apply world settings
        self.world.apply_settings(
            WorldSettings(
                no_rendering_mode=False,
                synchronous_mode=True,
                fixed_delta_seconds=1.0 / self.sync_params.fps,
            )
        )
        self.world.set_weather(self.weather)
        self.world.set_pedestrians_seed(self.client_params.seed)
        random.seed(self.client_params.seed)
        logger.info(
            f"Simulation's world settings: no_rendering_mode={self.world.get_settings().no_rendering_mode}, "
            f"synchronous_mode={self.world.get_settings().synchronous_mode}, fixed_delta_seconds={self.world.get_settings().fixed_delta_seconds}"
        )

        # Check for unwanted actors in simulation
        # Is desirable that the simulation starts without actors
        sim_actors = self.world.get_actors()
        if len(sim_actors) > 0:
            logger.info(f"There are {len(sim_actors)} actors at the beginning of the simulation")
            actors_distribution = Counter([actor.type_id for actor in sim_actors]).most_common()
            logger.info(str(actors_distribution))

        # setup spawn points
        self._spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(self._spawn_points)
        self._spawn_iter = cycle(self._spawn_points)

        # setup blueprint library
        self._blueprint_library = self.world.get_blueprint_library()

        # setup simulator's traffic manager
        self.traffic_manager = self.get_trafficmanager(self.simulation_params.traffic_manager_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.client_params.seed)
        logger.info(
            f"Simulation's traffic manager settings: port={self.traffic_manager.get_port()}"
        )

        # Check if synchronous:
        self.world.tick()
        t_prev = self.world.get_snapshot().timestamp.elapsed_seconds
        self.world.tick()
        t_curr = self.world.get_snapshot().timestamp.elapsed_seconds
        t_between_ticks = t_curr - t_prev
        logger.info(
            f"Time between ticks: {t_between_ticks} and Delta seconds: {self.world.get_settings().fixed_delta_seconds}"
        )

        return True

    def get_spawn_point(self, from_navigation: bool = False) -> Transform:
        spawn_point = None

        if from_navigation:
            # This points are recommended for walkers actors
            spawn_point = Transform()
            spawn_point.location = self.world.get_random_location_from_navigation()
        else:
            spawn_point = next(self._spawn_iter, None)

        return spawn_point

    def batch_spawn(self, actors: list[Actor]) -> bool:
        # Spawn actors using commands and tick simulator
        spawns = [actor.spawn() for actor in actors]
        commands = [next(spawn) for spawn in spawns]
        responses = self.apply_batch_sync(commands, do_tick=True)

        spawned_actors = []
        for actor, spawn, response in zip(actors, spawns, responses):
            try:
                ret = spawn.send(response)
            except StopIteration as ex:
                if ex.value:
                    spawned_actors.append(actor)

        # return actors that successfully spawned
        return spawned_actors

    def apply_batch_sensor_step(self, controllers: list[Controller]) -> bool:
        commands = [controller.step() for controller in controllers]
        commands = [command for command in commands if command]

        # update camera position
        # NOTE: The 'do_tick' parameter is set to True to perform
        # a tick after the batch is applied. The 'do_tick' parameter by
        # default is False.
        for response in self.apply_batch_sync(commands, do_tick=True):
            if response.error:
                logging.error(f"Step failed: {response.error}")
                return False

        return True
