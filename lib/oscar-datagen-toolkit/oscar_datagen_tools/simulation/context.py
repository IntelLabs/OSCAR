# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import logging
from itertools import cycle

import coloredlogs
from carla import Client, Transform, WeatherParameters, WorldSettings
from numpy import random
from packaging import version

from .parameters import ClientParameters, SimulationParameters, SyncParameters
from .utils import run_request

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

    @run_request
    def verify_connection(self) -> bool:
        client_version = self.get_client_version().split("-")[0]
        server_version = self.get_server_version().split("-")[0]
        logger.info(f"CARLA client version: {client_version}")
        logger.info(f"CARLA server version: {server_version}")

        assert client_version == server_version, "CARLA server and client should have same version"
        assert version.parse(client_version) >= version.parse(
            "0.9.13"
        ), "CARLA version needs be >= '0.9.13'"

    @run_request
    def setup_simulator(self) -> bool:
        # setup simulation's world
        logger.info(f"Loading map {self.simulation_params.townmap}.")
        self.world = self.load_world(self.simulation_params.townmap)
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

        # setup spawn points
        self._spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(self._spawn_points)
        self._spawn_iter = cycle(self._spawn_points)

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

    def get_spawn_point(self, from_navigation: bool = False) -> Transform:
        spawn_point = None

        if from_navigation:
            # This points are recommended for walkers actors
            spawn_point = Transform()
            spawn_point.location = self.world.get_random_location_from_navigation()
        else:
            spawn_point = next(self._spawn_iter, None)

        return spawn_point

    def get_number_of_spawn_points(self):
        return len(self._spawn_points)
