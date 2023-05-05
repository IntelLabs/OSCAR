#
# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import logging
import time

import carla
import coloredlogs

__all__ = ["SyncMode"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class SyncMode:
    """Context manager to synchronize output from different sensors.

    Synchronous
    mode is enabled as long as we are inside this context
        with SyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, context, sensors):
        assert context is not None

        self.context = context
        self.sensors = sensors
        self.delta_seconds = 1.0 / self.context.sync_params.fps
        self.frame = None
        self._settings = None

        logger.info(f"Init sensors: {self.sensors}, kwargs: {self.context.sync_params.fps}")

        assert self.context.world is not None
        assert self.context.traffic_manager is not None
        self.context.traffic_manager.set_respawn_dormant_vehicles(
            self.context.simulation_params.respawn
        )

    def __enter__(self):
        self._settings = self.context.world.get_settings()
        self.frame = self.context.world.apply_settings(
            carla.WorldSettings(
                no_rendering_mode=False,
                synchronous_mode=True,
                fixed_delta_seconds=self.delta_seconds,
            )
        )
        self.context.traffic_manager.set_synchronous_mode(True)
        self.context.traffic_manager.set_random_device_seed(self.context.client_params.seed)
        time.sleep(1)

        # Start listening sensor's data
        for sensor in self.sensors:
            sensor.start_listening()

        return self

    def tick(self, timeout) -> carla.WorldSnapshot:
        self.frame = self.context.world.tick()

        # wait for simulation data
        snapshot = self.context.world.get_snapshot()
        assert self.frame == snapshot.frame

        for sensor in self.sensors:
            # wait for sensors data
            sensor.wait_sensors_data(self.frame, timeout)

        return snapshot

    def __exit__(self, *args, **kwargs) -> None:
        self.context.world.apply_settings(self._settings)
        self.context.traffic_manager.set_synchronous_mode(False)

        for sensor in self.sensors:
            sensor.stop_listening()
