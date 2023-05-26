#
# Copyright (C) 2023 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from dataclasses import dataclass

from numpy import random

__all__ = ["ClientParameters", "SimulationParameters", "SyncParameters"]


MAX_RAND_VALUE = int("ffffffff", 16)


@dataclass
class SyncParameters:
    fps: float = 0.2
    timeout: float = 2.0


@dataclass
class ClientParameters:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout: float = 5.0
    retry: int = 5
    seed: int = random.randint(1, MAX_RAND_VALUE)


@dataclass
class SimulationParameters:
    traffic_manager_port: int = 8000
    respawn: bool = True
    townmap: str = "Town01"
