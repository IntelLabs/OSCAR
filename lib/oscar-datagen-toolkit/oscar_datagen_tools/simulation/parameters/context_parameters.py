#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
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
    warmup: int = 150
    interval: int = 1
