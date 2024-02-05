# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
from typing import List

import coloredlogs
from numpy import random
from hydra.utils import instantiate

from ..context import Context

__all__ = ["ActorsGenerator"]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)


class ActorsGenerator:
    def __new__(
        cls,
        count: int,
        _actor_: str,
        context: Context = Context(),
        seed: int = None,
        **kwargs,
    ):
        # Reseed random number generators so that these actors are generated deterministically
        if seed is not None:
            random.seed(seed)

        for _ in range(count):
            yield instantiate({"_target_": _actor_, **kwargs}, _convert_="all")
