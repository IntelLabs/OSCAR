#
# Copyright (C) 2020 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

import math
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlipColorMode(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, H, W, C = x.size()
        assert C == 3

        return x[:, :, :, [2, 1, 0]]


class BGR2RGB(FlipColorMode):
    pass


class RGB2BGR(FlipColorMode):
    pass


class NormalizeColorSpace(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 255.0)
        return x / 255.0


class UnNormalizeColorSpace(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        return x * 255.0


class ChannelsFirst(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, H, W, C = x.size()
        assert C == 3

        return x.permute(0, 3, 1, 2)


class Resize(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.size = size
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, C, H, W = x.size()
        assert C == 3

        return F.interpolate(x, self.size, mode=self.mode, align_corners=False)


class StandardizeTimesteps(nn.Module):
    def __init__(self, timesteps: int):
        super().__init__()

        self.timesteps = timesteps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        ndim = x.ndimension()

        repeat = math.ceil(float(self.timesteps) / T)
        if repeat >= 2:
            repeat_args = [repeat] + ([1] * (ndim - 1))
            x = x.repeat(*repeat_args)
        x = x[: self.timesteps]

        return x
