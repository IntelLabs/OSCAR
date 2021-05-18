#
# Copyright (C) 2020 Georgia Institute of Technology. All rights reserved.
# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Dict, Tuple

import torch
import torch.nn as nn

_CACHE_KEY_TYPE = Tuple[int, int]
_CACHE_VAL_TYPE = Dict[str, torch.Tensor]


class TwoDeeArgmax(nn.Module):
    def __init__(self, temperature: int = 100):
        super().__init__()

        self._indices_tensors_cache: Dict[_CACHE_KEY_TYPE, _CACHE_VAL_TYPE] = dict()

        self.temperature = temperature

    def _get_or_create_indices_tensors(self, H: int, W: int) -> _CACHE_VAL_TYPE:
        cache_key: _CACHE_KEY_TYPE = (H, W)

        if cache_key not in self._indices_tensors_cache:
            row_indices = torch.arange(H).unsqueeze(-1).repeat((1, W)).float()
            col_indices = torch.arange(W).unsqueeze(0).repeat((H, 1)).float()

            self._indices_tensors_cache[cache_key] = dict(
                row_indices=row_indices,
                col_indices=col_indices,
                row_indices_flat=row_indices.flatten(),
                col_indices_flat=col_indices.flatten(),
            )

        return self._indices_tensors_cache[cache_key]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = x.size()

        cached_tensors = self._get_or_create_indices_tensors(H, W)

        if self.training:
            row_indices = cached_tensors["row_indices"].to(x.device)
            col_indices = cached_tensors["col_indices"].to(x.device)

            x_max = x.detach().max()
            x_min = x.detach().min()

            x_scaled = self.temperature * (x - x_min) / (x_max - x_min)
            x_scaled = x_scaled.flatten().unsqueeze(0)
            x_scaled = x_scaled.softmax(dim=1)
            x_scaled = x_scaled[0].view(H, W)

            y_argmax = (row_indices * x_scaled).sum()
            x_argmax = (col_indices * x_scaled).sum()

        else:
            row_indices_flat = cached_tensors["row_indices_flat"].to(x.device)
            col_indices_flat = cached_tensors["col_indices_flat"].to(x.device)

            i = x.argmax()
            y_argmax = row_indices_flat[i]
            x_argmax = col_indices_flat[i]

        return x_argmax.clamp(0, W - 1), y_argmax.clamp(0, H - 1)


class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale):
        if len(inputs) == 0:
            return inputs

        return (scale * inputs).round().clip(0, scale) / scale

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None
