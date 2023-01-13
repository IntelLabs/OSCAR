#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

__all__ = ["ToDepth"]


class ToDepth:
    def __init__(self, far=1000, dim=0, scale=255):
        self.far = far
        self.dim = dim
        self.scale = scale

    def __call__(self, tensor, **kwargs):
        assert tensor.shape[self.dim] == 3
        assert 0 <= tensor.min() and tensor.max() <= 1.

        r, g, b = tensor.split(1, dim=self.dim)

        depth = (self.scale*(r*(256**0) +
                             g*(256**1) +
                             b*(256**2)) /
                 (256**3 - 1))

        depth = self.far * depth

        return depth
