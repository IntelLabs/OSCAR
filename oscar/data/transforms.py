#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#
import logging

import torch

from torchvision.transforms import transforms as T

logger = logging.getLogger(__name__)


class ExTransform:
    pass


class ExCompose(T.Compose, ExTransform):
    def __init__(self, transforms, return_kwargs=False):
        super().__init__(transforms)
        self.return_kwargs = return_kwargs

    def __call__(self, x_or_dict, **kwargs):
        if isinstance(x_or_dict, dict):
            x = x_or_dict.pop('x')
            kwargs = {**kwargs, **x_or_dict}
        else:
            x = x_or_dict

        for transform in self.transforms:
            if transform is None:
                continue

            if isinstance(transform, ExTransform):
                x = transform(x, **kwargs)
            else:
                x = transform(x)

        kwargs['x'] = x
        if self.return_kwargs:
            x = kwargs

        return x


class ExLambda(T.Lambda, ExTransform):
    def __init__(self, lambd):
        super().__init__(lambd)

    def __call__(self, tensor, **kwargs):
        return self.lambd(tensor, **kwargs)


class ExSplitLambda(ExLambda):
    def __init__(self, lambd, split_size_or_sections, lambd_section=0, dim=0):
        super().__init__(lambd)
        self.split_size_or_sections = split_size_or_sections
        self.lambd_section = lambd_section
        self.dim = dim

    def __call__(self, tensor, **kwargs):
        sections = list(torch.split(tensor, self.split_size_or_sections, dim=self.dim))
        sections[self.lambd_section] = super().__call__(sections[self.lambd_section], **kwargs)
        tensor = torch.cat(sections, self.dim)

        return tensor


class Cat:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        return torch.cat(tensors, dim=self.dim)


class Permute:
    def __init__(self, *dims):
        self.dims = dims

    def __call__(self, tensor):
        return tensor.permute(*self.dims)


class Unsqueeze:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        return tensor.unsqueeze(self.dim)


class Squeeze:
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, tensor):
        return tensor.squeeze(self.dim)


# TODO: Add Cat transform
# FIXME: Change to Split transform
class Chunk:
    def __init__(self, chunks, dim=0):
        self.chunks = chunks
        self.dim = dim

    def __call__(self, tensor):
        chunks = tensor.chunk(self.chunks, dim=self.dim)
        #print("chunks =", type(chunks))
        return [*chunks] # tuple -> list
