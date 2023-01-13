#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
from torch import Tensor
from typing import List

from torchvision.models.detection.anchor_utils import AnchorGenerator as AnchorGenerator_
from torchvision.models.detection.image_list   import ImageList

class AnchorGenerator(AnchorGenerator_, torch.nn.Module):
    """
    Anchor generator that takes as input (sizes, aspect_ratios, priors) for
    each resolution level.
    Returns anchors and prior for each anchor.

    Unlike the torchvision implementation, when priors are requested, anchor
    generation is 'zipped' instead of Cartesian product between (sizes, aspect_ratios).
    """

    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
        priors=((-1, -1, -1),),
        use_priors=False
    ):

        torch.nn.Module.__init__(self)
### BEGIN CHANGE
        self.use_priors = use_priors
        if self.use_priors:
            # Check that all sizes, aspect ratios and priors have correct sizes
            assert len(sizes) == len(aspect_ratios) == len(priors), 'Invalid inputs'
            for idx in range(len(sizes)):
                assert len(sizes[idx]) == len(aspect_ratios[idx]) == len(priors[idx]), \
                    'Invalid inputs at resolution level %d' % idx
        else:
### END CHANGE
            if not isinstance(sizes[0], (list, tuple)):
                sizes = tuple((s,) for s in sizes)
            if not isinstance(aspect_ratios[0], (list, tuple)):
                aspect_ratios = (aspect_ratios,) * len(sizes)

        self.priors = torch.tensor(priors, dtype=torch.long)
        self.sizes  = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio) for size, aspect_ratio in zip(sizes, aspect_ratios)
        ]

    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

### BEGIN CHANGE
        if self.use_priors:
            ws = (w_ratios * scales).view(-1)
            hs = (h_ratios * scales).view(-1)
        else:
            # Legacy
            ws = (w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h_ratios[:, None] * scales[None, :]).view(-1)
### END CHANGE

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def num_anchors_per_location(self) -> List[int]:
### BEGIN CHANGE
        if self.use_priors:
            return [len(s) for s in self.sizes]
        else:
            # Legacy
            return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]
### END CHANGE

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
### BEGIN CHANGE
        anchors, global_shifts, priors = [], [], []
### END CHANGE
        cell_anchors = self.cell_anchors
        torch._assert(cell_anchors is not None, "cell_anchors should not be None")
        torch._assert(
            len(grid_sizes) == len(strides) == len(cell_anchors),
            "Anchors should be Tuple[Tuple[int]] because each feature "
            "map could potentially have different sizes and aspect ratios. "
            "There needs to be a match between the number of "
            "feature maps passed and the number of sizes / aspect ratios specified.",
        )

        for idx, (size, stride, base_anchors) in \
            enumerate(zip(grid_sizes, strides, cell_anchors)):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
            global_shifts.append(shifts)
### BEGIN CHANGE
            if self.use_priors:
                local_priors  = self.priors[idx].repeat(shifts.shape[0])
                priors.append(local_priors)
            else:
                # Append dummy '-1' priors (no class restrictions)
                priors.append(-torch.ones(shifts.shape[0] * base_anchors.shape[0],
                                          device=device))

        return anchors, priors
### END CHANGE

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),
            ]
            for g in grid_sizes
        ]

        self.set_cell_anchors(dtype, device)
### BEGIN CHANGE
        anchors_over_all_feature_maps, \
            priors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []
        priors: List[List[torch.Tensor]]  = []
        for _ in range(len(image_list.image_sizes)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            priors_in_image  = [priors_per_feature_map for priors_per_feature_map in priors_over_all_feature_maps]
            anchors.append(anchors_in_image)
            priors.append(priors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        priors  = [torch.cat(priors_per_image) for priors_per_image in priors]
### END CHANGE
        return anchors, priors
