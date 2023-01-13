#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torchvision

from torch import Tensor
from torch.jit.annotations import BroadcastingList2
from torchvision.ops.poolers import MultiScaleRoIAlign as MultiScaleRoIAlign_

from oscar.utils.monkey_patch import MonkeyPatch

class MultiScaleRoIAlign(MultiScaleRoIAlign_):
    """
        Adds ability to specify aligned parameter, i.e., RoIAlignV2.

            aligned (bool): whether to used aligned RoIs (default: False)
    """
    def __init__(self, *args, aligned=False, **kwargs):
        super().__init__(*args, **kwargs)

        if aligned:
            self.aligned = self.roi_align_v2
        else:
            self.aligned = torchvision.ops.roi_align

    def forward(self, *args, **kwargs):
        with MonkeyPatch(torchvision.ops.poolers, "roi_align", self.aligned):
            ret = super().forward(*args, **kwargs)

        return ret

    @staticmethod
    def roi_align_v2(input: Tensor,
                     boxes: Tensor,
                     output_size: BroadcastingList2[int],
                     spatial_scale: float = 1.0,
                     sampling_ratio: int = -1,
                     aligned: bool = True):
        return torchvision.ops.roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned)
