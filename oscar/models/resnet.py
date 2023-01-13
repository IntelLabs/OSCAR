#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import mart.utils

logger = mart.utils.get_pylogger(__name__)

import torch
from torch.nn import Conv2d, init
from torchvision.models.resnet import ResNet as ResNet_

class ResNet(ResNet_):
    """
    Adds ability to set number of in_channels in ResNet stem and load
    pre-trained weights.

    Args:
        in_channels (int): number of input channels (default: 3)
    """
    def __init__(
        self,
        *args,
        in_channels: int = 3,
        weights = None,
        progress: bool = True,
        strict: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if weights:
            self.load_state_dict(weights.get_state_dict(progress=progress), strict=strict)

        # Check if we need to replace conv1
        init_in_channels = self.conv1.in_channels

        if in_channels != init_in_channels:
            logger.warning(f"Replacing conv1's in_channels from {init_in_channels} to {in_channels}")

            init_weight = self.conv1.weight

            self.conv1 = Conv2d(in_channels,
                                self.conv1.out_channels,
                                kernel_size=self.conv1.kernel_size,
                                stride=self.conv1.stride,
                                padding=self.conv1.padding,
                                bias=self.conv1.bias)


            if in_channels == 1:
                logger.warning("Initializing conv1's weights to the average of the initial weights")
                init_weight = init_weight.sum(dim=1, keepdim=True)
                self.conv1.weight.data.copy_(init_weight)

            elif in_channels % init_in_channels == 0:
                logger.warning("Initializing conv1's weights by replicating the initial weights")
                init_weight = init_weight.repeat(1, in_channels//init_in_channels, 1, 1)
                init_weight /= in_channels // init_in_channels
                self.conv1.weight.data.copy_(init_weight)

            elif in_channels == 4:
                logger.warning("Initializing conv1's weights by copying and averaging")
                init_weight = torch.cat([init_weight, init_weight.mean(dim=1, keepdim=True)], dim=1)
                init_weight /= in_channels / init_in_channels
                self.conv1.weight.data.copy_(init_weight)

            else:
                logger.warning(f"Initializing conv1's weights randomly ({init_weight.shape} -?> {self.conv1.weight.shape})")
                init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
