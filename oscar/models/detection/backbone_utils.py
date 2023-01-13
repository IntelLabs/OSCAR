#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
import torchvision

from collections import OrderedDict

from torchvision.models.detection.backbone_utils import BackboneWithFPN as BackboneWithFPN_
from torchvision.models.detection.backbone_utils import FeaturePyramidNetwork

from oscar.utils.monkey_patch import MonkeyPatch

import mart.utils

logger = mart.utils.get_pylogger(__name__)


class MultiBackbone(torch.nn.ModuleDict):
    def __init__(self, **modules):
        super().__init__(modules)

    def forward(self, x):
        outputs = {}
        for name, module in self.items():
            outputs[name] = module(x)

        return outputs

class BackboneWithFPN(BackboneWithFPN_):
    """
    Add ability to specify trainable layers.

    See torchvision.models.detection.backbone_utils.resnet_fpn_backbone
    """
    def __init__(
        self,
        backbone,
        *args,
        trainable_layers: int = 3,
        channel_slice = None,
        **kwargs
    ):
        super().__init__(backbone, *args, **kwargs)

        # select layers that wont be frozen
        assert 0 <= trainable_layers <= 5
        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
        if trainable_layers == 5:
            layers_to_train.append('bn1')
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        # Torchvision's BackboneWithFPN has a default extra_block when None.
        # We override it here if we explicitly set None.
        if "extra_blocks" in kwargs:
            if kwargs["extra_blocks"] is None:
                logger.warning("Setting fpn.extra_blocks to None")
                self.fpn.extra_blocks = None

        self.channel_slice = channel_slice
        if self.channel_slice is not None:
            self.channel_slice = slice(*self.channel_slice)

    def forward(self, x):
        if self.channel_slice is not None:
            x = x[:, self.channel_slice]
        return super().forward(x)

class BackboneWithNeck(BackboneWithFPN):
    """
    Adds ability to use differnent neck/pyramid class.

    Args:
        neck_class (nn.Module): neck to use (e.g., torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork)
    """
    def __init__(self, *args, neck_class, **kwargs):
        with MonkeyPatch(torchvision.models.detection.backbone_utils, "FeaturePyramidNetwork", neck_class):
            super().__init__(*args, **kwargs)

class PyramidalFeatureNetwork(FeaturePyramidNetwork):
    """
        Like FeaturePyramidNetwork neck but without the top-down connections.

        See: torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork
    """
    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())

        results = []
        for idx in range(len(x)):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            results.append(self.get_result_from_layer_blocks(inner_lateral, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out
