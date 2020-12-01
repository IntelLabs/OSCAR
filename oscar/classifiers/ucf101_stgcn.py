#
# Copyright (C) 2020 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
from pathlib import Path
from typing import Tuple, Union

from art.classifiers import PyTorchClassifier
import torch
import torch.nn as nn

from mmskeleton.models.backbones.st_gcn_aaai18 import ST_GCN_18

from oscar.utils.transforms.keypoints import BatchStandardizeSTGCNInputTimesteps


logger = logging.getLogger(__name__)


class CustomSTGCN(ST_GCN_18):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        graph_layout: str = "openpose",
        graph_strategy: str = "spatial",
        edge_importance_weighting: bool = True,
    ):
        super().__init__(
            num_class=num_classes,
            in_channels=in_channels,
            edge_importance_weighting=edge_importance_weighting,
            graph_cfg={"layout": graph_layout, "strategy": graph_strategy},
        )

        self.input_transform = BatchStandardizeSTGCNInputTimesteps()

    def load_weights(self, weights_path: Union[Path, str], drop_fcn: bool = False):
        weights_path = Path(weights_path).resolve()

        state_dict = torch.load(weights_path)

        # if weights are a PyTorch Lightning checkpoint
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {
                k.replace("stgcn.", ""): v
                for k, v in state_dict.items()
                if k.startswith("stgcn")
            }

        # if loading pretrained model trained on 400 classes
        if drop_fcn:
            del state_dict["fcn.weight"]
            del state_dict["fcn.bias"]

        strict_mode = not drop_fcn

        self.load_state_dict(state_dict, strict=strict_mode)
        logger.info(f"Loaded STGCN weights from `{weights_path}`")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_transform(x)
        return super().forward(x)

    def extract_feature(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_transform(x)
        return super().extract_feature(x)


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: str
) -> PyTorchClassifier:

    if "num_classes" not in model_kwargs:
        model_kwargs.update(num_classes=101)

    stgcn = CustomSTGCN(**model_kwargs).eval()
    stgcn.load_weights(weights_path)

    wrapped_model = PyTorchClassifier(
        model=stgcn,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 300, 18, 1),
        nb_classes=101,
    )

    return wrapped_model
