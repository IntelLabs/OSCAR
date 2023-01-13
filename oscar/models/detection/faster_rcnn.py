#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Any, Optional

import logging
logger = logging.getLogger(__name__)

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FasterRCNN as FasterRCNN_
from torchvision.models.detection.image_list import ImageList

from torchvision.ops import misc as misc_nn_ops
from torchvision.models._utils import _ovewrite_value_param
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers

from oscar.utils.monkey_patch import MonkeyPatch
from oscar.models.detection.rcnn.rpn import TorchvisionRPN
from oscar.models.detection.roi_heads import RoIHeads, FastRCNNPredictor
from oscar.utils.layers import quantize
from oscar.utils.carla import log_to_linear, linear_to_log
from mart.nn import SequentialDict

class ModularFasterRCNN(SequentialDict):
    def __init__(
        self,
        sequence=None,
        model=None,
        **modules
    ):
        if model is not None:
            logger.warning("Ignoring 'model' module; it won't be used. This is an artifact of bad configuration in MART.")

        sequences = {
            "training": sequence,
        }

        super().__init__(modules, sequences=sequences)

    def forward(self, images, targets=None):
        ret = {}

        # Run the training sequence since it should output everything
        ret["training"] = super().forward(input=images, target=targets, step="training")
        # Mimic torchvision's outputs
        ret["eval"] = ret["training"].pop("preds")

        return ret

class FasterRCNN(FasterRCNN_):
    """
    A more composable FasterRCNN by allowing injection of RPN and RoIHeads

    Args:
        rpn (nn.Module): module to replace default region proposal network (default: None)
        interpolation (str): interpolation to use in transform (default: bilinear)
        input_slice (slice): which channels of input to select (default: all channels)
    """
    def __init__(
        self,
        *args,
        rpn_score_thresh=0.,
        rpn=None,
        interpolation="bilinear",
        input_slice=None,
        quant_kwargs=None,
        **kwargs
    ):
        with MonkeyPatch(torchvision.models.detection.faster_rcnn, "RoIHeads", RoIHeads, verbose=True):
            with MonkeyPatch(torchvision.models.detection.faster_rcnn, "FastRCNNPredictor", FastRCNNPredictor):
                with MonkeyPatch(torchvision.models.detection.faster_rcnn, "RegionProposalNetwork", TorchvisionRPN, verbose=True):
                    super().__init__(*args, **kwargs)

        if rpn is not None:
            logger.info("Replacing RPN with %s", rpn.__class__.__name__)
            self.rpn = rpn

        self.rpn.score_thresh = rpn_score_thresh

        # FIXME: Add non-binlinear interpolation
        if interpolation != "bilinear":
            logger.warn("Using bilinear interpolation instead of %s", interpolation)

        self.input_slice = input_slice
        if self.input_slice is not None:
            self.input_slice = slice(*self.input_slice)

        self.quant_kwargs = quant_kwargs

    def forward(self, x, targets=None):
        # Create new image_list by selecting slice of input
        if self.input_slice is not None:
            x = [img[self.input_slice] for img in x]

        # Quantize input
        if self.quant_kwargs is not None:
            # Convert to normalized [0-1] linear depth and quantize.
            x = [log_to_linear(img) for img in x]
            x = [quantize(img, **self.quant_kwargs) for img in x]

            # Convert back to log depth and...
            x = [linear_to_log(img) for img in x]
            # ...quantize to ~8-bits (clip to so there are no 0 depth values)
            x = [quantize(img, scale=255, clip_min=1/255) for img in x]

        return super().forward(x, targets=targets)



class DetectorRPN(FasterRCNN):
    """
        A detector that acts like a proposer. Uses the last N channels (i.e., depth) to make proposals.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, images, ignored_features, targets=None):
        # Create new image_list by selecting slice of input
        if self.input_slice is not None:
            images = ImageList(images.tensors[:, self.input_slice], images.image_sizes)

        # Get features from depth-trained backbone
        features = self.backbone(images.tensors)
        proposals, losses = self.rpn(images, features, targets)

        return proposals, losses


# Wrapper function for custom FasterRCNN model
def fasterrcnn_resnet50_fpn(
    *,
    weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNN:
    """
    Reuses part of https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py#L467
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else torch.nn.BatchNorm2d

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if weights == FasterRCNN_ResNet50_FPN_Weights.COCO_V1:
            overwrite_eps(model, 0.0)

    return model
