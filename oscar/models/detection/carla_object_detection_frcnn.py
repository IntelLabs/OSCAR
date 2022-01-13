#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
logger = logging.getLogger(__name__)

from typing import Optional
from collections import OrderedDict

from art.estimators.object_detection import PyTorchFasterRCNN
import torch
from torch import Tensor
from torch.jit.annotations import BroadcastingList2

import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, BackboneWithFPN, FeaturePyramidNetwork
from torchvision.models.detection.image_list import ImageList
from torchvision.models.resnet import Bottleneck

from armory.data.utils import maybe_download_weights_from_s3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MonkeyPatch:
    """ Temporarily replace a module's object value, i.e., its functionality """
    def __init__(self, obj, name, value, verbose=False):
        self.obj = obj
        self.name = name
        self.value = value
        self.verbose = verbose

    def __enter__(self):
        self.orig_value = getattr(self.obj, self.name)

        if self.orig_value == self.value:
            return

        if self.verbose:
            logger.info("Monkey patching %s to %s", self.orig_value, self.value)

        setattr(self.obj, self.name, self.value)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.orig_value == self.value:
            return

        if self.verbose:
            logger.info("Reverting monkey patch on %s", self.orig_value)

        setattr(self.obj, self.name, self.orig_value)

class FasterRCNN(torchvision.models.detection.faster_rcnn.FasterRCNN):
    """
        A more composable FasterRCNN by allowing injection of an RPN

            rpn (nn.Module): module to replace default region proposal network (default: None)
            interpolation (str): interpolation to use in transform (default: bilinear)
    """
    def __init__(self, *args, rpn_score_thresh=0., rpn=None, interpolation="bilinear", **kwargs):
        with MonkeyPatch(torchvision.models.detection.faster_rcnn, "RegionProposalNetwork", RegionProposalNetwork):
            super().__init__(*args, **kwargs)

        if rpn is not None:
            logger.info("Replacing RPN with %s", rpn.__class__.__name__)
            self.rpn = rpn

        self.rpn.score_thresh = rpn_score_thresh

        # FIXME: Add non-binlinear interpolation
        if interpolation != "bilinear":
            logger.warn("Using bilinear interpolation instead of %s", interpolation)

    def forward(self, x, targets=None):
        # Create new image_list by selecting first N channels
        x = [img[:len(self.transform.image_mean)] for img in x]

        return super().forward(x, targets=targets)

class RegionProposalNetwork(torchvision.models.detection.rpn.RegionProposalNetwork):
    """
        Adds ability to filter region proposals by score threshold.

            score_thresh (float): only propose regions with objectness score >= score_thresh (default: 0.0)
    """
    def __init__(self, *args, score_thresh=0.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.score_thresh = score_thresh

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        boxes, scores = super().filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level)

        # Keep boxes with score > score_thresh
        top_boxes = []
        top_scores = []
        for b, s in zip(boxes, scores):
            keep = torch.sigmoid(s) >= self.score_thresh
            top_boxes.append(b[keep])
            top_scores.append(s[keep])

        return top_boxes, top_scores

class GroupNorm32(torch.nn.GroupNorm):
    """
        GroupNorm with default num_groups=32; can be pass to ResNet's norm_layer.

        See: torch.nn.GroupNorm
    """
    def __init__(self, *args, **kwargs):
        super().__init__(32, *args, **kwargs)

class MultiScaleRoIAlign(torchvision.ops.poolers.MultiScaleRoIAlign):
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

class ResNet(torchvision.models.resnet.ResNet):
    """
        Adds ability to set number of in_channels in ResNet stem.

            in_channels (int): number of input channels (default: 3)
    """
    def __init__(self, *args, in_channels=3, **kwargs):
        super().__init__(*args, **kwargs)

        # Check if we need to replace conv1
        if in_channels != self.conv1.in_channels:
            self.conv1 = torch.nn.Conv2d(in_channels,
                                         self.conv1.out_channels,
                                         kernel_size=self.conv1.kernel_size,
                                         stride=self.conv1.stride,
                                         padding=self.conv1.padding,
                                         bias=self.conv1.bias)

class BackboneWithNeck(BackboneWithFPN):
    """
        Adds ability to use differnent neck/pyramid class.

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

class DepthDetectorRPN(FasterRCNN):
    """
        A detector that acts like a proposer. Uses the last N channels (i.e., depth) to make proposals.
    """
    def forward(self, images, ignored_features, targets=None):
        # Create new image_list by selecting depth channels (last N channels)
        images = ImageList(images.tensors[:, -len(self.transform.image_mean):], images.image_sizes)

        # Get features from depth-trained backbone
        features = self.backbone(images.tensors)
        proposals, losses = self.rpn(images, features, targets)

        return proposals, losses

def create_resnet_backbone(name, **kwargs):
    if name == "resnet50_fpn":
        return resnet_fpn_backbone("resnet50", pretrained=False, **kwargs)

    if name == "resnet50gn_fpn":
        backbone = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=GroupNorm32, **kwargs)
        assert backbone.inplanes == 2048
        return BackboneWithNeck(backbone,
                                return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
                                in_channels_list=[256, 512, 1024, 2048],
                                out_channels=256,
                                neck_class=FeaturePyramidNetwork)

    if name == "resnet50gn_pfn":
        backbone = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=GroupNorm32, **kwargs)
        assert backbone.inplanes == 2048
        return BackboneWithNeck(backbone,
                                return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
                                in_channels_list=[256, 512, 1024, 2048],
                                out_channels=256,
                                neck_class=PyramidalFeatureNetwork)

    raise Exception(f"Unknown backbone name: {name}")

def create_and_load_detector(detector_klass, weights_file=None, box_roi_pool=None, **kwargs):
    backbone = create_resnet_backbone(**kwargs.pop("backbone"))

    kwargs.setdefault("num_classes", 4) # default number of CARLA classes
    if box_roi_pool == "RoIAlignV2":
        box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                          output_size=7,
                                          sampling_ratio=2,
                                          aligned=True)
        kwargs["box_roi_pool"] = box_roi_pool

    detector = detector_klass(backbone, **kwargs)

    if weights_file is not None:
        weights_path = maybe_download_weights_from_s3(weights_file)
        state_dict = torch.load(weights_path, map_location=DEVICE)
        detector.load_state_dict(state_dict, strict="rpn" not in kwargs)

    return detector

def get_art_model(model_kwargs: dict, wrapper_kwargs: dict, model_weights_path: Optional[str] = None) -> PyTorchFasterRCNN:
    # Copy because we're going to modify them
    model_kwargs = model_kwargs.copy()

    # Load region proposal network
    rpn_kwargs = model_kwargs.pop("rpn", None)
    if rpn_kwargs is not None:
        assert isinstance(rpn_kwargs, dict)
        rpn = create_and_load_detector(DepthDetectorRPN, **rpn_kwargs)
        model_kwargs["rpn"] = rpn

    # Load detector
    model = create_and_load_detector(FasterRCNN, **model_kwargs)

    if model_weights_path is not None:
        state_dict = torch.load(model_weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

    return PyTorchFasterRCNN(model, clip_values=(0.0, 1.0), channels_first=False, **wrapper_kwargs)
