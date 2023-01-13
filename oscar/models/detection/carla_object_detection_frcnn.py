#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
logger = logging.getLogger(__name__)

from typing import Optional

from art.estimators.object_detection import PyTorchFasterRCNN
import torch

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, FeaturePyramidNetwork
from torchvision.models.resnet import Bottleneck

from armory.data.utils import maybe_download_weights_from_s3
from armory.baseline_models.pytorch.carla_multimodality_object_detection_frcnn import MultimodalNaive
from armory.baseline_models.pytorch.carla_multimodality_object_detection_frcnn_robust_fusion import MultimodalRobust

from oscar.models.resnet import ResNet
from oscar.models.detection.faster_rcnn import FasterRCNN, DetectorRPN
from oscar.models.detection.backbone_utils import BackboneWithNeck, PyramidalFeatureNetwork
from oscar.ops.poolers import MultiScaleRoIAlign
from mart.nn.nn import GroupNorm32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_resnet_backbone(name, **kwargs):
    if name == "multimodal_naive":
        return MultimodalNaive()

    if name == "multimodal_robust":
        return MultimodalRobust()

    if name == "resnet50_fpn":
        neck_kwargs = kwargs.pop("neck_kwargs", {})
        backbone = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        assert backbone.inplanes == 2048
        return BackboneWithNeck(backbone,
                                return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
                                in_channels_list=[256, 512, 1024, 2048],
                                out_channels=256,
                                neck_class=FeaturePyramidNetwork,
                                **neck_kwargs)

    if name == "resnet50gn_fpn":
        neck_kwargs = kwargs.pop("neck_kwargs", {})
        backbone = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=GroupNorm32, **kwargs)
        assert backbone.inplanes == 2048
        return BackboneWithNeck(backbone,
                                return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
                                in_channels_list=[256, 512, 1024, 2048],
                                out_channels=256,
                                neck_class=FeaturePyramidNetwork,
                                **neck_kwargs)

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
        rpn = create_and_load_detector(DetectorRPN, **rpn_kwargs)
        model_kwargs["rpn"] = rpn

    # Load detector
    model = create_and_load_detector(FasterRCNN, **model_kwargs)

    if model_weights_path is not None:
        state_dict = torch.load(model_weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

    return PyTorchFasterRCNN(model, clip_values=(0.0, 1.0), channels_first=False, **wrapper_kwargs)
