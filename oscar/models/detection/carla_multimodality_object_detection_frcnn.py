#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
logger = logging.getLogger(__name__)

from typing import Optional, List, Dict

import numpy as np
from armory.baseline_models.pytorch.carla_multimodality_object_detection_frcnn import MultimodalNaive
from armory.baseline_models.pytorch.carla_multimodality_object_detection_frcnn_robust_fusion import MultimodalRobust

from art.estimators.object_detection import PyTorchFasterRCNN as _PyTorchFasterRCNN
import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from oscar.defences.postprocessor.distance_estimator import DistanceEstimator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PyTorchFasterRCNN(_PyTorchFasterRCNN):
    def __init__(self, model,
                       clip_values=(0.0, 1.0),
                       channels_first=False,
                       num_classes=4,
                       **wrapper_kwargs,
                ) -> None:

        self.distance_estimator = None
        if wrapper_kwargs is not None:
            estimator_kwargs = wrapper_kwargs.pop('distance_estimator', None)
            if estimator_kwargs is not None:
                intercept = estimator_kwargs.pop('intercept', None)
                slope = estimator_kwargs.pop('slope', None)
                confidence = estimator_kwargs.pop('confidence', [0.9, 0.9, 0.9])
                penality_ratio = estimator_kwargs.pop('penality_ratio', 0.9)
                logarithmic = estimator_kwargs.pop('logarithmic', True)
                reciprocal = estimator_kwargs.pop('reciprocal', True)

                if intercept and slope:
                    # Current classes are [Pedestrian, Vehicle, TrafficLight, Patch]
                    # We only examine the first three classes, but not Patch class
                    assert (len(intercept) == num_classes - 1 and len(slope) == num_classes - 1)
                    self.distance_estimator = DistanceEstimator(
                                                intercept=intercept,
                                                slope=slope,
                                                confidence=confidence,
                                                penality_ratio=penality_ratio,
                                                logarithmic=logarithmic,
                                                reciprocal=reciprocal
                                                )
        super().__init__(model=model,
                         clip_values=clip_values,
                         channels_first=channels_first,
                         **wrapper_kwargs,
                         )

    def predict(self, x, batch_size=128):
        predictions = super().predict(x, batch_size)

        if self.distance_estimator is not None:
            predictions = self.distance_estimator.update_score(x, predictions)

        return predictions

def get_art_model_mm(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchFasterRCNN:

    num_classes = model_kwargs.pop("num_classes", 4)

    backbone = MultimodalNaive(**model_kwargs)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        image_mean=[0.485, 0.456, 0.406, 0.0, 0.0, 0.0],
        image_std=[0.229, 0.224, 0.225, 1.0, 1.0, 1.0],
    )
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchFasterRCNN(
        model,
        clip_values=(0.0, 1.0),
        channels_first=False,
        num_classes=num_classes,
        **wrapper_kwargs,
    )
    return wrapped_model


def get_art_model_mm_robust(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchFasterRCNN:

    num_classes = model_kwargs.pop("num_classes", 4)

    backbone = MultimodalRobust(**model_kwargs)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        image_mean=[0.485, 0.456, 0.406, 0.0, 0.0, 0.0],
        image_std=[0.229, 0.224, 0.225, 1.0, 1.0, 1.0],
    )
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchFasterRCNN(
        model,
        clip_values=(0.0, 1.0),
        channels_first=False,
        num_classes=num_classes,
        **wrapper_kwargs,
    )
    return wrapped_model
