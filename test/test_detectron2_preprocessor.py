#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import pytest
from pytest import approx
import torch
import numpy as np

logger = logging.getLogger(__name__)


def test_detectron2_preprocessor(detectron2_model, real_images_112x112_torch):
    x = real_images_112x112_torch[0]

    x_detected = detectron2_model(x)

    # FIXME: Ideally we would hardcode detections here but we don't know one a priori
    assert x_detected is not None
    assert len(x_detected) == len(x)


def test_detectron2_preprocessor_estimate_forward(detectron2_model, real_images_112x112_torch):
    x = real_images_112x112_torch[0]

    x_expected = detectron2_model(x)
    x_estimated = detectron2_model.estimate_forward(x)

    assert len(x_expected) == len(x_estimated)

    for expected_instances, estimated_instances in zip(x_expected, x_estimated):
        expected_instances = expected_instances.to('cpu')
        estimated_instances = estimated_instances.to('cpu')

        assert len(expected_instances) == len(estimated_instances)
        assert expected_instances.image_size == estimated_instances.image_size

        # All predicted classes must agree
        assert (expected_instances.pred_classes == estimated_instances.pred_classes).numpy().all()

        # Make sure all scores are within 0.2
        assert np.all(np.abs((expected_instances.scores - estimated_instances.scores).numpy()) < 0.15)

        # Make sure predicted boxes are near each other (within 3 pixels)
        assert np.all(np.abs((expected_instances.pred_boxes.tensor - estimated_instances.pred_boxes.tensor).numpy()) < 3)

        # Make sure predicted masks don't differ by more than 210 elements
        assert (expected_instances.pred_masks != estimated_instances.pred_masks).sum().numpy() < 210
