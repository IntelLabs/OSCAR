#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from oscar.defences.preprocessor.paletted_semantic_segmentor import PalettedSemanticSegmentor

import logging
import pytest
from pytest import approx
import torch
import numpy as np

logger = logging.getLogger(__name__)


@pytest.fixture
def paletted_semantic_segmentor(detectron2_model, mask_color):
    return PalettedSemanticSegmentor(mask_color, detectron2=detectron2_model)


def test_segmentor_estimate_forward(paletted_semantic_segmentor, real_images_112x112):
    x = real_images_112x112

    x_expected, _ = paletted_semantic_segmentor(x, y=None)

    x = torch.tensor(real_images_112x112, device=paletted_semantic_segmentor.module_device)
    x_mapped, _ = paletted_semantic_segmentor.estimate_forward(x, y=None)
    x_mapped = x_mapped.cpu().numpy()

    diff = np.sum(np.any(np.abs(x_expected - x_mapped) > 1/255, axis=-1))

    assert diff < 150


def test_segmentor_gradient(paletted_semantic_segmentor, real_images_112x112):
    x = torch.tensor(real_images_112x112, requires_grad=True, device=paletted_semantic_segmentor.module_device)
    x_segmented, _ = paletted_semantic_segmentor.forward(x)

    assert x_segmented.grad_fn == None


def test_segmentor_estimate_gradient(paletted_semantic_segmentor, real_images_112x112):
    x = real_images_112x112

    # Create fake gradient
    x_segmented, _ = paletted_semantic_segmentor(x)
    grad = np.ones_like(x_segmented)

    # Get x's gradient using estimate_gradient
    grad_mapped = paletted_semantic_segmentor.estimate_gradient(x, grad)

    # There should be gradients everywhere on the ablated image
    assert np.sum(grad_mapped == 0) == 0
