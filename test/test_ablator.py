#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from oscar.defences.preprocessor.ablator import Ablator

import logging
import numpy as np
import pytest
from pytest import approx
from functools import reduce
import operator
import torch

logger = logging.getLogger(__name__)


@pytest.fixture(params=[True, False])
def ablator(request, detectron2_model, mask_color):
    return Ablator(mask_color, detectron2=detectron2_model, invert_mask=request.param)


def test_ablator(ablator, real_images_112x112):
    x = real_images_112x112
    x_ablated, _ = ablator(x)

    # Make sure some pixels are ablated by forward.
    mask_color = ablator.mask_color.cpu().numpy()
    nb_ablated = np.sum(np.all(np.abs(x_ablated - mask_color) < 1e-8, axis=-1))

    assert nb_ablated > 0


def test_ablator_inversion(ablator, real_images_112x112):
    x = real_images_112x112

    x_bg_ablated, _ = ablator(x, y=None)

    ablator.invert_mask = not ablator.invert_mask
    x_fg_ablated, _ = ablator(x, y=None)

    assert x_bg_ablated.shape == x_fg_ablated.shape

    nb_pixels = reduce(operator.mul, x_bg_ablated.shape[:-1])

    # Count number of mask-related pixels
    mask_color = ablator.mask_color.cpu().numpy()
    nb_bg_pixels = np.sum(np.all(np.abs(x_bg_ablated - mask_color) < 1e-8, axis=-1))
    nb_fg_pixels = np.sum(np.all(np.abs(x_fg_ablated - mask_color) < 1e-8, axis=-1))

    assert nb_bg_pixels + nb_fg_pixels == nb_pixels
    assert 0.1 * nb_pixels < nb_bg_pixels < 0.9 * nb_pixels


def test_ablator_estimate_forward(ablator, real_images_112x112):
    x = real_images_112x112
    x_ablated, _ = ablator(x)

    x_ablated_estimate, _ = ablator.estimate_forward(x, y=None)

    # Make sure the number of ablated pixels from estimate_forward is around (+-50) the same amount
    # of pixels from forward.
    mask_color = ablator.mask_color.cpu().numpy()
    nb_ablated = np.sum(np.all(np.abs(x_ablated - mask_color) < 1e-8, axis=-1))
    nb_ablated_estimate = np.sum(np.all(np.abs(x_ablated_estimate - mask_color) < 1e-8, axis=-1))

    assert nb_ablated_estimate > 0
    assert nb_ablated - 50 < nb_ablated_estimate < nb_ablated + 50


def test_ablator_estimate_gradient(ablator, real_images_112x112):
    x = real_images_112x112

    # Create fake gradient
    x_ablated, _ = ablator(x)
    grad = np.ones_like(x_ablated)

    # Get gradient using estimate_gradient
    grad_ablated = ablator.estimate_gradient(x, grad)

    # There should be gradients everywhere on the ablated image
    assert np.sum(grad_ablated == 0) == 0
