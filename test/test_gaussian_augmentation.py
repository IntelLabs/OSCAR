#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from oscar.defences.preprocessor.gaussian_augmentation import GaussianAugmentation

import pytest
from pytest import approx
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)
SIGMA = 0.25


@pytest.fixture
def gaussian_augmentor():
    return GaussianAugmentation(sigma=SIGMA, augmentation=False)


@pytest.fixture
def gaussian_augmentor_clip01():
    return GaussianAugmentation(sigma=SIGMA, augmentation=False, clip_values=[0., 1.])


def test_gaussian_augmentation_forward(gaussian_augmentor, real_images_112x112):
    x = real_images_112x112
    x_noised, _ = gaussian_augmentor(x)
    delta = x_noised - x
    assert delta.mean() == approx(0, abs=1e-2)
    assert delta.std() == approx(SIGMA, abs=1e-2)


def test_gaussian_augmentation_clip01_forward(gaussian_augmentor_clip01, real_images_112x112):
    x = real_images_112x112
    x_noised, _ = gaussian_augmentor_clip01(x)
    # We don't test mean or std because they may not hold after clipping.
    assert np.all(x_noised <= 1)
    assert np.all(0 <= x_noised)


def test_gaussian_augmentation_estimate_gradient(gaussian_augmentor_clip01, real_images_112x112):
    x = real_images_112x112

    x_noised, _ = gaussian_augmentor_clip01(x)
    grad = np.ones_like(x_noised)

    grad = gaussian_augmentor_clip01.estimate_gradient(x, grad)

    assert grad is not None
