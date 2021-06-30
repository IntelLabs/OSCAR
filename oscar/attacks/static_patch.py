#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from art.attacks.attack import EvasionAttack
from art.utils import check_and_transform_label_format
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin

import numpy as np
from typing import Union


class StaticPatch(EvasionAttack):
    """
    Apply Static Patch attack to video input with shape (NTHWC),
    where the patch is same through all frames within a video.
    """
    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "max_iter",
        "verbose",
        "patch_ratio",
        "xmin",
        "ymin",
        "patch_width",
        "patch_height"
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)

    def __init__(
        self,
        estimator,
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        max_iter: int = 10,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 1,
        verbose: bool = True,
        xmin: int = 0,
        ymin: int = 0,
        patch_ratio: float = 0,
        patch_width: int = 0,
        patch_height: int = 0,
    ):
        super().__init__(estimator=estimator)

        self.norm = norm
        self.eps = eps  # Keep for science scenario's security curve
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.verbose = verbose
        self.xmin = xmin
        self.ymin = ymin
        self.patch_ratio = patch_ratio
        self.patch_width = patch_width
        self.patch_height = patch_height
        self._check_params()

        if self.norm not in [np.inf, "inf"]:
            raise ValueError(
                "Currently only Linf norm is supported"
            )

        if (self.patch_width <= 0 or self.patch_height <= 0) and self.patch_ratio <= 0:
            raise ValueError(
                "kwargs did not define 'patch_height' and 'patch_width', or it did not define 'patch_ratio'"
            )

    def generate(self, x, y=None, **generate_kwargs):
        # input x should be with shape (NTHWC)
        assert x.ndim == 5, "This attack is designed for videos with shape (NTHWC)"
        assert x.shape[-1] == 3, "Input should have 3 channels in the last dimension"

        width, height = x.shape[-2], x.shape[-3]
        if self.xmin >= width:
            raise ValueError("'xmin' should be smaller than input width")
        if self.ymin >= height:
            raise ValueError("'ymin' should be smaller than input height")

        patch_width = self.patch_width
        patch_height = self.patch_height

        if self.patch_ratio > 0:
            # Make patch shape a square
            patch_width = int(min(width, height) * self.patch_ratio ** 0.5)
            patch_height = patch_width

        xmax = min(self.xmin + patch_width, width)
        ymax = min(self.ymin + patch_height, height)

        if y is None:
            if self.targeted:
                raise ValueError("Targeted Static Patch attack requires labels 'y'")
            y = self.estimator.predict(x).argmax()
            y = np.expand_dims(y, 0)
        targets = check_and_transform_label_format(y, self.estimator.nb_classes)

        mask = np.zeros(shape=x.shape[1:], dtype=bool)
        mask[:, self.ymin:ymax, self.xmin:xmax, :] = 1
        init_patch = np.mean(self.estimator.clip_values, dtype=np.float32)

        # Fix me: add batching
        assert self.batch_size == 1
        for random_init in range(max(1, self.num_random_init)):
            # Set the masked area in x
            if random_init > 0:
                init_patch = np.float32(np.random.uniform(0.4, 0.6))
            x_masked = x * ~mask + init_patch * mask

            # Apply a constant patch to all frames
            for _ in range(self.max_iter):
                grad = self.estimator.loss_gradient(x=x_masked, y=targets) * (1 - 2 * int(self.targeted))
                grad = np.where(mask == 0.0, 0.0, grad)
                assert grad.shape == x_masked.shape

                # Average masked gradients through all frames with shape (NTHWC)
                ave_grad = np.mean(grad, axis=1, keepdims=True)

                perturbs = np.sign(ave_grad) * self.eps_step
                x_masked = x_masked + perturbs
                clip_min, clip_max = self.estimator.clip_values
                x_masked = np.clip(x_masked, clip_min, clip_max)

            y_pred = self.estimator.predict(x_masked).argmax()
            if y_pred != y:
                break

        return x_masked

