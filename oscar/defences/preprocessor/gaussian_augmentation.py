
#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING
from oscar.defences.preprocessor.preprocessor_pytorch import PreprocessorPyTorch

if TYPE_CHECKING:
    import torch
    from art.utils import CLIP_VALUES_TYPE


class GaussianAugmentation(PreprocessorPyTorch):
    def __init__(
        self,
        *args,
        device_type = 'gpu',
        batch_dim = (0, 1),
        batch_size = (1, 1),
        **kwargs,
    ):
        torch_module = GaussianAugmentationPyTorch(*args, **kwargs)

        super().__init__(torch_module, device_type=device_type, batch_dim=batch_dim, batch_size=batch_size)


# Some code is copied from art.defences.preprocessor.gaussian_augmentation.GaussianAugmentation
# I rewrote it in PyTorch to make it chainable.
class GaussianAugmentationPyTorch(torch.nn.Module):
    def __init__(
        self,
        sigma: float = 1.0,
        augmentation: bool = True,
        ratio: float = 1.0,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        apply_fit: bool = True,
        apply_predict: bool = False,
        **kwargs
    ):
        import torch  # lgtm [py/repeated-import]

        super().__init__(**kwargs)
        self._is_fitted = True
        if augmentation and not apply_fit and apply_predict:
            raise ValueError(
                "If `augmentation` is `True`, then `apply_fit` must be `True` and `apply_predict` must be `False`."
            )
        if augmentation and not (apply_fit or apply_predict):
            raise ValueError("If `augmentation` is `True`, then `apply_fit` and `apply_predict` can't be both `False`.")

        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.sigma = sigma
        self.augmentation = augmentation
        self.ratio = ratio
        self.clip_values = clip_values
        self._check_params()

    def forward(self, x):
        import torch  # lgtm [py/repeated-import]
        if self.augmentation:
            raise NotImplementedError("Gaussian Augmentation for training is not supported.")

        if self.sigma > 0:
            noise = torch.zeros_like(x)
            noise.data.normal_(0, std=self.sigma)
            x = x + noise

            if self.clip_values is not None:
                x = torch.clamp(x, self.clip_values[0], self.clip_values[1])

        return x

    def estimate_forward(self, x):
        # We have a differentiable forward().
        return self.forward(x)

    def _check_params(self) -> None:
        if self.augmentation and self.ratio <= 0:
            raise ValueError("The augmentation ratio must be positive.")

        if self.sigma < 0:
            raise ValueError("The Gaussian std must be non-negative.")

        if self.clip_values is not None:

            if len(self.clip_values) != 2:
                raise ValueError(
                    "`clip_values` should be a tuple of 2 floats or arrays containing the allowed data range."
                )
            if np.array(self.clip_values[0] >= self.clip_values[1]).any():
                raise ValueError("Invalid `clip_values`: min >= max.")


