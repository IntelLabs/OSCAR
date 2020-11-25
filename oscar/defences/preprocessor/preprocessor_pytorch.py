#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch as _PreprocessorPyTorch
from abc import abstractmethod
from typing import Optional, Tuple, TYPE_CHECKING


class PreprocessorPyTorch(_PreprocessorPyTorch):
    """ A PyTorch-style preprocessor defense base class.
    We implement more generalized features:
    1. Device choice;
    2. __call__();
    3. estimate_gradient().
    """

    def __init__(self, device_type: str = "gpu"):
        super().__init__()

        # Set device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    @abstractmethod
    def estimate_forward(self, x, **kwargs):
        pass

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply local spatial smoothing to sample `x`.
        :param x: Sample to smooth with shape `(batch_size, width, height, depth)`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Smoothed sample.
        """
        import torch  # lgtm [py/repeated-import]

        x = torch.tensor(x.copy(), device=self._device)
        if y is not None:
            y = torch.tensor(y, device=self._device)

        with torch.no_grad():
            x, y = self.forward(x, y)

        result = x.cpu().numpy()
        if y is not None:
            y = y.cpu().numpy()
        return result, y

    # Backward compatibility.
    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        import torch  # lgtm [py/repeated-import]

        x = torch.tensor(x.copy(), device=self._device, requires_grad=True)
        grad = torch.tensor(grad, device=self._device)

        x_prime = self.estimate_forward(x)
        x_prime.backward(grad)
        x_grad = x.grad.detach().cpu().numpy()
        if x_grad.shape != x.shape:
            raise ValueError("The input shape is {} while the gradient shape is {}".format(x.shape, x_grad.shape))
        return x_grad

    @property
    def apply_fit(self):
        return True

    @property
    def apply_predict(self):
        return True

    def fit(self, x, y=None, **kwargs):
        return None
