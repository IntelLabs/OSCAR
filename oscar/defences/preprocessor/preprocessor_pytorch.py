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
from typing import Optional, Tuple, TYPE_CHECKING
from oscar.utils import bmap

class PreprocessorPyTorch(_PreprocessorPyTorch):
    """
    A PyTorch-style preprocessor defense base class.
    We implement more generalized features:
    1. Device choice;
    2. __call__();
    3. estimate_gradient().
    """
    def __init__(self, torch_module, device_type: str = 'gpu', batch_dim=0, batch_size=1):
        super().__init__()

        self.module = torch_module
        if isinstance(self.module, dict):
            from armory.utils.config_loading import load
            self.module = load(torch_module)

        assert isinstance(self.module, torch.nn.Module)

        self.module_device = 'cpu'
        if device_type == 'gpu':
            if torch.cuda.is_available():
                logger.info("Putting torch module on GPU")
                self.module.cuda()
                self.module_device = 'cuda'
            else:
                logger.warning("Tried to put torch module on GPU, however no GPU detected.")

        self.batch_dim = batch_dim
        self.batch_size = batch_size


    @property
    def apply_predict(self):
        return True


    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        def _numpy_forward(x):
            with torch.no_grad():
                x = torch.tensor(x, device=self.module_device)
                x, _ = self.forward(x, y)
                x = x.cpu().numpy()

                return x

        x = bmap(_numpy_forward, x, batch_dim=self.batch_dim, batch_size=self.batch_size)

        # art.defenses.preprocessor.Preprocessor wants y
        return x, y


    def forward(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        # torch.nn.Module does not typically take y as input
        x = self.module(x)

        # art.defenses.preprocessor.Preprocessor wants y
        return x, y


    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        def _numpy_estimate_gradient(x, grad):
            # Move sample to torch and call estimate_forward
            x = torch.tensor(x, device=self.module_device, requires_grad=True)
            x_fwd, _ = self.estimate_forward(x)

            # Move grad sample to torch and use autograd
            grad = torch.tensor(grad, device=self.module_device)
            x_fwd.backward(grad)

            # Get gradient of input
            x_grad = x.grad.detach().cpu().numpy()

            return x_grad

        x_grad = bmap(_numpy_estimate_gradient, x, grad, batch_dim=self.batch_dim, batch_size=self.batch_size)

        # Gradient should be same shape as input
        assert x_grad.shape == x.shape

        return x_grad


    def estimate_forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        # torch.nn.Moduel does not typically take y as input
        x = self.module.estimate_forward(x)

        # art.defenses.preprocessor.Preprocessor wants y
        return x, y


    @property
    def apply_fit(self):
        return False


    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        return None
