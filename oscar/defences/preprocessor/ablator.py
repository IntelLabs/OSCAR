#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging

import torch

from oscar.utils.layers import Quantize
from oscar.defences.preprocessor.preprocessor_pytorch import PreprocessorPyTorch
from oscar.data.transforms import ExTransform

logger = logging.getLogger(__name__)


class Ablator(PreprocessorPyTorch):
    def __init__(
        self,
        *args,
        device_type = 'gpu',
        batch_dim = (0, 1),
        batch_size = (1, 1),
        **kwargs,
    ):
        torch_module = AblatorPyTorch(*args, **kwargs)

        super().__init__(torch_module, device_type=device_type, batch_dim=batch_dim, batch_size=batch_size)

    @property
    def invert_mask(self):
        return self.module.invert_mask

    @invert_mask.setter
    def invert_mask(self, value):
        self.module.invert_mask = value

    @property
    def mask_color(self):
        return self.module.mask_color


class AblatorPyTorch(torch.nn.Module, ExTransform):
    """
    A defense which ablates background pixels as the mean pixel value.
    """
    def __init__(
        self,
        mask_color,
        detectron2,
        invert_mask=False,
        enforce_binary_masks=True,
    ):
        super().__init__()

        self.detectron2 = detectron2
        if isinstance(detectron2, dict):
            from armory.utils.config_loading import load
            self.detectron2 = load(detectron2)

        self.register_buffer('mask_color', torch.tensor(mask_color).float())

        if self.mask_color.min() < 0 or self.mask_color.max() > 1:
            logger.warn("mask_color not in [0, 1] range! %s", self.mask_color)

        self.invert_mask = invert_mask
        self.enforce_binary_masks = enforce_binary_masks

    def forward(self, x, **kwargs):
        assert len(x.shape) == 5 # NTHWC
        assert x.shape[4] in [3, 5] # C=RGB | RGBXY
        assert 0 <= x.min() <= x.max() <= 1

        x_rgb = x[:, :, :, :, 0:3]

        # Get masks and apply them to the input images
        masks = self._get_masks(x_rgb, **kwargs)
        x_out = self._mask_images(x, masks)

        assert x_out.shape[:4] == x.shape[:4]
        assert x_out.dtype == x.dtype

        return x_out

    def _get_masks(self, x, **kwargs):
        assert len(x.shape) == 5 # NTHWC
        assert x.shape[4] == 3 # C=RGB

        masks = self._create_masks(x[:, :, :, :, :3], **kwargs)

        if self.enforce_binary_masks:
            masks = masks.round()

        if self.invert_mask:
            masks = 1 - masks

        return masks

    def _create_masks(self, x, **kwargs):
        assert len(x.shape) == 5 # NTHWC
        assert x.shape[0] == 1 # N=1
        assert x.shape[4] == 3 # C=RGB

        nb_batch, nb_frames, height, width, nb_channels = x.shape

        # Masks default to no masking (1)
        nb_channels = 1
        masks = torch.ones((nb_batch, nb_frames, height, width, nb_channels), dtype=torch.float32, device=x.device)

        predictions = self.detectron2.forward(x[0], **kwargs)

        assert len(predictions) == len(x[0])

        for i, instances in enumerate(predictions):
            if len(instances) == 0:
                continue

            mask = instances.pred_masks.any(dim=0, keepdims=True) # Collapse all prediction masks into single mask
            mask = mask.permute(1, 2, 0) # Channels First -> Channels Last

            masks[0, i, :, :, :] = mask

        return masks

    def _mask_images(self, x, masks):
        assert len(x.shape) == 5
        assert x.shape[4] >= 3
        assert masks.shape[4] == 1

        # FIXME: Can we avoid this repeat?
        masks = masks.repeat(1, 1, 1, 1, x.shape[4])

        # XXX: Should we warn when x.shape[4] != mask_color.shape[0]?
        x_masked = x * masks + self.mask_color[:x.shape[4]] * (1 - masks)

        return x_masked

    def estimate_forward(self, x):
        assert len(x.shape) == 5
        assert x.shape[4] == 3

        masks = []
        for x_batch in x:
            masks_batch = []
            for x_image in x_batch:
                # Add batch dimension
                x_image = x_image.unsqueeze(0)

                predictions = self.detectron2.estimate_forward(x_image)

                # Remove batch dimension
                assert len(predictions) == 1
                instances = predictions[0]

                pred_masks = instances.pred_masks
                if pred_masks.shape[0] == 0:
                    #  Set to image to preserve gradient but fill with 1
                    pred_masks = x_image.mean(dim=-1)
                    pred_masks.data[:] = 1

                mask = pred_masks.sum(dim=0, keepdims=True)
                mask = Quantize.apply(mask, 1)
                mask = mask.permute(1, 2, 0) # Channels First -> Channels Last

                if self.invert_mask:
                    mask = 1 - mask

                masks_batch.append(mask)

            masks_batch = torch.stack(masks_batch)
            masks.append(masks_batch)

        masks = torch.stack(masks)
        x_ablated = self._mask_images(x, masks)

        return x_ablated
