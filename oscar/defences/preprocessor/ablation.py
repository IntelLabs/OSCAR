#
# Copyright (C) 2020 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np
from typing import Optional, Tuple, List
from oscar.defences.preprocessor.preprocessor_pytorch import PreprocessorPyTorch
from oscar.defences.preprocessor.detectron2_preprocessor import Detectron2Preprocessor, GaussianDetectron2Preprocessor


class BackgroundAblator(PreprocessorPyTorch):
    """
    A defense which ablates background pixels as the mean pixel value.
    """

    def __init__(self,
                 mask_color,
                 invert_mask=False,
                 enforce_binary_masks=True,
                 detectron2_config_path=None,
                 detectron2_weights_path=None,
                 detectron2_score_thresh=None,
                 detectron2_iou_thresh=None,
                 detectron2_device_type='gpu',
                 device_type='gpu') -> None:
        super().__init__(device_type)

        if detectron2_config_path is not None and detectron2_weights_path is not None:
            self.detectron2 = Detectron2Preprocessor(detectron2_config_path,
                                                     detectron2_weights_path,
                                                     score_thresh=detectron2_score_thresh,
                                                     iou_thresh=detectron2_iou_thresh,
                                                     device_type=detectron2_device_type)

        self.mask_color = torch.tensor(mask_color, dtype=torch.float32, device=self._device)
        self.invert_mask = invert_mask
        self.enforce_binary_masks = enforce_binary_masks

    def forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        assert len(x.shape) == 5
        assert x.shape[4] in [3, 4]
        assert 0 <= x.min() <= x.max() <= 1

        # Get masks and apply them to the input images
        masks = self._get_masks(x)
        x_out = self._mask_images(x, masks)

        assert x_out.shape[:4] == x.shape[:4]
        assert x_out.shape[4] == 3
        assert x_out.dtype == x.dtype

        return x_out, y

    def _get_masks(self, x: "torch.Tensor") -> "torch.Tensor":
        assert(len(x.shape) == 5)

        masks = x[:, :, :, :, 3:]
        if masks.shape[4] == 0:
            masks = self._create_masks(x[:, :, :, :, :3])

        # FIXME: There is a better way of rounding by using fake quantization
        if self.enforce_binary_masks:
            masks = masks.round()

        if self.invert_mask:
            masks = 1 - masks

        return masks

    def _create_masks(self, x: "torch.Tensor") -> "torch.Tensor":
        assert len(x.shape) == 5
        assert x.shape[0] == 1
        assert x.shape[4] == 3

        nb_batch, nb_frames, height, width, nb_channels = x.shape

        # Masks default to no masking (1)
        nb_channels = 1
        masks = torch.ones((nb_batch, nb_frames, height, width, 1), dtype=torch.float32, device=self._device)

        for i, x_batch in enumerate(x):
            predictions, _ = self.detectron2.forward(x_batch)

            for j, instances in enumerate(predictions):
                if len(instances) == 0:
                    continue

                mask = instances.pred_masks.any(dim=0, keepdims=True) # Collapse all prediction masks into single mask
                mask = mask.permute(1, 2, 0) # Channels First -> Channels Last

                masks[i, j] = mask

        return masks

    def _mask_images(self, x: "torch.Tensor", masks: "torch.Tensor") -> "torch.Tensor":
        assert len(x.shape) == 5
        assert x.shape[4] >= 3
        assert masks.shape[4] == 1

        masks = masks.repeat(1, 1, 1, 1, 3)

        x_masked = x[:, :, :, :, :3] * masks + self.mask_color * (1 - masks)

        return x_masked

    def estimate_forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        assert len(x.shape) == 5
        assert x.shape[4] >= 3

        # Only pass 3-channels
        return x[:, :, :, :, :3]


class GaussianBackgroundAblator(PreprocessorPyTorch):
    """
    Add gaussian noise to detectron2 input but does not noise final outputs.
    """
    def __init__(self,
                 mask_color,
                 invert_mask=False,
                 enforce_binary_masks=True,
                 detectron2_config_path=None,
                 detectron2_weights_path=None,
                 detectron2_score_thresh=None,
                 detectron2_iou_thresh=None,
                 gaussian_sigma=0,
                 gaussian_clip_values=None) -> None:
        super().__init__()

        # Avoid creating a non-Gaussian Detectron2 model.
        self.ablator = BackgroundAblator(mask_color,
                                         invert_mask=invert_mask,
                                         enforce_binary_masks=enforce_binary_masks)

        if detectron2_config_path is not None and detectron2_weights_path is not None:
            self.detectron2 = GaussianDetectron2Preprocessor(sigma=gaussian_sigma,
                                                             clip_values=gaussian_clip_values,
                                                             config_path=detectron2_config_path,
                                                             weights_path=detectron2_weights_path,
                                                             score_thresh=detectron2_score_thresh,
                                                             iou_thresh=detectron2_iou_thresh)
            self.ablator.detectron2 = self.detectron2

    def forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        return self.ablator.forward(x, y)

    def estimate_forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        return self.ablator.estimate_forward(x, y)
