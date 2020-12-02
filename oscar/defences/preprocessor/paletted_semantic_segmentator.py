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
from oscar.defences.preprocessor.detectron2_preprocessor import GaussianDetectron2Preprocessor


class PalettedSemanticSegmentor(PreprocessorPyTorch):
    """
    A defense which uses paletted semantic segmentation.
    """

    def __init__(self,
                 mask_color,
                 detectron2_config_path,
                 detectron2_weights_path,
                 detectron2_score_thresh=0.5,
                 detectron2_iou_thresh=None,
                 detectron2_device_type='gpu',
                 palette=None,
                 device_type='gpu',
                 gaussian_sigma=0,
                 gaussian_clip_values=None) -> None:
        super().__init__(device_type)

        self.detectron2 = GaussianDetectron2Preprocessor(sigma=gaussian_sigma,
                                                         clip_values=gaussian_clip_values,
                                                         config_path=detectron2_config_path,
                                                         weights_path=detectron2_weights_path,
                                                         score_thresh=detectron2_score_thresh,
                                                         iou_thresh=detectron2_iou_thresh,
                                                         device_type=detectron2_device_type)

        # Get palette from detectron2 metadata
        self.palette = palette
        if self.palette is None:
            self.palette = torch.tensor(self.detectron2.detectron2.metadata.thing_colors, dtype=torch.float32, device=self._device) / 255

        self.mask_color = torch.tensor(mask_color, dtype=torch.float32, device=self._device)

    def forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        assert len(x.shape) == 5
        assert x.shape[4] == 3
        assert 0 <= x.min() <= x.max() <= 1

        nb_batch, nb_frames, height, width, nb_channels = x.shape

        # Paletted semantic maps default background is to mask_color
        nb_channels = self.palette.shape[1]
        background = torch.ones((nb_batch, nb_frames, height, width, nb_channels), dtype=torch.float32, device=self._device)
        seg_maps = background * self.mask_color

        for i, x_batch in enumerate(x):
            predictions, _ = self.detectron2.forward(x_batch)

            for j, instances in enumerate(predictions):
                if len(instances) == 0:
                    continue

                for k in torch.argsort(instances.scores):
                    pred_class = instances.pred_classes[k]
                    pred_class_color = self.palette[pred_class]

                    # Add channels dim so broadcasting works
                    pred_mask = instances.pred_masks[k].unsqueeze(-1)

                    # Replace segmentation map with current predicted class color according to predicted mask
                    seg_maps[i, j] = seg_maps[i, j] * ~pred_mask + pred_class_color * pred_mask

        assert x.shape == seg_maps.shape
        assert x.dtype == seg_maps.dtype

        return seg_maps, y


    def estimate_forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        assert len(x.shape) == 5
        assert x.shape[4] >= 3

        # Only pass 3-channels
        return x[:, :, :, :, :3]
