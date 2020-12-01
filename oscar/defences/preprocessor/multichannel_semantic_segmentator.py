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


class MultichannelSemanticSegmentor(PreprocessorPyTorch):
    """
    A defense which uses multichannel semantic segmentation. This multichannel semantic maps are represented by 80 channels: one channel per thing class in coco_2017_val dataset.
    """
    def __init__(self,
                 detectron2_config_path,
                 detectron2_weights_path,
                 detectron2_score_thresh=0.5,
                 detectron2_iou_thresh=None,
                 detectron2_device_type='gpu',
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

    def forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        assert len(x.shape) == 5
        assert x.shape[4] == 3

        nb_batch, nb_frames, height, width, _ = x.shape

        # Multichannel Semantic Maps
        #  shape: (batch, frames, height, width, channels)
        #  nb_channels = 80 (number of thing classes in coco 2017 dataset)
        #  default value (background) is 0
        nb_channels = len(self.detectron2.detectron2.metadata.thing_classes)
        seg_maps = torch.zeros((nb_batch, nb_frames, height, width, nb_channels), dtype=torch.float32, device=self._device)

        for i, x_batch in enumerate(x):
            predictions, _ = self.detectron2.forward(x_batch)

            for j, instances in enumerate(predictions):
                if len(instances) == 0:
                    continue

                for label in range(nb_channels):
                    # Find all detections of label
                    label_idx = (instances.pred_classes == label)

                    # Skip labels that have no detections
                    if not label_idx.any():
                        continue

                    # Create binary segmentation mask for label
                    label_mask = instances.pred_masks[label_idx]
                    label_mask = label_mask.any(axis=0)
                    # TODO: It may be worth separating background and foreground in to -1 and 1 and letting 0 represent no detections (rather than background).
                    seg_maps[i, j, :, :, label] = label_mask

        assert x.shape[:4] == seg_maps.shape[:4]
        assert seg_maps.shape[4] == nb_channels

        return seg_maps, y

    def estimate_forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        # This is a dumb gradient
        nb_channels = x.shape[4]
        x_out = torch.ones((*x.shape[:4], 80 - nb_channels))
        x_out = torch.cat([x, x_out.to(self._device)], dim=4)
        return x_out
