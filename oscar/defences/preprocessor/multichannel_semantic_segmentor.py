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


class MultichannelSemanticSegmentor(PreprocessorPyTorch):
    def __init__(
        self,
        *args,
        device_type = 'gpu',
        batch_dim = (0, 1),
        batch_size = (1, 1),
        **kwargs,
    ):
        torch_module = MultichannelSemanticSegmentorPyTorch(*args, **kwargs)

        super().__init__(torch_module, device_type=device_type, batch_dim=batch_dim, batch_size=batch_size)


class MultichannelSemanticSegmentorPyTorch(torch.nn.Module, ExTransform):
    """
    A defense which uses multichannel semantic segmentation. This multichannel semantic maps are represented by 80 channels: one channel per thing class in coco_2017_val dataset.
    """
    def __init__(self, detectron2, nb_channels=None):
        super().__init__()

        self.detectron2 = detectron2
        if isinstance(detectron2, dict):
            from armory.utils.config_loading import load
            self.detectron2 = load(detectron2)

        self.nb_channels = nb_channels
        if self.nb_channels is None:
            self.nb_channels = len(self.detectron2.metadata.thing_classes)

    def forward(self, x, **kwargs):
        assert len(x.shape) == 5 # NTHWC
        assert x.shape[0] == 1 # N=1
        assert x.shape[4] == 3 # C=RGB
        assert 0 <= x.min() <= x.max() <= 1

        nb_batch, nb_frames, height, width, _ = x.shape

        # Multichannel Semantic Maps
        #  shape: (batch, frames, height, width, channels)
        #  nb_channels = 80 (number of thing classes in coco 2017 dataset)
        #  default value (background) is 0
        # Keep on CPU because this is large!
        seg_maps = torch.zeros((nb_batch, nb_frames, height, width, self.nb_channels), device='cpu')

        predictions = self.detectron2.forward(x[0], **kwargs)

        assert len(predictions) == len(x[0])

        for i, instances in enumerate(predictions):
            if len(instances) == 0:
                continue

            for label in range(self.nb_channels):
                # Find all detections of label
                label_idx = (instances.pred_classes == label)

                # Skip labels that have no detections
                if not label_idx.any():
                    continue

                # Create binary segmentation mask for label
                label_mask = instances.pred_masks[label_idx]
                label_mask = label_mask.any(dim=0)
                # TODO: It may be worth separating background and foreground in to -1 and 1 and letting 0 represent no detections (rather than background).
                seg_maps[0, i, :, :, label] = label_mask.to(seg_maps.device)

        assert x.shape[:4] == seg_maps.shape[:4]
        assert seg_maps.shape[4] == self.nb_channels

        return seg_maps

    def estimate_forward(self, x):
        assert len(x.shape) == 5
        assert x.shape[4] == 3
        assert 0 <= x.min() <= x.max() <= 1

        segmaps = []
        for x_batch in x:
            segmaps_batch = []
            for x_image in x_batch:
                # Add batch dimension
                x_image = x_image.unsqueeze(0)

                predictions = self.detectron2.estimate_forward(x_image)

                # Remove batch dimension
                assert len(predictions) == 1
                instances = predictions[0]

                # For each class
                segmaps_image = []
                for label in range(self.nb_channels):
                    # Set to mean-channel image to preserve gradient but fill with 0's
                    label_mask = x_image[0].mean(dim=-1)
                    label_mask.data[:] = 0

                    label_idx = (instances.pred_classes == label)

                    if label_idx.any():
                        # Quantize pred_mask to 1-bit
                        label_mask = instances.pred_masks[label_idx]
                        label_mask = label_mask.sum(dim=0, keepdims=False)
                        label_mask = Quantize.apply(label_mask, 1)

                    segmaps_image.append(label_mask)

                segmap = torch.stack(segmaps_image, dim=-1)
                segmaps_batch.append(segmap)

            segmaps_batch = torch.stack(segmaps_batch)
            segmaps.append(segmaps_batch)

        x_mapped = torch.stack(segmaps)

        return x_mapped
