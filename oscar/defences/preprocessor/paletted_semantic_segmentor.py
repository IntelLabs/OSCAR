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


class PalettedSemanticSegmentor(PreprocessorPyTorch):
    def __init__(
        self,
        *args,
        device_type = 'gpu',
        batch_dim = (0, 1),
        batch_size = (1, 1),
        **kwargs,
    ):
        torch_module = PalettedSemanticSegmentorPyTorch(*args, **kwargs)

        super().__init__(torch_module, device_type=device_type, batch_dim=batch_dim, batch_size=batch_size)


class PalettedSemanticSegmentorPyTorch(torch.nn.Module, ExTransform):
    """
    A defense which uses paletted semantic segmentation.
    """
    def __init__(
        self,
        mask_color,
        detectron2,
        palette=None
    ) -> None:
        super().__init__()

        self.detectron2 = detectron2
        if isinstance(detectron2, dict):
            from armory.utils.config_loading import load
            self.detectron2 = load(detectron2)

        self.register_buffer('mask_color', torch.tensor(mask_color).float())

        if self.mask_color.min() < 0 or self.mask_color.max() > 1:
            logger.warn("mask_color not in [0, 1] range! %s", self.mask_color)

        if palette is None:
            # Get palette from detectron2 metadata
            palette = self.detectron2.metadata.thing_colors
        self.register_buffer('palette', torch.tensor(palette).float()/255)

    def forward(self, x, **kwargs):
        assert len(x.shape) == 5 # NTHWC
        assert x.shape[0] == 1 # N=1
        assert x.shape[4] == 3 # C=RGB
        assert 0 <= x.min() <= x.max() <= 1

        predictions = self.detectron2.forward(x[0], **kwargs)

        assert len(predictions) == len(x[0])

        seg_maps = self.create_seg_maps(x[0], predictions)[None]

        assert x.shape == seg_maps.shape
        assert x.dtype == seg_maps.dtype

        return seg_maps

    def create_seg_maps(self, x, predictions):
        nb_frames, height, width, nb_channels = x.shape

        # Paletted semantic maps default background is to mask_color
        nb_channels = self.palette.shape[-1]
        # XXX: Is there a way to avoid this device=x.device?
        background = torch.ones((nb_frames, height, width, nb_channels), device=x.device)
        # NOTE: Should we warn when nb_channels != mask_color.shape[0]?
        seg_maps = background * self.mask_color

        for i, instances in enumerate(predictions):
            if len(instances) == 0:
                continue

            for j in torch.argsort(instances.scores):
                pred_class = instances.pred_classes[j]
                pred_class_color = self.palette[pred_class]

                # Add channels dim so broadcasting works
                pred_mask = instances.pred_masks[j].unsqueeze(-1)

                # Replace segmentation map with current predicted class color according to predicted mask
                seg_maps[i] = seg_maps[i] * ~pred_mask + pred_class_color * pred_mask

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

                # Set to image to preserve gradient but fill with mask_color
                segmap = x_image[0]
                segmap.data[:] = self.mask_color

                # For each detection in order from least confident to most confident
                for i in torch.argsort(instances.scores):
                    # Background is 0
                    pred_class = instances.pred_classes[i]
                    pred_class_color = self.palette[pred_class]

                    # Quantize pred_mask to 1-bit
                    pred_mask = instances.pred_masks[i].unsqueeze(-1)
                    pred_mask = Quantize.apply(pred_mask, 1)

                    # Set all 1-values to the predicted class color
                    segmap = segmap * (1 - pred_mask) + pred_mask * pred_class_color

                segmaps_batch.append(segmap)

            segmaps_batch = torch.stack(segmaps_batch)
            segmaps.append(segmaps_batch)

        x_mapped = torch.stack(segmaps)

        return x_mapped
