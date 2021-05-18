#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#
import pkg_resources
import logging
logger = logging.getLogger(__name__)

import math
import torch
import numpy as np
from typing import Optional, Tuple, List
from oscar.defences.preprocessor.gaussian_augmentation import GaussianAugmentationPyTorch
import detectron2
from detectron2.model_zoo import get_config_file
from detectron2.config import get_cfg
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures.instances import Instances
import detectron2.data.transforms as T
from oscar.utils.utils import create_model, create_inputs
from oscar.utils.layers import Quantize
from torchvision.transforms import Resize, Pad
from armory.data.utils import maybe_download_weights_from_s3
from pathlib import Path

# Monkey patch paste_masks_in_image with our modified implememtation that supports threshold=None
from oscar.utils.detectron2.layers.mask_ops import paste_masks_in_image
detectron2.modeling.postprocessing.paste_masks_in_image = paste_masks_in_image

class Detectron2Preprocessor(torch.nn.Module):
    """
    A base class for defense preprocessors.
    """
    def __init__(self, config_path, weights_path, score_thresh=0.5, iou_thresh=None):
        super().__init__()

        # Find paths to configuration and weights for Detectron2.
        if config_path.startswith('detectron2://'):
            config_path = config_path[len('detectron2://'):]
            config_path = get_config_file(config_path)

        if config_path.startswith('oscar://'):
            config_path = config_path[len('oscar://'):]
            config_path = pkg_resources.resource_filename('oscar.model_zoo', config_path)

        if weights_path.startswith('oscar://'):
            weights_path = weights_path[len('oscar://'):]
            weights_path = pkg_resources.resource_filename('oscar.model_zoo', weights_path)

        if weights_path.startswith('armory://'):
            weights_path = weights_path[len('armory://'):]
            weights_path = maybe_download_weights_from_s3(weights_path)

        # Create Detectron2 Model CPU and rely upon torch.to to move to proper device since this is a proper nn.Module
        self.model, self.metadata = create_model(config_path, weights_path, device='cpu', score_thresh=score_thresh, iou_thresh=iou_thresh)
        logger.info(f"Detectron2 config: score_thresh={score_thresh}, iou_thresh={iou_thresh}.")

        # Get augmentation for resizing
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    def forward(self, x):
        assert(len(x.shape) == 4) # NHWC
        assert(x.shape[3] == 3) # C = RGB
        assert(0. <= x.min() <= x.max() <= 1.) # C in [0, 1]

        # NHWC -> NCHW
        x = x.permute((0, 3, 1, 2))

        # Run inference on examples
        self.model.eval()

        batched_instances = []
        for image in x:
            # Create inputs for Detectron2 model
            batched_inputs = create_inputs([image], transforms=self.aug, input_format='BGR')
            outputs = self.model(batched_inputs)
            assert len(outputs) == 1

            # Save instances from outputs
            instances = outputs[0]['instances']
            batched_instances.append(instances)

        return batched_instances

    def estimate_forward(self, x):
        assert(len(x.shape) == 4)
        assert(x.shape[3] == 3)

        # Make sure model is RCNN-style model
        if not isinstance(self.model, GeneralizedRCNN):
            raise NotImplementedError(f"There is no differentiable forward implementation for {self.model.__class__} currently")

        # Put into eval mode since we don't have groundtruth annotations
        self.model.eval()

        batch_instances = []
        for image in x:
            orig_height, orig_width, _ = image.shape

            # Create inputs for Detectron2 model, we can't use create_inputs as above.
            image = 255*image                   # [0, 1] -> [0, 255]
            image = image.flip(2)               # RGB -> BGR
            image = image.permute((2, 0, 1))    # NHWC -> NCHW
            image = Resize(800)(image)          # Resize to cfg.INPUT.MIN_SIZE_TEST

            # Pad to self.backbone.size_divisibility
            _, height, width = image.shape
            size_divisibility = self.model.backbone.size_divisibility

            pad_height = size_divisibility * math.ceil(height / size_divisibility) - height
            pad_width = size_divisibility * math.ceil(width / size_divisibility) - width

            pad_left = math.ceil(pad_width / 2)
            pad_top = math.ceil(pad_height / 2)
            pad_right = math.floor(pad_width / 2)
            pad_bottom = math.floor(pad_height / 2)

            image = Pad((pad_left, pad_top, pad_right, pad_bottom))(image)

            # Mostly cribbed from https://github.com/facebookresearch/detectron2/blob/d135f1d9bddf68d11804e09722f2d54c0672d96b/detectron2/modeling/meta_arch/rcnn.py#L125
            batched_inputs = [{'image': image}]
            images = self.model.preprocess_image(batched_inputs)
            features = self.model.backbone(images.tensor)
            proposals, _ = self.model.proposal_generator(images, features, None)
            results, _ = self.model.roi_heads(images, features, proposals, None)

            assert len(results) == 1

            instances = results[0]

            if len(instances) > 0:
                # Mostly cribbed from https://github.com/facebookresearch/detectron2/blob/d135f1d9bddf68d11804e09722f2d54c0672d96b/detectron2/modeling/meta_arch/rcnn.py#L233
                # However, we monkey-patch to support mask_threshold=None to give us a gradient
                instances = detector_postprocess(instances, orig_height, orig_width, mask_threshold=None)

            # Convert pred_masks score to 0-1 according to threshold
            pred_masks = instances.pred_masks
            # XXX: This implicitly thresholds at whatever threshold `round` uses. This is fine for now since
            #      mask_threshold=0.5 normally and is not externally configurable.
            pred_masks = Quantize.apply(pred_masks, 1)
            instances.pred_masks = pred_masks

            batch_instances.append(instances)

        return batch_instances


class CachedDetectron2Preprocessor(torch.nn.Module):
    def __init__(
        self,
        cache_dir,
    ):
        super().__init__()

        self.cache_dir = Path(cache_dir)

    def forward(self, x, parent, indices):
        assert(len(x.shape) == 4) # NHWC
        assert(x.shape[3] == 3) # C = RGB
        assert(0. <= x.min() <= x.max() <= 1.) # C in [0, 1]

        cache_path = self.cache_dir / parent.parts[-2] / (parent.parts[-1] + '.npz')
        dicts = np.load(cache_path, allow_pickle=True, mmap_mode='r')['instances'][indices]
        instances = [Instances(d['image_size'], **d['fields']) for d in dicts]

        return instances

    def estimate_forward(self, x):
        raise NotImplementedError


class GaussianDetectron2Preprocessor(torch.nn.Module):
    """Add Gaussian noise to Detectron2 input. It behaves the same as Detectron2Preprocessor when sigma is 0.
    """
    def __init__(self, sigma=0, clip_values=None, **detectron2_kwargs):
        super().__init__()

        self.noise_generator = GaussianAugmentationPyTorch(sigma=sigma, augmentation=False, clip_values=clip_values)
        self.detectron2 = Detectron2Preprocessor(**detectron2_kwargs)

        logger.info(f"Add Gaussian noise sigma={sigma:.4f} clip_values={clip_values} to Detectron2's input.")

    def forward(self, x):
        x = self.noise_generator(x)
        x = self.detectron2(x)

        return x

    def estimate_forward(self, x):
        x = self.noise_generator.estimate_forward(x)
        x = self.detectron2.estimate_forward(x)

        return x
