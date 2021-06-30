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
from oscar.defences.preprocessor.preprocessor_pytorch import PreprocessorPyTorch
from oscar.defences.preprocessor.gaussian_augmentation import GaussianAugmentationPyTorch
import detectron2
from detectron2.model_zoo import get_config_file
from detectron2.config import get_cfg
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Instances, ImageList
import detectron2.data.transforms as T
from oscar.utils.utils import create_model, create_inputs
from oscar.utils.layers import Quantize
from torchvision.transforms import Resize
from torch.nn import functional as F
from armory.data.utils import maybe_download_weights_from_s3
from pathlib import Path

# Monkey patch paste_masks_in_image with our modified implememtation that supports threshold=None
from oscar.utils.detectron2.layers.mask_ops import paste_masks_in_image
detectron2.modeling.postprocessing.paste_masks_in_image = paste_masks_in_image

class Detectron2PreprocessorPyTorch(torch.nn.Module):
    """
    A base class for defense preprocessors.
    """
    def __init__(self, config_path, weights_path, score_thresh=0.5, iou_thresh=None, resize=True):
        super().__init__()

        # Find paths to configuration and weights for Detectron2.
        if config_path.startswith('detectron2://'):
            config_path = config_path[len('detectron2://'):]
            config_path = get_config_file(config_path)
        elif config_path.startswith('oscar://'):
            config_path = config_path[len('oscar://'):]
            config_path = pkg_resources.resource_filename('oscar.model_zoo', config_path)
        elif config_path.startswith('armory://'):
            config_path = config_path[len('armory://'):]
            config_path = maybe_download_weights_from_s3(config_path)

        if weights_path.startswith('oscar://'):
            weights_path = weights_path[len('oscar://'):]
            weights_path = pkg_resources.resource_filename('oscar.model_zoo', weights_path)
        elif weights_path.startswith('armory://'):
            weights_path = weights_path[len('armory://'):]
            weights_path = maybe_download_weights_from_s3(weights_path)

        # Create Detectron2 Model CPU and rely upon torch.to to move to proper device since this is a proper nn.Module
        self.model, self.metadata = create_model(config_path, weights_path, device='cpu', score_thresh=score_thresh, iou_thresh=iou_thresh)
        logger.info(f"Detectron2 config: score_thresh={score_thresh}, iou_thresh={iou_thresh}.")

        # Get augmentation for resizing
        cfg = get_cfg()
        cfg.merge_from_file(config_path)

        self.aug = None
        if resize:
            # FIXME: We always resize to 800 in estimate_forward, so error loudly if the model expects something else
            assert cfg.INPUT.MIN_SIZE_TEST == 800

            self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )

    def forward(self, x):
        assert len(x.shape) == 4 # NHWC
        assert x.shape[3] == 3 # C = RGB
        assert 0. <= x.min() <= x.max() <= 1. # C in [0, 1]

        # NHWC -> NCHW
        x = x.permute((0, 3, 1, 2))

        # Run inference on examples
        self.model.eval()

        batched_inputs = create_inputs(x, transforms=self.aug, input_format=self.model.input_format)
        outputs = self.model(batched_inputs)
        batched_instances = [output['instances'] for output in outputs]

        return batched_instances

    def estimate_forward(self, x):
        assert len(x.shape) == 4 # NHWC
        assert x.shape[3] == 3 # C = RGB
        assert 0. <= x.min() <= x.max() <= 1. # C in [0, 1]

        # Make sure model is RCNN-style model
        if not isinstance(self.model, GeneralizedRCNN):
            raise NotImplementedError(f"There is no differentiable forward implementation for {self.model.__class__} currently")

        # Put into eval mode since we don't have groundtruth annotations
        self.model.eval()

        images = x
        _, orig_height, orig_width, _ = images.shape

        # Create inputs for Detectron2 model, we can't use create_inputs as above.
        images = 255*images                   # [0, 1] -> [0, 255]
        if self.model.input_format == 'BGR':
            images = images.flip(3)           # RGB -> BGR
        images = images.permute((0, 3, 1, 2)) # NHWC -> NCHW
        if self.aug is not None:
            images = Resize(800)(images)      # Resize to cfg.INPUT.MIN_SIZE_TEST
        images = (images - self.model.pixel_mean) / self.model.pixel_std

        # Mostly cribbed from https://github.com/facebookresearch/detectron2/blob/61457a0178939ec8f7ce130fcb733a5a5d47df9f/detectron2/structures/image_list.py#L70
        # Pad to self.backbone.size_divisibility
        _, _, height, width = images.shape
        size_divisibility = self.model.backbone.size_divisibility

        max_size = torch.tensor([height, width])

        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)) // stride * stride

        # left, right, top, bottom
        padding_size = [0, max_size[-1] - width, 0, max_size[-2] - height]
        images = F.pad(images, padding_size, value=0)

        # Mostly cribbed from https://github.com/facebookresearch/detectron2/blob/d135f1d9bddf68d11804e09722f2d54c0672d96b/detectron2/modeling/meta_arch/rcnn.py#L125
        image_list = ImageList(images, [(height, width) for _ in images])
        features = self.model.backbone(image_list.tensor)
        proposals, _ = self.model.proposal_generator(image_list, features, None)
        list_of_instances, _ = self.model.roi_heads(image_list, features, proposals, None)

        assert len(list_of_instances) == len(x)

        # In-place post-process instances and quantize prediction masks
        for i in range(len(list_of_instances)):
            instances = list_of_instances[i]

            #if len(instances) > 0:
                # Mostly cribbed from https://github.com/facebookresearch/detectron2/blob/d135f1d9bddf68d11804e09722f2d54c0672d96b/detectron2/modeling/meta_arch/rcnn.py#L233
                # However, we monkey-patch to support mask_threshold=None to give us a gradient
            instances = detector_postprocess(instances, orig_height, orig_width, mask_threshold=None)

            # Convert pred_masks score to 0-1 according to threshold
            pred_masks = instances.pred_masks
            # XXX: This implicitly thresholds at whatever threshold `round` uses. This is fine for now since
            #      mask_threshold=0.5 normally and is not externally configurable.
            pred_masks = Quantize.apply(pred_masks, 1)
            instances.pred_masks = pred_masks

            list_of_instances[i] = instances

        return list_of_instances


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


class GaussianDetectron2PreprocessorPyTorch(torch.nn.Module):
    """Add Gaussian noise to Detectron2 input. It behaves the same as Detectron2PreprocessorPyTorch when sigma is 0.
    """
    def __init__(self, sigma=0, clip_values=None, **detectron2_kwargs):
        super().__init__()

        self.noise_generator = GaussianAugmentationPyTorch(sigma=sigma, augmentation=False, clip_values=clip_values)
        self.detectron2 = Detectron2PreprocessorPyTorch(**detectron2_kwargs)

        logger.info(f"Add Gaussian noise sigma={sigma:.4f} clip_values={clip_values} to Detectron2's input.")

    def forward(self, x):
        x = self.noise_generator(x)
        x = self.detectron2(x)

        return x

    def estimate_forward(self, x):
        x = self.noise_generator.estimate_forward(x)
        x = self.detectron2.estimate_forward(x)

        return x
