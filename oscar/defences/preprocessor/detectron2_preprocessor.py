#
# Copyright (C) 2020 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#
import pkg_resources
import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np
from abc import abstractmethod
from typing import Optional, Tuple, List
from oscar.defences.preprocessor.preprocessor_pytorch import PreprocessorPyTorch
from oscar.defences.preprocessor.gaussian_augmentation_pytorch import GaussianAugmentationPyTorch
from detectron2.model_zoo import get_config_file
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from oscar.utils.utils import create_model, create_inputs
from armory.data.utils import maybe_download_weights_from_s3

class Detectron2Preprocessor(PreprocessorPyTorch):
    """
    A base class for defense preprocessors.
    """

    def __init__(self, config_path, weights_path, score_thresh=0.5, iou_thresh=None, device_type='gpu') -> None:
        super().__init__(device_type=device_type)

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

        # Create Detectron2 Model
        self.model, self.metadata = create_model(config_path, weights_path, device=self._device, score_thresh=score_thresh, iou_thresh=iou_thresh)
        logger.info(f"Detectron2 config: score_thresh={score_thresh}, iou_thresh={iou_thresh}.")

        # Get augmentation for resizing
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    def forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> List[List[dict]]:
        assert(len(x.shape) == 4)
        assert(x.shape[3] == 3)
        assert(0. <= x.min() <= x.max() <= 1.)

        # Save old device
        device = x.device
        x = x.to(self._device)

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

            # Extract instance from outputs and put on x's device
            instances = outputs[0]['instances']
            instances = instances.to(device)
            batched_instances.append(instances)

        return batched_instances, y

    def estimate_forward(self, x, **kwargs):
        return x


class GaussianDetectron2Preprocessor(PreprocessorPyTorch):
    """Add Gaussian noise to Detectron2 input. It behaves the same as Detectron2Preprocessor when sigma is 0.
    """
    def __init__(self, sigma=0, clip_values=None, **kwargs):
        super().__init__()
        self.noise_generator = GaussianAugmentationPyTorch(sigma=sigma, augmentation=False, clip_values=clip_values)
        self.detectron2 = Detectron2Preprocessor(**kwargs)
        logger.info(f"Add Gaussian noise sigma={sigma:.4f} clip_values={clip_values} to Detectron2's input.")

    def forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> List[List[dict]]:
        x, _ = self.noise_generator.forward(x, y)
        return self.detectron2.forward(x, y)

    def estimate_forward(self, x, **kwargs):
        x = self.noise_generator.estimate_forward(x)
        return self.detectron2.estimate_forward(x, **kwargs)
