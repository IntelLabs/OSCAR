#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import pkg_resources
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING
from copy import deepcopy

import torch
import numpy as np

from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.pytorch import PyTorchEstimator

from oscar.utils import create_inputs, create_model

from detectron2.model_zoo import get_config_file
from detectron2.utils.events import EventStorage
from detectron2.structures.boxes import BoxMode

from armory.data.utils import maybe_download_weights_from_s3

logger = logging.getLogger(__name__)

class Detectron2Estimator(ObjectDetectorMixin, PyTorchEstimator):
    estimator_params = PyTorchEstimator.estimator_params + ["attack_losses"]

    def __init__(
        self,
        config_file: str,
        weights_file: str,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: Optional[bool] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: Tuple[str, ...] = (
            "loss_cls",
            "loss_box_reg",
            "loss_rpn_cls",
            "loss_rpn_loc",
        ),
        device_type: str = "gpu",
    ):
        # Set device
        self._device: torch.device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        # Create model
        model, metadata = create_model(config_file,
                                       weights_file,
                                       device=self._device,
                                       score_thresh=0.5,
                                       config={'MODEL.KEYPOINT_ON': False,
                                               'MODEL.MASK_ON': False})
        self._metadata = metadata

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        if self.clip_values is not None:
            if self.clip_values[0] != 0:
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, max_value).")
            if self.clip_values[1] <= 0:
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, max_value).")

        if preprocessing is not None:
            raise ValueError("This estimator does not support `preprocessing`.")
        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        self.attack_losses: Tuple[str, ...] = attack_losses

    @property
    def native_label_is_pytorch_format(self) -> bool:
        raise NotImplementedError

    @property
    def input_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def predict(
        self,
        x: np.ndarray,
        batch_size: int = 128,
        standardise_output: bool = False,
        **kwargs
    ) -> List[Dict[str, np.ndarray]]:
        """
        Perform prediction for a batch of inputs.
        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param batch_size: Batch size.
        :param standardise_output: True if output should be standardised to PyTorch format. Box coordinates will be
                                   scaled from [0, 1] to image dimensions, label index will be increased by 1 to adhere
                                   to COCO categories and the boxes will be changed to [x1, y1, x2, y2] format, with
                                   0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The
                 fields of the Dict are as follows:
                 - boxes [N, 4]: the boxes in [y1, x1, y2, x2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                                 Can be changed to PyTorch format with `standardise_output=True`.
                 - labels [N]: the labels for each image in TensorFlow format. Can be changed to PyTorch format with
                               `standardise_output=True`.
                 - scores [N]: the scores or each prediction.
        """
        self._model.eval()

        x = torch.from_numpy(x.copy())
        images = x.permute((0, 3, 1, 2)).clip(0., 1.) # NHWC -> NCHW
        batched_inputs = create_inputs(images, input_format=self.model.input_format)
        list_of_instances = self._model(batched_inputs)

        y_pred = []

        for instances in list_of_instances:
            instances = instances["instances"]

            pred = {}
            pred["boxes"] = instances.pred_boxes.tensor.cpu().numpy()
            # Detectron2 produces labels starting at 0 so we add 1 to make them 1-based since that is what PyTorch produces
            pred["labels"] = instances.pred_classes.cpu().numpy() + 1
            pred["scores"] = instances.scores.cpu().numpy()

            y_pred.append(pred)

        # Our model outputs everything in PyTorch format, so convert to TensorFlow format as necessary
        if not standardise_output:
            from art.estimators.object_detection.utils import convert_pt_to_tf

            # NOTE: This does an in-place update of y_pred
            convert_pt_to_tf(y=y_pred, height=x.shape[1], width=x.shape[2])

        return y_pred

    def loss_gradient(
        self,
        x: np.ndarray,
        y: Union[List[Dict[str, np.ndarray]], List[Dict[str, "torch.Tensor"]]],
        standardise_output: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.
        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param y: Targets of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict are
                  as follows:
                 - boxes [N, 4]: the boxes in [y1, x1, y2, x2] in scale [0, 1] (`standardise_output=False`) or
                                 [x1, y1, x2, y2] in image scale (`standardise_output=True`) format,
                                 with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                 - labels [N]: the labels for each image in TensorFlow (`standardise_output=False`) or PyTorch
                               (`standardise_output=True`) format
        :param standardise_output: True if `y` is provided in standardised PyTorch format. Box coordinates will be
                                   scaled back to [0, 1], label index will be decreased by 1 and the boxes will be
                                   changed from [x1, y1, x2, y2] to [y1, x1, y2, x2] format, with
                                   0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        :return: Loss gradients of the same shape as `x`.
        """
        # Deepcopy y since it can mess up Armory/ART otherwise
        y = deepcopy(y)

        # Convert labels to PyTorch format if they are not already
        if not standardise_output:
            from art.estimators.object_detection.utils import convert_tf_to_pt

            # NOTE: This does an in-place update of y
            convert_tf_to_pt(y=y, height=x.shape[1], width=x.shape[2])

        # Detectron2 wants labels starting at 0, so we subtract 1
        for i, _ in enumerate(y):
            y[i]["labels"] = y[i]["labels"] - 1

        # TODO: Switch to eval mode and make sure output is functional
        self._model.train()

        x_grad = torch.from_numpy(x.copy())
        x_grad.requires_grad_()

        images = x_grad.permute((0, 3, 1, 2)).clip(0., 1.) # NHWC -> NCHW
        gt_bboxes = [torch.from_numpy(gt['boxes']) for gt in y]
        gt_classes = [torch.from_numpy(gt['labels']) for gt in y]
        batched_inputs = create_inputs(images,
                                       input_format=self._model.input_format,
                                       gt_bboxes=gt_bboxes,
                                       gt_classes=gt_classes,
                                       bbox_mode=BoxMode.XYXY_ABS)
        with EventStorage():
            # FIXME: Multiple calls to this do not yield the same loss values. This is because the RPN samples
            #        proposals. We should figure out how to disable this and more accurately model inference.
            output = self._model(batched_inputs)

        # Compute the gradient and return
        loss = None
        for loss_name in self.attack_losses:
            if loss is None:
                loss = output[loss_name]
            else:
                loss = loss + output[loss_name]

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()  # type: ignore

        grads = x_grad.grad.cpu().numpy().copy()

        if self.clip_values is not None:
            grads = grads / self.clip_values[1]

        assert grads.shape == x.shape

        return grads

def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    config_file = model_kwargs['config_file']
    if weights_file is None:
        weights_file = model_kwargs['weights_file']

    if config_file.startswith('detectron2://'):
        config_file = config_file[len('detectron2://'):]
        config_file = get_config_file(config_file)
    elif config_file.startswith('oscar://'):
        config_file = config_file[len('oscar://'):]
        config_file = pkg_resources.resource_filename('oscar.model_zoo', config_file)
    elif config_file.startswith('armory://'):
        config_file = config_file[len('armory://'):]
        config_file = maybe_download_weights_from_s3(config_file)

    if weights_file.startswith('oscar://'):
        weights_file = weights_file[len('oscar://'):]
        weights_file = pkg_resources.resource_filename('oscar.model_zoo', weights_file)
    elif weights_file.startswith('armory://'):
        weights_file = weights_file[len('armory://'):]
        weights_file = maybe_download_weights_from_s3(weights_file)

    logger.info('config_file = %s', config_file)
    logger.info('weights_file = %s', weights_file)

    classifier = Detectron2Estimator(config_file, weights_file)

    return classifier
