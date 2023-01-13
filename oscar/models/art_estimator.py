#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging

logger = logging.getLogger(__name__)

import torch
import hydra
from omegaconf import OmegaConf
from typing import Optional
from art.estimators.object_detection import PyTorchFasterRCNN
from oscar.transforms import ToDepth

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MARTModelWrapper(torch.nn.Module):
    """ Wrap a LitModular model from MART like a torchvision model.
    """

    def __init__(self, model, modality) -> None:
        super().__init__()

        self.model = model

        # Preprocess
        self.modality = modality
        self.to_depth = ToDepth(far=1000, scale=1)

        # Postprocessing with result routing.
        self.loss_keys = {
            "loss_classifier": "box_loss.loss_classifier",
            "loss_box_reg": "box_loss.loss_box_reg",
            "loss_objectness": "rpn_loss.loss_objectness",
            "loss_rpn_box_reg": "rpn_loss.loss_rpn_box_reg"
        }
        self.prediction_keys = "preds"

    def preprocess(self, images):
        # Make Armory input compliant to MART.

        if self.modality == "d":
            # Convert 6-channel RGBDepth to 1-Channel Depth in linear scale.
            images = [self.to_depth(inp[3:, :, :]) for inp in images]
        elif self.modality == "rgbd":
            # Convert 6-channel RGBDepth to 4-Channel RGBD in linear scale.
            images = [
                torch.vstack((inp[:3, :, :], self.to_depth(inp[3:, :, :])))
                for inp in images
            ]
        elif self.modality == "rgb" or self.modality == "rgbrgb":
            # We shouldn't need to convert depth for rgb, and rgbrgb, i.e. fusion with 3-channel raw depth.
            pass
        else:
            raise ValueError(f"Unrecognized modality {self.modality}")

        # Scaling from Armory's [0, 1] to MART's [0, 255]
        images = [inp * 255 for inp in images]
        return images

    def forward(self, images, targets=None):
        keys = self.loss_keys

        if targets is None:
            # Pseudo targets for calculating losses that are not used.
            device = images[0].device
            # Make a valid pseudo target because torchvision has a strict check.
            target = {
                "boxes": torch.tensor([[0., 0., 1., 1.]], device=device),
                "labels": torch.zeros(1, dtype=torch.int64, device=device),
                "scores": torch.ones(1, device=device)
            }
            targets = [target] * len(images)

            keys = self.prediction_keys

        # Make Armory input compliant to MART.
        images = self.preprocess(images)

        # Run the training sequence since it should output everything
        # Add model=None because input_adv_* may require it.
        # We will parse the result later in select_outputs().
        outputs = self.model(input=images,
                             target=targets,
                             model=None,
                             step="training")
        ret = self.select_outputs(outputs, keys)

        return ret

    @staticmethod
    def select_outputs(outputs, keys):
        # Remap outputs using keys as new keys and values
        if isinstance(keys, dict):
            selected_outputs = {k: outputs[v] for k, v in keys.items()}

        # Remap outputs as list using keys in list
        elif isinstance(keys, list):
            selected_outputs = [outputs[k] for k in keys]

        # Remap output as singular output using key
        else:
            selected_outputs = outputs[keys]

        return selected_outputs


def get_art_model(model_kwargs: Optional[dict] = None,
                  wrapper_kwargs: dict = {},
                  weights_path: Optional[dict] = {}) -> PyTorchFasterRCNN:
    """
    The order of arguments are fixed due to the invocation in armory.utils.config_loading.load_model().
    model_kwargs are not used, because the model config is in weights_path["config_yaml"].
    """

    # No need to run maybe_download_weights_from_s3() here as Armory takes care of the weights_path dict.

    # Load the model architecture.
    config_yaml_path = weights_path["config_yaml"]
    model_config = OmegaConf.load(config_yaml_path)
    model = hydra.utils.instantiate(model_config)

    # Load checkpoint/weights.
    checkpoint_path = weights_path["checkpoint"]
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # model.modules.input_adv_training may contain unwanted parameters after we remove adversarial training.
    # FIXME: We had better fix the weight file and use strict loading, otherwise it may silence other errors.
    model.load_state_dict(state_dict, strict=False)

    # Wrap the MART model as a torchvision model.
    rcnn_model = MARTModelWrapper(model, **model_kwargs)
    rcnn_model.to(DEVICE)

    wrapped_model = PyTorchFasterRCNN(
        rcnn_model,
        clip_values=(0.0, 1.0),
        channels_first=False,
        **wrapper_kwargs,
    )
    return wrapped_model
