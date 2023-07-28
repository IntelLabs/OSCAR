#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
from typing import Optional

import hydra
import torch
import torchvision
from armory.baseline_models.pytorch.carla_multimodality_object_detection_frcnn import MultimodalNaive
from art.estimators.object_detection import PyTorchFasterRCNN
from omegaconf import OmegaConf
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN

from oscar.models.likelihood import LikelihoodEstimator

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUSTOM_ARMORY_KEYS = ["predicted_hallucinations", "boxes_raw", "labels_raw", "scores_raw"]


class FilteredDetectionWrapper(torch.nn.Module):
    """Wrap a torchvision model and a LikelihoodEstimator like a torchvision model."""

    def __init__(self, object_detector, state_dict, ode_config) -> None:
        super().__init__()

        self.object_detector = object_detector
        self.num_channels = ode_config["num_channels"]

        # Instantiate likelihood model
        self.likelihood_model = LikelihoodEstimator(
            state_dict,
            num_channels=self.num_channels,
            threshold=ode_config["threshold"],
            method=ode_config["method"],
            step_size=ode_config["step_size"],
            device=DEVICE,
        )  # DEVICE needed explicitly for fixed PRNG
        self.likelihood_model.to(DEVICE)

    def forward(self, images, targets=None):
        # Get detections
        preds = self.object_detector(images, targets)

        if targets is None:
            # RGB
            if self.num_channels == 3:
                # Armory's inputs are between [0, 1] but score model wants [-1, 1]
                images = [inp * 2 - 1 for inp in images]
            # RGB + D
            elif self.num_channels == 4:
                # Armory's RGB inputs are between [0, 1] but score model wants [-1, 1]
                rgbs = [inp[:3] * 2 - 1 for inp in images]

                # Convert depth to linear
                depths = [inp[3] * 255 + inp[4] * 255 * 256 + inp[5] * 255 * 256 * 256 for inp in images]
                depths = [depth * 1000.0 / (256**3 - 1) for depth in depths]
                # Convert to log-scale (zero-guarded)
                depths = [torch.log2(1 + depth) / 2 - 2.5 for depth in depths]

                # Concatenate channels
                images = [torch.cat((rgb, depth[None, ...])) for (rgb, depth) in zip(rgbs, depths)]

            # Post-process detections
            filtered_preds = self.likelihood_model(images, preds)

            # Store non-processed results for custom scenarios
            for filtered_pred, pred in zip(filtered_preds, preds):
                filtered_pred["boxes_raw"] = pred["boxes"].detach().clone()
                filtered_pred["labels_raw"] = pred["labels"].detach().clone()
                filtered_pred["scores_raw"] = pred["scores"].detach().clone()
        else:
            # Leave losses intact
            filtered_preds = preds

        return filtered_preds


class FilteredPyTorchFasterRCNN(PyTorchFasterRCNN):
    """Adds an option to output more prediction keys to the ART PyTorchFasterRCNN."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, x, batch_size: int = 128, **kwargs):
        # Set model to evaluation mode
        self._model.eval()

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Convert samples into tensors
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0

        if self.channels_first:
            x_preprocessed = [torch.from_numpy(x_i / norm_factor).to(self.device) for x_i in x_preprocessed]
        else:
            x_preprocessed = [transform(x_i / norm_factor).to(self.device) for x_i in x_preprocessed]

        predictions = []
        # Run prediction
        num_batch = (len(x_preprocessed) + batch_size - 1) // batch_size
        for m in range(num_batch):
            # Batch using indices
            i_batch = x_preprocessed[m * batch_size : (m + 1) * batch_size]

            with torch.no_grad():
                predictions_x1y1x2y2 = self._model(i_batch)

            for prediction_x1y1x2y2 in predictions_x1y1x2y2:
                prediction = {}

                prediction["boxes"] = prediction_x1y1x2y2["boxes"].detach().cpu().numpy()
                prediction["labels"] = prediction_x1y1x2y2["labels"].detach().cpu().numpy()
                prediction["scores"] = prediction_x1y1x2y2["scores"].detach().cpu().numpy()
                if "masks" in prediction_x1y1x2y2:
                    prediction["masks"] = prediction_x1y1x2y2["masks"].detach().cpu().numpy().squeeze()

                # Pass-through for other custom outputs
                for key in CUSTOM_ARMORY_KEYS:
                    if key in prediction_x1y1x2y2:
                        prediction[key] = prediction_x1y1x2y2[key].detach().cpu().numpy()

                predictions.append(prediction)

        return predictions


class MARTModelWrapper(torch.nn.Module):
    """Wrap a LitModular model from MART like a torchvision model."""

    def __init__(self, model) -> None:
        super().__init__()

        self.model = model

        # Postprocessing with result routing.
        self.loss_keys = {
            "loss_classifier": "box_loss.loss_classifier",
            "loss_box_reg": "box_loss.loss_box_reg",
            "loss_objectness": "rpn_loss.loss_objectness",
            "loss_rpn_box_reg": "rpn_loss.loss_rpn_box_reg",
        }
        self.prediction_keys = "preds"

    def forward(self, images, targets=None):
        keys = self.loss_keys

        if targets is None:
            # Pseudo targets for calculating losses that are not used.
            device = images[0].device
            # Make a valid pseudo target because torchvision has a strict check.
            target = {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device),
                "labels": torch.zeros(1, dtype=torch.int64, device=device),
                "scores": torch.ones(1, device=device),
            }
            targets = [target] * len(images)

            keys = self.prediction_keys

        # Armory's inputs are between [0, 1] but MART wants inputs between [0, 255]
        images = [inp * 255 for inp in images]

        # Run the training sequence since it should output everything
        # Add model=None because input_adv_* may require it.
        # We will parse the result later in select_outputs().
        outputs = self.model(input=images, target=targets, model=None, step="training")
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


def get_art_model(
    model_kwargs: Optional[dict] = None,
    wrapper_kwargs: dict = {},
    weights_path: Optional[dict] = {},
) -> FilteredPyTorchFasterRCNN:
    """The order of arguments are fixed due to the invocation in armory.utils.config_loading.load_model()."""

    # No need to run maybe_download_weights_from_s3() here as Armory takes care of the weights_path dict.

    use_mart = model_kwargs.pop("use_mart", False)
    if use_mart:
        # Load the model architecture.
        config_yaml_path = weights_path["detector_yaml"]
        model_config = OmegaConf.load(config_yaml_path)
        model = hydra.utils.instantiate(model_config)

        # Load checkpoint/weights.
        checkpoint_path = weights_path["detector_checkpoint"]
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # model.modules.input_adv_training may contain unwanted parameters after we remove adversarial training.
        # FIXME: We had better fix the weight file and use strict loading, otherwise it may silence other errors.
        model.load_state_dict(state_dict, strict=False)

        # Wrap the MART model as a torchvision model.
        rcnn_model = MARTModelWrapper(model, **model_kwargs)
        rcnn_model.to(DEVICE)

    else:
        # Use ART model
        rcnn_model = fasterrcnn_resnet50_fpn(**model_kwargs)
        rcnn_model.to(DEVICE)

        checkpoint_path = weights_path["detector_checkpoint"]
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        assert "roi_heads.box_predictor.cls_score.bias" in checkpoint, "invalid checkpoint for current model, layers do no match."
        assert rcnn_model.roi_heads.box_predictor.cls_score.out_features == checkpoint["roi_heads.box_predictor.cls_score.bias"].shape[0], (
            f"provided model checkpoint does not match supplied model_kwargs['num_classes']: "
            f"{model_kwargs['num_classes']} != {checkpoint['roi_heads.box_predictor.cls_score.bias'].shape[0]}"
        )
        rcnn_model.load_state_dict(checkpoint)

    ode_config = wrapper_kwargs.pop("filtered_detection_wrapper", None)
    if ode_config is not None:
        # Load checkpoint/weights for score model.
        checkpoint_path = weights_path["filter_checkpoint"]
        score_state_dict = torch.load(checkpoint_path, map_location=DEVICE)

        # Wrap model with post-processing.
        rcnn_model = FilteredDetectionWrapper(rcnn_model, score_state_dict, ode_config)
        rcnn_model.to(DEVICE)

    wrapped_model = FilteredPyTorchFasterRCNN(
        rcnn_model,
        clip_values=(0.0, 1.0),
        channels_first=False,
        **wrapper_kwargs,
    )
    return wrapped_model


def get_art_model_mm(model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None) -> FilteredPyTorchFasterRCNN:

    use_mart = model_kwargs.pop("use_mart", False)
    assert not use_mart, "Use ART model (use_mart=False) for multimodal"

    num_classes = model_kwargs.pop("num_classes", 3)
    frcnn_kwargs = {arg: model_kwargs.pop(arg) for arg in ["min_size", "max_size", "box_detections_per_img"] if arg in model_kwargs}

    backbone = MultimodalNaive(**model_kwargs)
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        image_mean=[0.485, 0.456, 0.406, 0.0, 0.0, 0.0],
        image_std=[0.229, 0.224, 0.225, 1.0, 1.0, 1.0],
        **frcnn_kwargs,
    )
    model.to(DEVICE)

    checkpoint = torch.load(weights_path["detector_checkpoint"], map_location=DEVICE)
    model.load_state_dict(checkpoint)

    ode_config = wrapper_kwargs.pop("filtered_detection_wrapper", None)
    if ode_config is not None:
        # Load checkpoint/weights for score model.
        checkpoint_path = weights_path["filter_checkpoint"]
        score_state_dict = torch.load(checkpoint_path, map_location=DEVICE)

        # Wrap model with post-processing.
        model = FilteredDetectionWrapper(model, score_state_dict, ode_config)
        model.to(DEVICE)

    wrapped_model = FilteredPyTorchFasterRCNN(
        model,
        clip_values=(0.0, 1.0),
        channels_first=False,
        **wrapper_kwargs,
    )
    return wrapped_model
