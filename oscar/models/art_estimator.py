#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import types
from typing import Optional

import torch
import torchvision
from armory.baseline_models.pytorch.carla_multimodality_object_detection_frcnn import MultimodalNaive
from art.estimators.object_detection import PyTorchFasterRCNN
from torch.utils.checkpoint import checkpoint
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN

from oscar.models.likelihood import LikelihoodEstimator, get_preprocessing_pipe, multistep_resampling

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUSTOM_ARMORY_KEYS = ["predicted_hallucinations", "boxes_raw", "labels_raw", "scores_raw"]


class FilteredDetectionWrapper(torch.nn.Module):
    """Wrap a torchvision model and a LikelihoodEstimator like a torchvision model."""

    def __init__(self, object_detector, state_dict, ode_config, preproc_config, num_channels) -> None:
        super().__init__()

        self.object_detector = object_detector
        self.num_channels = num_channels

        # Instantiate preprocessing pipeline
        self.preproc_fn = lambda images, *args: images
        self.preprocess_pipe, self.init_preproc_t, self.num_preproc_steps, self.deterministic_preproc = None, None, None, None
        if preproc_config is not None:
            self.preproc_fn = multistep_resampling
            self.preprocess_pipe = get_preprocessing_pipe(preproc_config["unet_config"], preproc_config["unet_weights"], DEVICE)
            self.init_preproc_t = preproc_config["init_t"]
            self.num_preproc_steps = preproc_config["num_steps"]
            self.deterministic_preproc = preproc_config.pop("deterministic", False)

        # Instantiate likelihood model
        self.likelihood_model = lambda images, preds: (preds, torch.zeros(1, device=DEVICE))
        self.differentiable_ode = False
        if ode_config is not None:
            self.likelihood_model = LikelihoodEstimator(
                state_dict,
                num_channels=self.num_channels,
                threshold=ode_config["threshold"],
                method=ode_config["method"],
                step_size=ode_config["step_size"],
                is_differentiable=ode_config["is_differentiable"],
                device=DEVICE,
            )  # DEVICE needed explicitly for fixed PRNG
            self.likelihood_model.to(DEVICE)
            self.differentiable_ode = ode_config["is_differentiable"]

    def forward(self, images, targets=None):
        # Preprocess samples
        preproc_images = checkpoint(
            self.preproc_fn,
            images,
            self.preprocess_pipe,
            self.init_preproc_t,
            self.num_preproc_steps,
            self.num_channels,
            self.deterministic_preproc,
            use_reentrant=False,
        )

        # Get detections and losses
        losses, preds = self.object_detector(preproc_images, targets)

        # In training mode, get predictions separately by temporarily forcing evaluation mode
        # The reason is that in "training mode", predictions are empty, e.g. https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py#L811
        if targets is not None and self.differentiable_ode:
            self.object_detector.train(False)
            with torch.no_grad():
                _, preds = self.object_detector(preproc_images, None)
            self.object_detector.train(True)

        # Scale RGB samples
        if self.num_channels == 3:
            # Armory's inputs are between [0, 1] but score model wants [-1, 1]
            images = [inp * 2 - 1 for inp in images]
        # Scale RGB + D samples
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

        # Post-process detections using unaltered images in two cases:
        # 1. Whenever Armory runs plain inference (training = False)
        # 2. And whenever we want to backpropagate through post-processing
        if targets is None or self.differentiable_ode:
            filtered_preds, evaluated_nlls = self.likelihood_model(images, preds)

            # Store non-processed results for custom scenarios
            for filtered_pred, pred in zip(filtered_preds, preds):
                filtered_pred["boxes_raw"] = pred["boxes"].detach().clone()
                filtered_pred["labels_raw"] = pred["labels"].detach().clone()
                filtered_pred["scores_raw"] = pred["scores"].detach().clone()

            # Add "nll_loss" to output dictionary for each sample
            losses["loss_nll"] = torch.sum(evaluated_nlls[0])

        else:
            # Don't use post-processing in the adversarial optimization loop
            filtered_preds = losses

        # Restore torchvision/armory output convention
        if targets is None:
            return filtered_preds
        else:
            return losses


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
        assert False, "'use_mart' flag is not supported for Eval8 submission"
    else:
        # Use Torchvision model
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

        # Patch forward method to always return (losses, detections)
        def eager_outputs(self, losses, detections):
            return losses, detections

        rcnn_model.eager_outputs = types.MethodType(eager_outputs, rcnn_model)

        preproc_config = wrapper_kwargs.pop("preprocessing_wrapper", None)
        if preproc_config is not None:
            preproc_config["unet_config"] = weights_path["preproc_unet_config"]
            preproc_config["unet_weights"] = weights_path["preproc_unet_weights"]

        ode_config = wrapper_kwargs.pop("filtered_detection_wrapper", None)
        score_state_dict = None
        if ode_config is not None:
            # Load checkpoint/weights for score model.
            checkpoint_path = weights_path["filter_checkpoint"]
            score_state_dict = torch.load(checkpoint_path, map_location=DEVICE)

            # Determine whether we need to backpropagate through ODE solver or not
            used_losses = wrapper_kwargs.get("attack_losses", [])
            ode_config["is_differentiable"] = "loss_nll" in used_losses

        # Wrap model with pre- and post-processing.
        rcnn_model = FilteredDetectionWrapper(rcnn_model, score_state_dict, ode_config, preproc_config, num_channels=3)
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
    if use_mart:
        assert False, "'use_mart' flag is not supported for Eval8 submission"

    else:
        num_classes = model_kwargs.pop("num_classes", 3)
        frcnn_kwargs = model_kwargs.pop("frcnn_kwargs", {})
        frcnn_kwargs = {
            arg: model_kwargs.pop(arg)
            for arg in [
                "min_size",
                "max_size",
                "box_detections_per_img",
                "box_score_thresh",
                "box_detections_per_img",
                "rpn_score_thresh",
                "rpn_pre_nms_top_n_train",
                "rpn_pre_nms_top_n_test",
                "rpn_post_nms_top_n_train",
                "rpn_post_nms_top_n_test",
            ]
            if arg in model_kwargs
        }

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

        # Patch forward method to always return (losses, detections)
        def eager_outputs(self, losses, detections):
            return losses, detections

        model.eager_outputs = types.MethodType(eager_outputs, model)

        preproc_config = wrapper_kwargs.pop("preprocessing_wrapper", None)
        if preproc_config is not None:
            preproc_config["unet_config"] = weights_path["preproc_unet_config"]
            preproc_config["unet_weights"] = weights_path["preproc_unet_weights"]

        ode_config = wrapper_kwargs.pop("filtered_detection_wrapper", None)
        score_state_dict = None
        if ode_config is not None:
            # Load checkpoint/weights for score model.
            checkpoint_path = weights_path["filter_checkpoint"]
            score_state_dict = torch.load(checkpoint_path, map_location=DEVICE)

            # Determine whether we need to backpropagate through ODE solver or not
            used_losses = wrapper_kwargs.get("attack_losses", [])
            ode_config["is_differentiable"] = "loss_nll" in used_losses

        # Wrap model with post-processing.
        model = FilteredDetectionWrapper(model, score_state_dict, ode_config, preproc_config, num_channels=4)
        model.to(DEVICE)

    wrapped_model = FilteredPyTorchFasterRCNN(
        model,
        clip_values=(0.0, 1.0),
        channels_first=False,
        **wrapper_kwargs,
    )
    return wrapped_model
