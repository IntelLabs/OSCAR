#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#


import json

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMPipeline, DDPMScheduler
from diffusers.models import UNet2DModel
from score_sde_pytorch.models import ncsnpp  # noqa: Needed to register models
from score_sde_pytorch.models import utils as mutils
from score_sde_pytorch.models.ema import ExponentialMovingAverage
from score_sde_pytorch.sde_lib import subVPSDE
from torch.utils.checkpoint import checkpoint
from torchdiffeq import odeint as odeint
from torchdiffeq import odeint_adjoint as odeint_adjoint
from tqdm import tqdm


def get_preprocessing_pipe(unet_config_file, unet_weight_file, device):
    # Create SD pipeline using minimalistic loading
    scheduler = DDPMScheduler()  # Dummy with default parameters
    with open(unet_config_file) as handle:
        unet_config = json.load(handle)

    unet = UNet2DModel(**unet_config)
    # Convert self-attention state dictionary to recent `diffusers` version
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/modeling_utils.py#L875C27-L875C27
    state_dict = torch.load(unet_weight_file, map_location=device)
    unet._convert_deprecated_attention_blocks(state_dict)
    unet.load_state_dict(state_dict)

    pipe = DDPMPipeline(unet, scheduler)
    pipe = pipe.to(device, torch_dtype=torch.bfloat16)

    # Replace scheduler with DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return pipe


# Differentiable multistep resampling with gradient checkpointing
def multistep_resampling(image, pipe, desired_start, num_steps, num_channels, deterministic=False):
    # [0, 1] -> [-1, 1] only for RGB channels
    original_image = image
    image = [2 * img[:3] - 1.0 for img in image]
    assert len(image) == 1, "Batch processing not yet supported!"
    image = image[0][None, ...]
    orig_type = image.dtype
    image = image.type(pipe.unet.dtype)
    # Get RNG object
    g = torch.Generator(device=image.device)

    # Set step values
    num_inference_steps = int(pipe.scheduler.config.num_train_timesteps // desired_start * num_steps)
    pipe.scheduler.set_timesteps(num_inference_steps)

    # Select actual inference timesteps
    inference_timesteps = pipe.scheduler.timesteps[-(num_steps + 1) : -1]

    # Add noise to data
    if deterministic:
        g.manual_seed(2023)
    noise = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=g)
    recovered_image = pipe.scheduler.add_noise(image, noise, inference_timesteps[0])

    for idx, t in enumerate(inference_timesteps):
        # Predict noise
        model_output = checkpoint(pipe.unet, recovered_image, t, use_reentrant=False).sample

        # Reverse diffusion
        recovered_image = pipe.scheduler.step(model_output, t, recovered_image).prev_sample

    # [-1, 1] -> [0, 1], clamp and merge back with depth channels
    recovered_image = (recovered_image / 2 + 0.5).clamp(0, 1).type(orig_type)
    if num_channels == 4:
        recovered_image = [torch.cat((rec_img, img[3:]), dim=0) for (rec_img, img) in zip(recovered_image, original_image)]

    return [recovered_image[0]]


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps, is_differentiable):
        # Force gradients even when under no_grad context
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x, create_graph=is_differentiable)[0]
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


class ODEFunc(torch.nn.Module):
    def __init__(self, model, shape, drift_fn, div_fn, epsilon, is_differentiable):
        super().__init__()
        self.shape = shape
        self.model = model
        self.drift_fn = drift_fn
        self.div_fn = div_fn
        self.epsilon = epsilon
        self.is_differentiable = is_differentiable

    def forward(self, t, x):
        sample = torch.reshape(x[: -self.shape[0]], self.shape).type(torch.float32)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        # Checkpoint every drift / divergence wrapper calls
        drift = checkpoint(self.drift_fn, self.model, sample, vec_t, use_reentrant=False).reshape((-1,))
        logp_grad = checkpoint(self.div_fn, self.model, sample, vec_t, self.epsilon, self.is_differentiable, use_reentrant=False).reshape((-1,))

        return torch.concat([drift, logp_grad], dim=0)


class ODESolver(torch.nn.Module):
    def __init__(
        self,
        model,
        sde,
        inverse_scaler,
        timesteps=(1e-5, 1),
        rtol=1e-5,
        atol=1e-5,
        method="rk4",
        adjoint_grads=False,
        eps=1e-5,
        ode_step_size=0.01,
        epsilon=None,
        is_differentiable=False,
        shape=None,
    ):
        super().__init__()

        # Store objects
        self.model = model
        self.method = method
        self.sde, self.eps = sde, eps
        self.timesteps = torch.tensor(timesteps)
        self.rtol, self.atol = rtol, atol
        self.inverse_scaler = inverse_scaler
        self.ode_step_size = ode_step_size
        self.adjoint_grads = adjoint_grads
        self.is_differentiable = is_differentiable

        # Instantiate trace wrapper around model
        # !!! This is the nn.Module that is being called repeatedly during inference
        self.ode_func = ODEFunc(self.model, shape, self.drift_fn, self.div_fn, epsilon, self.is_differentiable)

    def drift_fn(self, model, x, t):
        score_fn = mutils.get_score_fn(self.sde, model, train=False, continuous=True)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = self.sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def div_fn(self, model, x, t, noise, is_differentiable):
        return get_div_fn(lambda xx, tt: self.drift_fn(model, xx, tt))(x, t, noise, is_differentiable)

    def forward(self, data):
        shape = data.shape
        init = torch.concat([torch.flatten(data), torch.zeros(shape[0], device=data.device)], dim=0)

        if self.method == "rk4" or self.method == "midpoint" or self.method == "euler":
            options = {"step_size": self.ode_step_size}
        elif self.method == "bosh3" or self.method == "adaptive_heun":
            options = None

        if self.adjoint_grads:
            solution = odeint_adjoint(self.ode_func, init, self.timesteps.to(data.device), options=options, rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            solution = odeint(self.ode_func, init, self.timesteps.to(data.device), options=options, rtol=self.rtol, atol=self.atol, method=self.method)

        zp = solution[-1]
        z = torch.reshape(zp[: -shape[0]], shape)
        delta_logp = torch.reshape(zp[-shape[0] :], (shape[0],))
        prior_logp = self.sde.prior_logp(z)
        bpd = -(prior_logp + delta_logp) / np.log(2)
        N = np.prod(shape[1:])
        nll = bpd / N

        return nll


class LikelihoodEstimator(torch.nn.Module):
    def __init__(self, state_dict, num_channels, threshold, device, patch_size=160, method="rk4", time_eps=1e-5, step_size=0.05, is_differentiable=False):
        super().__init__()

        # Patch size during NLL evaluation
        self.patch_size = patch_size
        self.num_channels = num_channels
        half_size = torch.tensor(self.patch_size // 2, dtype=torch.long)
        self.register_buffer("half_size", half_size)

        # Instantiate score-based diffusion model and load EMA weights
        config_score = state_dict["config"]
        score_model = mutils.create_model(config_score)
        ema_helper = ExponentialMovingAverage(score_model.parameters(), decay=config_score.model.ema_rate)

        assert "ema_state" in state_dict.keys(), "EMA model state not found in the checkpoint file"
        ema_helper.load_state_dict(state_dict["ema_state"])
        ema_helper.copy_to(score_model.parameters())
        self.register_module("score_model", score_model)
        self.score_model.eval()

        # Instantiate SDE object
        assert config_score.training.sde.lower() == "subvpsde", "SDEs other than subVP currently not supported"
        self.sde = subVPSDE(beta_min=config_score.model.beta_min, beta_max=config_score.model.beta_max, N=config_score.model.num_scales)

        # Sample a random but always-reproducible and fixed once-and-for-all
        # 'epsilon' tensor to use in the HS trace estimator wrapper function
        eps_size = (1, num_channels, self.patch_size, self.patch_size)
        g_cpu = torch.Generator(device="cpu")
        g_cpu.manual_seed(2023)
        ode_hs_epsilon = (torch.randint(low=0, high=2, size=eps_size, generator=g_cpu).float() * 2 - 1.0).to(device)
        self.register_buffer("ode_hs_epsilon", ode_hs_epsilon)

        # Instantiate ODE solver
        likelihood_fn = ODESolver(
            self.score_model,
            self.sde,
            inverse_scaler=None,
            method=method,
            eps=time_eps,
            ode_step_size=step_size,
            epsilon=self.ode_hs_epsilon,
            is_differentiable=is_differentiable,
            shape=self.ode_hs_epsilon.shape,
        )
        self.register_module("likelihood_fn", likelihood_fn)

        # NLL detection threshold
        self.nll_threshold = threshold

    # Score-based diffusion model should always stay in 'eval' mode
    def train(self, mode):
        super().train(mode)
        self.score_model.eval()

    def forward(self, images, preds):
        # Post-process detections sample-by-sample
        filtered_preds = []
        evaluated_nlls = []
        for image_idx, (image, pred) in enumerate(zip(images, preds)):
            # Initialize predicted hallucination flag for each object
            pred["predicted_hallucinations"] = torch.zeros(len(pred["boxes"]), device=pred["boxes"].device, dtype=torch.bool)

            # No objects detected in image, early exit
            if len(pred["boxes"]) == 0:
                filtered_preds.append(pred)
                continue

            # Surviving objects
            filtered_pred = {"boxes": [], "labels": [], "scores": []}
            # Create a likelihood map to mark done patches in the image
            likelihood_done_map = [torch.zeros(image.shape[-2], image.shape[-1], device=image.device)]  # Template map
            object_nll = torch.zeros(len(pred["boxes"]), device=image.device)

            # For each detection in the sample
            for object_idx, box in tqdm(enumerate(pred["boxes"])):
                x1, y1, x2, y2 = torch.floor(box).type(torch.long)

                # Get centers and dimensions
                cx = torch.round((x1 + x2) / 2).type(torch.long)
                cy = torch.round((y1 + y2) / 2).type(torch.long)
                w = x2 - x1
                h = y2 - y1

                # Adjust center if patch would overflow
                cx = torch.minimum(torch.maximum(cx, self.half_size), image.shape[-1] - self.half_size)
                cy = torch.minimum(torch.maximum(cy, self.half_size), image.shape[-2] - self.half_size)

                # Check if object was already fully included in a previous patch
                done_flag = torch.tensor(False, device=image.device)
                if object_idx > 0:
                    # Stack likelihood maps ad-hoc
                    stacked_maps = torch.stack(likelihood_done_map)
                    done_flag = torch.all(stacked_maps[:, cy - h // 2 : cy + h // 2, cx - w // 2 : cx + w // 2].reshape(stacked_maps.shape[0], -1), dim=-1)

                # If so, decide using NLL values from previous patches and majority vote
                if torch.any(done_flag):
                    previous_idx = torch.where(done_flag)[0]
                    previous_decisions = stacked_maps[previous_idx, cy, cx] > self.nll_threshold
                    flag_anomaly = torch.sum(previous_decisions) >= (len(previous_decisions) // 2)
                else:
                    # Get patch, evaluate likelihood, and fill completed map
                    local_patch = image[None, ..., cy - self.half_size : cy + self.half_size, cx - self.half_size : cx + self.half_size]

                    nll_est = self.likelihood_fn(local_patch)
                    flag_anomaly = nll_est > self.nll_threshold

                    # Mark as done only when actually evaluated
                    local_map = torch.zeros_like(likelihood_done_map[0])
                    local_map[cy - self.half_size : cy + self.half_size, cx - self.half_size : cx + self.half_size] = nll_est.detach()
                    likelihood_done_map.append(local_map)

                    # Store evaluated NLL
                    object_nll[object_idx] = nll_est

                # Store hallucination decision for this object
                pred["predicted_hallucinations"][object_idx] = flag_anomaly.item()
                if flag_anomaly.item() is False:
                    for key in filtered_pred.keys():
                        filtered_pred[key].append(pred[key][object_idx])

            # Save surviving boxes, if there are any
            if len(filtered_pred["boxes"]) > 0:
                for key in filtered_pred.keys():
                    filtered_pred[key] = torch.stack(filtered_pred[key], dim=0)
            # Empty detection tensors
            else:
                filtered_pred = {
                    "boxes": torch.zeros(size=(0, 4), device=image.device),
                    "labels": torch.zeros(size=(0,), device=image.device),
                    "scores": torch.zeros(size=(0,), device=image.device),
                }
            # Store hallucination decision for all objects in this image
            filtered_pred["predicted_hallucinations"] = pred["predicted_hallucinations"].detach().clone()
            filtered_preds.append(filtered_pred)

            # Store evaluated NLLs for all objects that were actually evaluated
            evaluated_nlls.append(object_nll)

        return filtered_preds, evaluated_nlls
