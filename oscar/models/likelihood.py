#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#


import numpy as np
import torch
from score_sde_pytorch.models import ncsnpp  # noqa: Needed to register models
from score_sde_pytorch.models import utils as mutils
from score_sde_pytorch.models.ema import ExponentialMovingAverage
from score_sde_pytorch.sde_lib import subVPSDE
from torchdiffeq import odeint as odeint
from torchdiffeq import odeint_adjoint as odeint_adjoint
from tqdm import tqdm


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        # Force gradients even when under no_grad context
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


class ODEFunc(torch.nn.Module):
    def __init__(self, model, shape, drift_fn, div_fn, epsilon):
        super().__init__()
        self.shape = shape
        self.model = model
        self.drift_fn = drift_fn
        self.div_fn = div_fn
        self.epsilon = epsilon

    def forward(self, t, x):
        sample = torch.reshape(x[: -self.shape[0]], self.shape).type(torch.float32)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = self.drift_fn(self.model, sample, vec_t).reshape((-1,))
        logp_grad = self.div_fn(self.model, sample, vec_t, self.epsilon).reshape((-1,))

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
        adjoint_grads=True,
        eps=1e-5,
        ode_step_size=0.01,
        epsilon=None,
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

        # Instantiate trace wrapper around model
        # !!! This is the nn.Module that is being called repeatedly during inference
        self.ode_func = ODEFunc(self.model, shape, self.drift_fn, self.div_fn, epsilon)

    def drift_fn(self, model, x, t):
        score_fn = mutils.get_score_fn(self.sde, model, train=False, continuous=True)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = self.sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def div_fn(self, model, x, t, noise):
        return get_div_fn(lambda xx, tt: self.drift_fn(model, xx, tt))(x, t, noise)

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
    def __init__(self, state_dict, num_channels, threshold, device, patch_size=160, method="rk4", time_eps=1e-5, step_size=0.05):
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
            likelihood_done_map = torch.zeros(len(pred["boxes"]), image.shape[-2], image.shape[-1], device=image.device)

            # For each detection in the sample
            for object_idx, box in tqdm(enumerate(pred["boxes"])):
                x1, y1, x2, y2 = torch.floor(box).type(torch.long)

                # Get centers and dimensions
                cx = torch.round((x1 + x2) / 2).type(torch.long)
                cy = torch.round((y1 + y2) / 2).type(torch.long)
                w = x2 - x1
                h = y2 - y1

                # Check if object was already fully included in a previous patch
                done_flag = torch.tensor(False, device=image.device)
                if object_idx > 0:
                    done_flag = torch.all(
                        likelihood_done_map[:object_idx, cy - h // 2 : cy + h // 2, cx - w // 2 : cx + w // 2].reshape(object_idx, -1), dim=-1
                    )

                # If so, decide using NLL values from previous patches and majority vote
                if torch.any(done_flag):
                    previous_idx = torch.where(done_flag)[0]
                    previous_decisions = likelihood_done_map[previous_idx, cy, cx] > self.nll_threshold
                    flag_anomaly = torch.sum(previous_decisions) >= (len(previous_decisions) // 2)
                else:
                    # Adjust center if patch would overflow
                    cx = torch.minimum(torch.maximum(cx, self.half_size), image.shape[-1] - self.half_size)
                    cy = torch.minimum(torch.maximum(cy, self.half_size), image.shape[-2] - self.half_size)

                    # Get patch, evaluate likelihood, and fill completed map
                    local_patch = image[None, ..., cy - self.half_size : cy + self.half_size, cx - self.half_size : cx + self.half_size]

                    nll_est = self.likelihood_fn(local_patch)
                    flag_anomaly = nll_est > self.nll_threshold

                    # Mark as done only when actually evaluated
                    likelihood_done_map[object_idx, cy - self.half_size : cy + self.half_size, cx - self.half_size : cx + self.half_size] = nll_est

                # Store hallucination decision for this object
                pred["predicted_hallucinations"][object_idx] = flag_anomaly.item()
                if flag_anomaly.item() is False:
                    for key in filtered_pred.keys():
                        filtered_pred[key].append(pred[key][object_idx])

            # Save surviving boxes, if there are any
            if len(filtered_pred["boxes"]) > 0:
                for key in filtered_pred.keys():
                    filtered_pred[key] = torch.stack(filtered_pred[key], dim=0)
                # Store hallucination decision for all objects in this image
                filtered_pred["predicted_hallucinations"] = pred["predicted_hallucinations"].detach().clone()
            # Empty detection tensors
            else:
                filtered_pred = {
                    "boxes": torch.zeros(size=(0, 4), device=image.device),
                    "labels": torch.zeros(size=(0,), device=image.device),
                    "scores": torch.zeros(size=(0,), device=image.device),
                    "predicted_hallucinations": torch.zeros(size=(0,), device=image.device, dtype=torch.bool),
                }
            filtered_preds.append(filtered_pred)

        return filtered_preds
