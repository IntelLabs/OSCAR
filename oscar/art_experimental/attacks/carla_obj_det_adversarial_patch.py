import os
from typing import Optional

from oscar.art_experimental.attacks.adversarial_patch_pytorch import (
    AdversarialPatchPyTorch,
)
import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image

from armory import paths
from armory.art_experimental.attacks.carla_obj_det_utils import (
    linear_depth_to_rgb,
    linear_to_log,
    log_to_linear,
    rgb_depth_to_linear,
)
from armory.logs import log

from torchvision.transforms.functional import resize, perspective
from torch.utils.checkpoint import checkpoint

from diffusers import DDPMPipeline, StableDiffusionXLPipeline
from diffusers.utils import randn_tensor
from diffusers.schedulers import DDIMScheduler

VALID_IMAGE_TYPES = ["image/png", "image/jpeg", "image/jpg"]


class CARLADiffusionPatchPyTorch(AdversarialPatchPyTorch):
    """
    Apply patch attack to RGB channels and (optionally) masked PGD attack to depth channels.
    """

    def __init__(self, estimator, **kwargs):

        # Maximum depth perturbation from a flat patch
        self.depth_delta_meters = kwargs.pop("depth_delta_meters", 3)
        self.learning_rate_depth = kwargs.pop("learning_rate_depth", 0.0001)
        self.depth_perturbation = None
        self.min_depth = None
        self.max_depth = None
        self.patch_base_image = kwargs.pop("patch_base_image", None)
        self.patch_init = kwargs.pop("patch_init", [0, 255])

        # HSV bounds are user-defined to limit perturbation regions
        self.hsv_lower_bound = np.array(
            kwargs.pop("hsv_lower_bound", [0, 0, 0])
        )  # [0, 0, 0] means unbounded below
        self.hsv_upper_bound = np.array(
            kwargs.pop("hsv_upper_bound", [255, 255, 255])
        )  # [255, 255, 255] means unbounded above

        # Desired attack type
        self.attack_method = kwargs.pop("attack_method", "inpaint")

        # Regularization hyper-parameters
        self.patch_box_penalty_weight = kwargs.pop("patch_box_penalty_weight", 1) # Soft penalize everything outside the valid pixel box
        self.patched_input_tv_loss_weight = kwargs.pop("patched_input_tv_loss", 0.)
        self.masked_patched_input_tv_loss_weight = kwargs.pop("masked_patched_input_tv_loss", 0.)

        # Diffusion parameters
        self.prompt = kwargs.pop("prompt", "")
        self.num_reverse_steps = kwargs.pop("num_reverse_steps", 20)
        self.seed = kwargs.pop("seed", 2023) # Seed for reverse diffusion
        self.model_path = kwargs.pop("model_path", None)
        self.text_guidance = kwargs.pop("text_guidance", 5.0) # Classifier-free guidance weight
        assert self.num_reverse_steps <= 1000, "Default pipeline is limited to 1000 discrete time points!"

        # Initialize estimator
        super().__init__(estimator=estimator, **kwargs)
        self.sample_index = 0

        # Initialize diffusion pipeline
        if self.attack_method == "inpaint":
            # Custom DDPM pipeline
            pipe = DDPMPipeline.from_pretrained(self.model_path)
            pipe.set_progress_bar_config(disable=True)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif self.attack_method == "end-to-end":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", variant='fp16')
            # Always keep some parts of the VAE in float32 due to
            # https://github.com/huggingface/diffusers/blob/29f15673ed5c14e4843d7c837890910207f72129/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L860C31-L860C31
            pipe.upcast_vae()

        self.pipe = pipe.to(self.estimator.device)
        # Get RNG object
        self.rng_state = torch.Generator(device=self.pipe.device)


    def prepare_diffusion(self, prompt):
        assert type(prompt) == str, "Prompt must be a string!"
        prompt          = [prompt]
        negative_prompt = [""] * len(prompt)

        self.diff_height, self.diff_width = 1024, 1024 # Native resolution for SDXL
        # Configure scheduler timesteps
        self.pipe.scheduler.set_timesteps(self.num_reverse_steps, device=self.pipe.device)

        # Pre-compute prompt embeddings
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            self.pipe.encode_prompt(prompt=prompt, device=self.pipe.device, num_images_per_prompt=1, negative_prompt=negative_prompt)

        # Other static embeddings and parameters
        add_time_ids = self.pipe._get_add_time_ids(
            (self.diff_height, self.diff_width), (0, 0), (self.diff_height, self.diff_width),
            dtype=prompt_embeds.dtype).to(self.pipe.device)
        self.extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(self.rng_state, 0)

        self.prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        self.add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        self.add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)


    def create_initial_image(self, size, hsv_lower_bound, hsv_upper_bound):
        """
        Create initial patch based on a user-defined image and
        create perturbation mask based on HSV bounds
        """
        module_path = globals()["__file__"]
        module_folder = os.path.dirname(module_path)
        # user-defined image is assumed to reside in the same location as the attack module
        patch_base_image_path = os.path.abspath(
            os.path.join(module_folder, self.patch_base_image)
        )
        # if the image does not exist, check cwd
        if not os.path.exists(patch_base_image_path):
            patch_base_image_path = os.path.abspath(
                os.path.join(paths.runtime_paths().cwd, self.patch_base_image)
            )
        # image not in cwd or module, check if it is a url to an image
        if not os.path.exists(patch_base_image_path):
            # Send a HEAD request
            response = requests.head(self.patch_base_image, allow_redirects=True)

            # Check the status code
            if response.status_code != 200:
                raise FileNotFoundError(
                    f"Cannot find patch base image at {self.patch_base_image}. "
                    f"Make sure it is in your cwd or {module_folder} or provide a valid url."
                )
            # If the status code is 200, check the content type
            content_type = response.headers.get("content-type")
            if content_type not in VALID_IMAGE_TYPES:
                raise ValueError(
                    f"Returned content at {self.patch_base_image} is not a valid image type. "
                    f"Expected types are {VALID_IMAGE_TYPES}, but received {content_type}"
                )

            # If content type is valid, download the image
            response = requests.get(self.patch_base_image, allow_redirects=True)
            im = cv2.imdecode(
                np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR
            )
        else:
            im = cv2.imread(patch_base_image_path)

        im = cv2.resize(im, size)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        # find the colors within the boundaries
        mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)
        mask = np.expand_dims(mask, 2)
        # cv2.imwrite(
        #     "mask.png", mask
        # )  # visualize perturbable regions. Comment out if not needed.

        patch_base = np.transpose(im, (2, 0, 1))
        patch_base = patch_base / 255.0
        mask = np.transpose(mask, (2, 0, 1))
        mask = mask / 255.0
        return patch_base, mask


    def _diffusion_sampling(self, latents, num_steps, guidance):
        for step_idx in range(num_steps):
            # Get the current time and noise power
            t = self.pipe.scheduler.timesteps[step_idx]
            # Apply checkpointed diffusion step
            latents = checkpoint(self._diffusion_step, latents, t, guidance, use_reentrant=False)

        return latents


    def _diffusion_step(self, latents, t, guidance, learning_rate=1):
        # Normalized latents
        normalized_latents = self.pipe.scheduler.scale_model_input(torch.cat([latents.to(self.pipe.unet.dtype),
                                                                              latents.to(self.pipe.unet.dtype)]), t)

        # Model prediction
        noise_pred = self.pipe.unet(
            normalized_latents, t,
            encoder_hidden_states=self.prompt_embeds,
            cross_attention_kwargs=None,
            added_cond_kwargs={"text_embeds": self.add_text_embeds,
                               "time_ids": self.add_time_ids},
            return_dict=False,
        )[0]

        # Bias gradient of log-p.d.f. towards text prompt
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

        # Reverse diffusion step
        new_latents = self.pipe.scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs, return_dict=False)[0]

        # Scale score function with its learning rate
        score_update = new_latents - latents
        latents      = latents + score_update * learning_rate

        return latents


    def _train_step_end_to_end(
        self,
        images: "torch.Tensor",
        target: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":

        # Create fresh optimization variable to avoid memory leaks
        latents_var = torch.tensor(self._latents, requires_grad=True, device=self.estimator.device, dtype=self.pipe.unet.dtype)
        # Run the entire reverse diffusion process
        unnorm_latents = self._diffusion_sampling(latents_var, self.num_reverse_steps, self.text_guidance)
        norm_latents = unnorm_latents.to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)

        # Get patch from latents and de-normalize to [0, 1]
        local_patch = self.pipe.vae.decode(norm_latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        local_patch = (1 + local_patch[0]) / 2

        # Penalize decoding out-of-bounds pixels using linear loss
        reg_loss = torch.tensor(0., dtype=local_patch.dtype, device=local_patch.device)
        with torch.no_grad():
            negative_pixels = local_patch.flatten() < self.estimator.clip_values[0]
            positive_pixels = local_patch.flatten() > self.estimator.clip_values[1]
        if torch.any(negative_pixels).item():
            reg_loss = reg_loss + torch.mean(self.estimator.clip_values[0] - local_patch.flatten()[negative_pixels])
        if torch.any(positive_pixels).item():
            reg_loss = reg_loss + torch.mean(local_patch.flatten()[positive_pixels] - self.estimator.clip_values[1])

        # Clip w/ passthrough and resize to scene-compatible shape
        local_patch = local_patch + (torch.clip(local_patch, self.estimator.clip_values[0],
                                                self.estimator.clip_values[1]) - local_patch).detach()
        self._patch = resize(local_patch, self.patch_shape[1:])

        # Pass through downstream model and back-propagate
        det_loss = self._loss(images, target, mask)
        loss     = det_loss - self.patch_box_penalty_weight * reg_loss
        # Calling torch.autograd.grad and checkpointing with use_reentrant = False to not get memory leaks
        latent_grads = torch.autograd.grad(loss, latents_var)[0].cpu().detach().numpy()

        if self._optimizer_string == "pgd":
            # Update latents using gradient descent without projection
            self._latents = self._latents + self.learning_rate * latent_grads
        else:
            raise ValueError("Adam optimizer for end-to-end diffusion patch not supported.")

        return


    @torch.no_grad()
    def _train_step_inpainting(
        self,
        images: "torch.Tensor",
        target: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        # Generate patched input
        if mask is None:
            _, mask = self._random_overlay(images, self._patch, return_mask=True) # NHWC

        # Convert patched input and mask to be same shape and input range for diffusion
        diffusion_input = images
        diffusion_input = torch.permute(diffusion_input, (0, 3, 1, 2)) # NHWC -> NCHW
        diffusion_input = 2*diffusion_input - 1
        diffusion_input = F.interpolate(diffusion_input, size=self.pipe.unet.config.sample_size, mode="bilinear")

        diffusion_mask = mask
        diffusion_mask = torch.permute(diffusion_mask, (0, 3, 1, 2)) # NHWC -> NCHW
        diffusion_mask = F.interpolate(diffusion_mask, size=self.pipe.unet.config.sample_size, mode="nearest")

        # We save diffusion state in diffusion_output. If we have no state, then we must start from gaussian noise.
        if self.diffusion_output is None:
            diffusion_output = randn_tensor(diffusion_input.shape, device=diffusion_input.device, generator=self.rng_state)
        else:
            diffusion_output = self.diffusion_output

        # Add noise to original input for current timestep and mask diffused image and static image
        t = next(self.timesteps)
        noise = randn_tensor(diffusion_output.shape, device=diffusion_output.device, generator=self.rng_state)
        noised_input = self.pipe.scheduler.add_noise(diffusion_input, noise, t)
        diffusion_output = diffusion_mask * diffusion_output + (1 - diffusion_mask) * noised_input

        # Compute and normalize diffusion gradient (i.e., reverse diffusion process)
        diffusion_grad = self.pipe.unet(diffusion_output, t).sample
        diffusion_grad_norm = torch.linalg.norm(diffusion_grad)
        diffusion_grad = diffusion_grad / diffusion_grad_norm

        # Compute and normalize model gradient (i.e., backprop through model)
        model_weight = -self.learning_rate # negative because we are running untargeted target and scheduler.step minimizes
        model_grad = torch.zeros_like(diffusion_grad)
        model_grad_norm = torch.tensor(0)
        loss = torch.tensor(0)

        if model_weight != 0:
            with torch.enable_grad():
                model_input = diffusion_output
                model_input.requires_grad_(True)

                model_image = F.interpolate(model_input, size=(images.shape[1], images.shape[2]), mode="nearest")
                model_image = (model_image + 1) / 2
                model_image = torch.permute(model_image, (0, 2, 3, 1)) # NCHW -> NHWC
                model_image = model_image + (model_image.clamp(0, 1) - model_image).detach()

                loss = self._loss(model_image, target, mask)

                if model_input.grad is not None:
                    model_input.grad.zero_()
                loss.backward()

                model_grad = model_input.grad
                model_grad_norm = torch.linalg.norm(model_grad)
                model_grad = model_grad / model_grad_norm

        # Combine normalized diffusion and model gradients and take a step
        diffusion_grad = diffusion_grad + model_weight * model_grad
        diffusion_grad = diffusion_grad_norm * diffusion_grad / torch.linalg.norm(diffusion_grad)
        diffusion_output = self.pipe.scheduler.step(diffusion_grad, t, diffusion_output).prev_sample

        # Verbose
        if t.item() % 10 == 0:
            print("Reverse diffusion t =", t.item(), "loss =", loss.item(), "diffusion_grad =", diffusion_grad_norm.item(), "model_grad =", model_grad_norm.item())

        # Homography warp the patch to a 2D rectangular tensor
        if t == 0:
            diffusion_image = diffusion_output
            diffusion_image = (diffusion_image + 1) / 2
            diffusion_image = diffusion_image.clamp(0, 1)

            startpoints  = self.gs_coords[0]
            endpoints = [[0, 0],
                         [self.patch_shape[2] - 1, 0],
                         [self.patch_shape[2] - 1, self.patch_shape[1] - 1],
                         [0, self.patch_shape[1] - 1]
                         ]

            self._patch = perspective(
                img=diffusion_image,
                startpoints=startpoints,
                endpoints=endpoints,
                interpolation=2, # Bilinear
                fill=0, # Black pixels
                )[0]

            # Extract only the patch from the morphed scene
            self._patch = self._patch[..., :self.patch_shape[1], :self.patch_shape[2]]
            self._patch = torch.clamp(self._patch, min=self.estimator.clip_values[0], max=self.estimator.clip_values[1])

        # Save diffusion output for next iteration
        self.diffusion_output = diffusion_output

        return


    def _train_step(
        self,
        images: "torch.Tensor",
        target: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":

        if self.attack_method == "inpaint":
            self._train_step_inpainting(images, target, mask)

        elif self.attack_method == "end-to-end":
            self._train_step_end_to_end(images, target, mask)


    def _get_circular_patch_mask(
        self, nb_samples: int, sharpness: int = 40
    ) -> "torch.Tensor":
        """
        Return a circular patch mask.
        """
        import torch  # lgtm [py/repeated-import]

        image_mask = np.ones(
            (self.patch_shape[self.i_h_patch], self.patch_shape[self.i_w_patch])
        )

        image_mask = np.expand_dims(image_mask, axis=0)
        image_mask = np.broadcast_to(image_mask, self.patch_shape)
        image_mask = torch.Tensor(np.array(image_mask)).to(self.estimator.device)
        image_mask = torch.stack([image_mask] * nb_samples, dim=0)
        return image_mask

    def _random_overlay(
        self,
        images: "torch.Tensor",
        patch: "torch.Tensor",
        scale: Optional[float] = None,
        mask: Optional["torch.Tensor"] = None,
        return_mask: Optional[bool] = False,
    ) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]
        import torchvision

        # Ensure channels-first
        if not self.estimator.channels_first:
            images = torch.permute(images, (0, 3, 1, 2))

        nb_samples = images.shape[0]

        images_rgb = images[:, :3, :, :]
        if images.shape[1] == 6:
            images_depth = images[:, 3:, :, :]

        image_mask = self._get_circular_patch_mask(nb_samples=nb_samples)
        image_mask = image_mask.float()

        self.image_shape = images_rgb.shape[1:]

        pad_h_before = int(
            (self.image_shape[self.i_h] - image_mask.shape[self.i_h_patch + 1]) / 2
        )
        pad_h_after = int(
            self.image_shape[self.i_h]
            - pad_h_before
            - image_mask.shape[self.i_h_patch + 1]
        )

        pad_w_before = int(
            (self.image_shape[self.i_w] - image_mask.shape[self.i_w_patch + 1]) / 2
        )
        pad_w_after = int(
            self.image_shape[self.i_w]
            - pad_w_before
            - image_mask.shape[self.i_w_patch + 1]
        )

        image_mask = torchvision.transforms.functional.pad(
            img=image_mask,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )

        if self.nb_dims == 4:
            image_mask = torch.unsqueeze(image_mask, dim=1)
            image_mask = torch.repeat_interleave(
                image_mask, dim=1, repeats=self.input_shape[0]
            )

        image_mask = image_mask.float()

        patch = patch.float()
        padded_patch = torch.stack([patch] * nb_samples)

        padded_patch = torchvision.transforms.functional.pad(
            img=padded_patch,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )

        if self.nb_dims == 4:
            padded_patch = torch.unsqueeze(padded_patch, dim=1)
            padded_patch = torch.repeat_interleave(
                padded_patch, dim=1, repeats=self.input_shape[0]
            )

        padded_patch = padded_patch.float()

        image_mask_list = []
        padded_patch_list = []

        for i_sample in range(nb_samples):

            image_mask_i = image_mask[i_sample]

            height = padded_patch.shape[self.i_h + 1]
            width = padded_patch.shape[self.i_w + 1]

            startpoints = [
                [pad_w_before, pad_h_before],
                [width - pad_w_after - 1, pad_h_before],
                [width - pad_w_after - 1, height - pad_h_after - 1],
                [pad_w_before, height - pad_h_after - 1],
            ]

            endpoints = self.gs_coords[
                i_sample
            ]  # [topleft, topright, botright, botleft]
            enlarged_coords = np.copy(
                endpoints
            )  # enlarge the green screen coordinates a bit to fully cover the screen
            pad_amt_x = int(0.03 * (enlarged_coords[2, 0] - enlarged_coords[0, 0]))
            pad_amt_y = int(0.03 * (enlarged_coords[2, 1] - enlarged_coords[0, 1]))
            enlarged_coords[0, 0] -= pad_amt_x
            enlarged_coords[0, 1] -= pad_amt_y
            enlarged_coords[1, 0] += pad_amt_x
            enlarged_coords[1, 1] -= pad_amt_y
            enlarged_coords[2, 0] += pad_amt_x
            enlarged_coords[2, 1] += pad_amt_y
            enlarged_coords[3, 0] -= pad_amt_x
            enlarged_coords[3, 1] += pad_amt_y
            endpoints = enlarged_coords

            image_mask_i = torchvision.transforms.functional.perspective(
                img=image_mask_i,
                startpoints=startpoints,
                endpoints=endpoints,
                interpolation=2,
                fill=0,  # None
            )

            image_mask_list.append(image_mask_i)

            padded_patch_i = padded_patch[i_sample]

            padded_patch_i = torchvision.transforms.functional.perspective(
                img=padded_patch_i,
                startpoints=startpoints,
                endpoints=endpoints,
                interpolation=2,
                fill=0,  # None
            )

            padded_patch_list.append(padded_patch_i)

        image_mask = torch.stack(image_mask_list, dim=0)
        padded_patch = torch.stack(padded_patch_list, dim=0)
        inverted_mask = (
            torch.from_numpy(np.ones(shape=image_mask.shape, dtype=np.float32)).to(
                self.estimator.device
            )
            - image_mask
        )

        foreground_mask = torch.all(
            torch.tensor(self.binarized_patch_mask == 0), dim=-1, keepdim=True
        ).to(self.estimator.device)
        foreground_mask = torch.permute(foreground_mask, (2, 0, 1))
        foreground_mask = torch.unsqueeze(foreground_mask, dim=0)

        # Adjust green screen brightness
        v_avg = (
            0.5647  # average V value (in HSV) for the green screen, which is #00903a
        )
        green_screen = images_rgb * image_mask
        values, _ = torch.max(green_screen, dim=1, keepdim=True)
        values_ratio = values / v_avg
        values_ratio = torch.repeat_interleave(values_ratio, dim=1, repeats=3)

        patched_images = (
            images_rgb * inverted_mask
            + padded_patch * values_ratio * image_mask
            - padded_patch * values_ratio * foreground_mask * image_mask
            + images_rgb * foreground_mask * image_mask
        )

        patched_images = torch.clamp(
            patched_images,
            min=self.estimator.clip_values[0],
            max=self.estimator.clip_values[1],
        )

        patched_mask = image_mask * ~foreground_mask
        if not self.estimator.channels_first:
            patched_images = torch.permute(patched_images, (0, 2, 3, 1))
            patched_mask = torch.permute(patched_mask, (0, 2, 3, 1))

        # Apply perturbation to depth channels
        if images.shape[1] == 6:
            perturbed_images = images_depth + self.depth_perturbation * ~foreground_mask

            perturbed_images = torch.clamp(
                perturbed_images,
                min=self.estimator.clip_values[0],
                max=self.estimator.clip_values[1],
            )

            if not self.estimator.channels_first:
                perturbed_images = torch.permute(perturbed_images, (0, 2, 3, 1))

            return torch.cat([patched_images, perturbed_images], dim=-1)

        if return_mask:
            return patched_images, patched_mask
        return patched_images

    def generate(self, x, y=None, y_patch_metadata=None):
        """
        param x: Sample images. For single-modality, shape=(NHW3). For multimodality, shape=(NHW6)
        param y: [Optional] Sample labels. List of dictionaries,
            ith dictionary contains bounding boxes, class labels, and class scores
        param y_patch_metadata: Patch metadata. List of N dictionaries, ith dictionary contains patch metadata for x[i]
        """

        if x.shape[0] > 1:
            log.info("To perform per-example patch attack, batch size must be 1")
        assert x.shape[-1] in [3, 6], "x must have either 3 or 6 color channels"

        num_imgs = x.shape[0]
        attacked_images = []

        for i in range(num_imgs):
            # Adversarial patch attack, when used for object detection, requires ground truth
            y_gt = dict()
            y_gt["labels"] = y[i]["labels"]
            non_patch_idx = np.where(
                y_gt["labels"] != 4
            )  # exclude the patch class, which doesn't exist in the training data
            y_gt["boxes"] = y[i]["boxes"][non_patch_idx]
            y_gt["labels"] = y_gt["labels"][non_patch_idx]
            y_gt["scores"] = np.ones(len(y_gt["labels"]), dtype=np.float32)

            gs_coords = y_patch_metadata[i]["gs_coords"]  # patch coordinates
            self.gs_coords = [gs_coords]
            patch_width = int(np.max(gs_coords[:, 0]) - np.min(gs_coords[:, 0]))
            patch_height = int(np.max(gs_coords[:, 1]) - np.min(gs_coords[:, 1]))
            self.patch_shape = (
                3,
                patch_height,
                patch_width,
            )

            # Use this mask to embed patch into the background in the event of occlusion
            self.binarized_patch_mask = y_patch_metadata[i]["mask"]

            # Eval7 contains a mixture of patch locations.
            # Patches that lie flat on the sidewalk or street are constrained to 0.03m depth perturbation, and they are best used to create disappearance errors.
            # Patches located elsewhere (i.e., that do not impede pedestrian/vehicle motion) are constrained to 3m depth perturbation, and they are best used to create hallucinations.
            # Therefore, the depth perturbation bound for each patch is input-dependent.
            if x.shape[-1] == 6:
                if "max_depth_perturb_meters" in y_patch_metadata[i].keys():
                    self.depth_delta_meters = y_patch_metadata[i][
                        "max_depth_perturb_meters"
                    ]
                    log.info(
                        'This dataset contains input-dependent depth perturbation bounds, and the user-defined "depth_delta_meters" has been reset to {} meters'.format(
                            y_patch_metadata[i]["max_depth_perturb_meters"]
                        )
                    )

            # self._patch needs to be re-initialized with the correct shape
            if self.patch_base_image is not None:
                # TODO: Encode an initial image
                assert False, "Not yet supported!"
            else:
                assert len(self.patch_init) == 2

                # Set RNG for exact reproducibility
                self.rng_state.manual_seed(self.seed)

                # Configure diffusion pipeline if necessary for handling text prompts
                if self.attack_method == "end-to-end":
                    self.prepare_diffusion(self.prompt)
                    latent_init = self.pipe.prepare_latents(1, self.pipe.unet.config.in_channels,
                                                            self.diff_height, self.diff_width,
                                                            self.pipe.unet.dtype, self.pipe.device, self.rng_state, None)
                    # Initialize latents
                    self._latents = latent_init.detach().cpu().numpy()

                # Initialize patch
                patch_init = np.random.randint(*self.patch_init, size=self.patch_shape) / 255.
                patch_mask = np.ones_like(patch_init)

            self._patch = torch.tensor(
                patch_init, requires_grad=True, device=self.estimator.device, dtype=torch.float32
            )
            self.patch_mask = torch.Tensor(patch_mask).to(self.estimator.device)

            # initialize depth variables
            if x.shape[-1] == 6:
                # check if depth image is log-depth
                if np.all(x[i, :, :, 3] == x[i, :, :, 4]) and np.all(
                    x[i, :, :, 3] == x[i, :, :, 5]
                ):
                    self.depth_type = "log"
                    depth_linear = log_to_linear(x[i, :, :, 3:])
                    max_depth = linear_to_log(depth_linear + self.depth_delta_meters)
                    min_depth = linear_to_log(depth_linear - self.depth_delta_meters)
                    max_depth = np.transpose(np.minimum(1.0, max_depth), (2, 0, 1))
                    min_depth = np.transpose(np.maximum(0.0, min_depth), (2, 0, 1))
                else:
                    self.depth_type = "linear"
                    depth_linear = rgb_depth_to_linear(
                        x[i, :, :, 3], x[i, :, :, 4], x[i, :, :, 5]
                    )
                    max_depth = depth_linear + self.depth_delta_meters
                    min_depth = depth_linear - self.depth_delta_meters
                    max_depth = np.minimum(1000.0, max_depth)
                    min_depth = np.maximum(0.0, min_depth)

                self.max_depth = torch.tensor(
                    np.expand_dims(max_depth, axis=0),
                    dtype=torch.float32,
                    device=self.estimator.device,
                )
                self.min_depth = torch.tensor(
                    np.expand_dims(min_depth, axis=0),
                    dtype=torch.float32,
                    device=self.estimator.device,
                )
                self.depth_perturbation = torch.zeros(
                    1,
                    3,
                    x.shape[1],
                    x.shape[2],
                    requires_grad=True,
                    device=self.estimator.device,
                )

            if self.attack_method == "inpaint":
                # In-painting attack uses max_iter reverse diffusion steps
                self.pipe.scheduler.set_timesteps(self.max_iter)
                self.timesteps = iter(self.pipe.scheduler.timesteps)
            self.diffusion_output = None

            patch, _ = super().generate(np.expand_dims(x[i], axis=0), y=[y_gt])

            # Patch image
            x_tensor = torch.tensor(np.expand_dims(x[i], axis=0)).to(
                self.estimator.device
            )
            patched_image = (
                self._random_overlay(
                    images=x_tensor, patch=self._patch, scale=None, mask=None
                )
                .detach()
                .cpu()
                .numpy()
            )
            patched_image = np.squeeze(patched_image, axis=0)

            # Embed patch into background
            patched_image[np.all(self.binarized_patch_mask == 0, axis=-1)] = x[i][
                np.all(self.binarized_patch_mask == 0, axis=-1)
            ]

            patched_image = np.clip(
                patched_image,
                self.estimator.clip_values[0],
                self.estimator.clip_values[1],
            )

            attacked_images.append(patched_image)

            # Save patches to file
            save_image = Image.fromarray(np.uint8(torch.permute(self._patch, (1, 2, 0)).cpu().detach().numpy() * 255.))
            save_image.save("diffusion_%s_patch_sample%d.png" % (self.attack_method, self.sample_index), "PNG")
            self.sample_index = self.sample_index + 1

        return np.array(attacked_images)
