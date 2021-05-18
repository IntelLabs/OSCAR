#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
import numpy as np
import kornia
import pyredner

def create_perturbable_object(object_image, mask, initial_value=0, sign_grad=True):
    """
        Create perturbable object using mask to define perturbable area. Will also initialize
        perturbable area to initial_value. Gradients of this perturbable object will be masked
        and magnitudeless, if sign_grad=True.

        Parameters:
            object_image (torch.Tensor): Object to perturb.
            mask (torch.Tensor): Mask of perturbable area of object.
            initial_value (int): Initial value to set perturbable area. (default: 0)
            sign_grad (bool): Whether to remove gradient magnitude for the gradient of the perturbable object. (default: True)

        Returns:
            torch.Tensor[1, C, H, W]: Perturbable object that requires gradients.
    """
    object_rgb = object_image[:, :3, :, :]
    object_alpha = object_image[:, 3:, :, :]

    perturbed_object = object_image.clone()
    perturbed_object[:, :3, :, :] = object_rgb*(1 - mask) + initial_value*mask
    perturbed_object[:, 3:, :, :] = object_alpha

    perturbed_object.requires_grad_(True)
    perturbed_object.register_hook(lambda grad: grad*mask)
    if sign_grad:
        perturbed_object.register_hook(lambda grad: grad.sign())

    return perturbed_object

def _perspective_transform(resolution, x, y, z, pitch, yaw, roll):
    pitch = kornia.geometry.conversions.deg2rad(pitch)[:, None, None]
    yaw = kornia.geometry.conversions.deg2rad(yaw)[:, None, None]
    roll = kornia.geometry.conversions.deg2rad(roll)[:, None, None]

    x = x[:, None, None]
    y = y[:, None, None]
    z = z[:, None, None]

    h = resolution[:, 0:1, None]
    w = resolution[:, 1:2, None]

    zero = torch.zeros_like(pitch)
    one = torch.ones_like(pitch)

    # Project to 3d
    P_3d = torch.cat([torch.cat([ one, zero, -w/2], 2),
                      torch.cat([zero,  one, -h/2], 2),
                      torch.cat([zero, zero,  one], 2),
                      torch.cat([zero, zero,  one], 2)],
                     1)

    # Rotate about the X-axis
    cos = torch.cos(pitch)
    sin = torch.sin(pitch)
    R_x = torch.cat([torch.cat([ one, zero, zero, zero], 2),
                     torch.cat([zero,  cos,  sin, zero], 2),
                     torch.cat([zero, -sin,  cos, zero], 2),
                     torch.cat([zero, zero, zero,  one], 2)],
                    1)

    # Rotate about the Y-axis
    cos = torch.cos(yaw)
    sin = torch.sin(yaw)
    R_y = torch.cat([torch.cat([ cos, zero,  sin, zero], 2),
                     torch.cat([zero,  one, zero, zero], 2),
                     torch.cat([-sin, zero,  cos, zero], 2),
                     torch.cat([zero, zero, zero,  one], 2)],
                    1)

    # Rotate about the Z-axis
    cos = torch.cos(roll)
    sin = torch.sin(roll)
    R_z = torch.cat([torch.cat([ cos, -sin, zero, zero], 2),
                     torch.cat([ sin,  cos, zero, zero], 2),
                     torch.cat([zero, zero,  one, zero], 2),
                     torch.cat([zero, zero, zero,  one], 2)],
                    1)

    # Project back to 2d
    P_2d = torch.cat([torch.cat([   z, zero, w/2, zero], 2),
                      torch.cat([zero,    z, h/2, zero], 2),
                      torch.cat([zero, zero, one, zero], 2)],
                     1)

    # Translate
    T = torch.cat([torch.cat([ one, zero, x*w/2], 2),
                   torch.cat([zero,  one, y*h/2], 2),
                   torch.cat([zero, zero,   one], 2)],
                  1)

    # Batch multiply transformations to get final transformation
    transform = torch.matmul(T, torch.matmul(P_2d, torch.matmul(R_z, torch.matmul(R_y, torch.matmul(R_x, P_3d)))))

    return transform

def generate_transforms(count, resolution,
                        yaw=None, yaw_range=(0., 0.), yaw_bins=100, yaw_fn=np.linspace,
                        pitch=None, pitch_range=(0., 0.), pitch_bins=100, pitch_fn=np.linspace,
                        roll=None, roll_range=(0., 0.), roll_bins=100, roll_fn=np.linspace,
                        x=None, x_range=(0., 0.), x_bins=100, x_fn=np.linspace,
                        y=None, y_range=(0., 0.), y_bins=100, y_fn=np.linspace,
                        z=None, z_range=(1., 1.), z_bins=100, z_fn=np.linspace,
                        device=torch.device('cpu')):
    """
        Generate perspective transformation matrices by sampling parameters.

        Parameters:
            count (int): Number of perspective transforms to sample.
            resolution (torch.Tensor[2]): Height and width of the output.
            for param in [yaw, pitch, roll, x, y, s]:
                params (list[float]): List of params to sample from. (default: None)
                param_range (tuple(int, int)): Range of params to sample from. (default: (0, 0))
                param_bins (int): Number of bins to discretize param_range using param_fn. (default: 100)
                param_fn (func): Function to discretize param_range using param_bins. (default: np.linspace)

        Returns:
            list[torch.Tensor[N, 3, 3]]: List of sampled perspective transforms.
    """
    # Discretize ranges
    if yaw is None:
        yaw = yaw_fn(*yaw_range, num=yaw_bins)
    if pitch is None:
        pitch = pitch_fn(*pitch_range, num=pitch_bins)
    if roll is None:
        roll = roll_fn(*roll_range, num=roll_bins)
    if x is None:
        x = x_fn(*x_range, num=x_bins)
    if y is None:
        y = y_fn(*y_range, num=y_bins)
    if z is None:
        z = z_fn(*z_range, num=z_bins)

    # TODO: Convert to torch!
    # Make N=count random choices for each parameter unless we have a seed
    yaw = np.random.choice(yaw, count)
    pitch = np.random.choice(pitch, count)
    roll = np.random.choice(roll, count)
    x = np.random.choice(x, count)
    y = np.random.choice(y, count)
    z = np.random.choice(z, count)

    resolution = np.repeat(np.expand_dims(resolution, axis=0), count, axis=0)

    resolution = torch.as_tensor(resolution).to(dtype=torch.float32, device=device)
    x = torch.as_tensor(x).to(dtype=torch.float32, device=device)
    y = torch.as_tensor(y).to(dtype=torch.float32, device=device)
    z = torch.as_tensor(z).to(dtype=torch.float32, device=device)
    yaw = torch.as_tensor(yaw).to(dtype=torch.float32, device=device)
    pitch = torch.as_tensor(pitch).to(dtype=torch.float32, device=device)
    roll = torch.as_tensor(roll).to(dtype=torch.float32, device=device)

    transforms = _perspective_transform(resolution=resolution, x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)

    return transforms

def cameras_to_transforms(cameras):
    transforms = []
    for camera in cameras:
        h, w = camera.resolution
        x, y = camera.intrinsic_mat[0:2, 2]
        z = camera.position[2]

        P_2d = torch.tensor([[1/z,   0, w/2, 0],
                             [  0, 1/z, h/2, 0],
                             [  0,   0,    1, 0]]).float()

        # NOTE: transpose!
        R_xyz = pyredner.gen_look_at_matrix(pos=camera.position, look=camera.look_at, up=camera.up).T

        P_3d = torch.tensor([[1, 0, -w/2],
                             [0, 1, -h/2],
                             [0, 0,  -1],
                             [0, 0,   0]]).float()

        T = torch.tensor([[1, 0, x*w/2],
                          [0, 1, y*h/2],
                          [0, 0,     1]]).float()

        transform = T @ P_2d @ R_xyz @ P_3d

        transforms.append(transform)
    transforms = torch.stack(transforms)

    return transforms

def render(list_of_object_image, transforms, resolution):
    """
        Render each object warped under each transform onto canvas of size resolution.

        Parameters:
            list_of_object_image (list[torch.Tensor[N, C, H, W]): List of objects to render.
            transforms (list[torch.Tensor[M, 3, 3]): List of transforms.
            resolution (torch.Tensor[2]): Resolution in (H2, W2) to render onto

        Returns:
            list[torch.Tensor[N*M, C, H2, W2]]: List of rendered objects
    """
    renders = []

    for object_image in list_of_object_image:
        object_image = object_image[None]

        # Quantized object to 8-bits
        object_image = torch.fake_quantize_per_tensor_affine(object_image, scale=1/255, zero_point=0, quant_min=0, quant_max=255)

        # Pad image to resolution
        object_image = kornia.geometry.transform.center_crop(object_image, resolution)

        for transform in transforms:
            transform = transform[None]

            render = kornia.geometry.transform.warp_perspective(object_image, transform, dsize=resolution)

            renders.append(render[0])

    return renders

