#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
import numpy as np
import pyredner
import kornia

def load_mesh(path, normalize=True):
    """
        Loads meshes from OBJ file and normalizes then unit cube, if specified.
        This will place the vertices and indices on the default PyRedner device
        and everything else on the 'CPU' for performance reasons.

        Parameters:
            path (str): Path to OBJ file.
            normalize (bool): Whether to normalize mesh to unit cube.

        Returns:
            dict[str => pyredner.TriangleMesh]: dictionary of material names and meshes in OBJ file.
    """
    print('Loading', path)

    # Don't load anything onto the GPU just yet
    use_gpu = pyredner.get_use_gpu()
    pyredner.set_use_gpu(False)
    _, mesh_list, _ = pyredner.load_obj(path, return_objects=False)
    pyredner.set_use_gpu(use_gpu)

    # Normalize mesh to [-1, 1] (x, y, z)-cube like Lucid does
    if normalize:
        vertices_min = torch.stack([mesh.vertices.max(0)[0] for _, mesh in mesh_list]).max(0)[0]
        vertices_max = torch.stack([mesh.vertices.min(0)[0] for _, mesh in mesh_list]).min(0)[0]
        mean_vertices_range = (vertices_max + vertices_min) / 2.
        centered_vertices_absmax = torch.stack([vertices_min - mean_vertices_range,
                                                vertices_max - mean_vertices_range]).abs().max()

        for _, mesh in mesh_list:
            mesh.vertices -= mean_vertices_range
            mesh.vertices /= centered_vertices_absmax

    # Load vertices and indices onto pyredner device
    for _, mesh in mesh_list:
        mesh.vertices = mesh.vertices.to(pyredner.get_device())
        mesh.indices = mesh.indices.to(pyredner.get_device())

    return dict(mesh_list)

def load_meshes(*paths, normalize=True):
    """
        See `load_mesh`.
    """
    # Would be nice to use torch.multiprocessing.Pool but it doesn't work
    return [load_mesh(path, normalize=normalize) for path in paths]

def create_perturbable_texels(texels_map, mask_map, initial_value=0., sign_grad=True):
    """
        Creates perturbable texels using mask to define perturbable area. Will also initialize
        perturbable area to some value. If no mask is specified for texel, we treat the entire
        texel as not perturbable. Gradients of these perturbable texels will be masked and
        magnitudeless, if sign_grad=True.

        Parameters:
            texels_map (dict[str -> torch.Tensor[N, C, H, W]]): Dictionaries of texels.
            mask_map (dict[str -> torch.Tensor[M, 1, H, W]]): Dictionary of masks.
            initial_value (float or torch.Tensor): Initial value to set perturbable area, must be broadcastable to mask.
            sign_grad (bool): Whether to remove gradient magnitude for the gradient of perturable texel.

        Returns:
            dict(str -> torch.Tensor[N, C, H, W]): List of perturbable texels that require gradients.
    """
    perturbed_texels_map = {}

    for name, texels in texels_map.items():
        if name in mask_map:
            texels_rgb = texels[:3, :, :]
            mask = mask_map[name]

            perturbed_texels = texels_rgb*(1 - mask) + initial_value*mask
            perturbed_texels.requires_grad_(True)
            perturbed_texels.register_hook(lambda grad: grad*mask)
            if sign_grad:
                perturbed_texels.register_hook(lambda grad: grad.sign())
        else:
            perturbed_texels = texels

        perturbed_texels_map[name] = perturbed_texels

    return perturbed_texels_map

def create_textures(texels_map, mipmap=True):
    """
        Create textures from texels where each texel is discretized to 8-bits.

        Parameters:
            texels_map (dict[str -> torch.Tensor[N, C, H, W]]): Dictionary of texels.
            mipmap (bool): where to apply mipmapping to the texture. (default: True)

        Returns:
            dict(str -> pyredner.Texture): List of textures.
    """
    # Create texture for each texels
    texture_map = {}
    for name, texels in texels_map.items():
        # Redner wants texels in HWC format where C is RGB
        texels = texels[:3, :, :].permute(1, 2, 0)

        # Quantized texels to 8-bits
        texels = torch.fake_quantize_per_tensor_affine(texels, scale=1/255, zero_point=0, quant_min=0, quant_max=255)

        # Create texture and turn off mipmapping if requested
        texture = pyredner.Texture(texels)
        if not mipmap:
            texture.mipmap = [texels]

        texture_map[name] = texture

    return texture_map

def create_objects(texture_map, list_of_meshes):
    """
        Create an object for each mesh. For each object, we set the diffuse reflectance as the corresponding
        texture.

        Parameters:
            texture_map (dict[str -> pyreder.Texture): List of textures to apply to diffuse reflectance.
            list_of_meshes (list[dict[str -> pyredner.TriangleMesh]]): List of meshes to create objects from.

        Returns:
            list[list[pyreder.Object]]: List of list of objects.
    """
    # Create object where the diffuse reflectance is the desired texture
    list_of_perturbed_objects = []

    for mesh_map in list_of_meshes:
        perturbed_objects = []
        for name, mesh in mesh_map.items():
            texture = texture_map[name]

            perturbed_material = pyredner.Material(diffuse_reflectance=texture)

            perturbed_object = pyredner.Object(vertices=mesh.vertices,
                                             indices=mesh.indices,
                                             uvs=mesh.uvs,
                                             uv_indices=mesh.uv_indices,
                                             normals=mesh.normals,
                                             normal_indices=mesh.normal_indices,
                                             material=perturbed_material)
            perturbed_objects.append(perturbed_object)

        list_of_perturbed_objects.append(perturbed_objects)

    return list_of_perturbed_objects


def generate_cameras(num_cameras, resolution, fov=90,
                     origin=(0, 0, 0),
                     yaws=None, yaw_range=(0, 0), yaw_bins=1, yaw_fn=np.linspace,
                     pitchs=None, pitch_range=(0, 0), pitch_bins=1, pitch_fn=np.linspace,
                     rolls=None, roll_range=(0, 0), roll_bins=1, roll_fn=np.linspace,
                     xs=None, x_range=(0, 0), x_bins=1, x_fn=np.linspace,
                     ys=None, y_range=(0, 0), y_bins=1, y_fn=np.linspace,
                     zs=None, z_range=(1, 1), z_bins=1, z_fn=np.linspace):
    """
        Generate cameras by sampling specified cameras.

        Parameters:
            num_cameras (int): Number of camers to sample.
            resolution (torch.Tensor[2]): Resolution of the cameras.
            fov (int): Field of view of the camera. (default: 90)
            for param in [yaw, pitch, roll, x, y, s]:
                params (list[float]): List of params to sample from. (default: None)
                param_range (tuple(int, int)): Range of params to sample from. (default: (0, 0))
                param_bins (int): Number of bins to discretize param_range using param_fn. (default: 100)
                param_fn (func): Function to discretize param_range using param_bins. (default: np.linspace)

        Returns:
            list[pyredner.Camera]: List of cameras sampled from parameters.

    """
    # Discretize ranges
    if yaws is None:
        yaws = yaw_fn(*yaw_range, num=yaw_bins)
    if pitchs is None:
        pitchs = pitch_fn(*pitch_range, num=pitch_bins)
    if rolls is None:
        rolls = roll_fn(*roll_range, num=roll_bins)
    if xs is None:
        xs = x_fn(*x_range, num=x_bins)
    if ys is None:
        ys = y_fn(*y_range, num=y_bins)
    if zs is None:
        zs = z_fn(*z_range, num=z_bins)

    # Convert from degrees to radians
    yaws = yaws * np.pi/180
    pitchs = pitchs * np.pi/180
    rolls = rolls * np.pi/180

    fov = torch.tensor([fov], dtype=torch.float32)

    cameras = []

    for _ in range(num_cameras):
        # Randomly sample camera parameters
        yaw = np.random.choice(yaws)
        pitch = np.random.choice(pitchs)
        roll = np.random.choice(rolls)
        x = np.random.choice(xs)
        y = np.random.choice(ys)
        z = np.random.choice(zs)

        # Create camera
        look_at = torch.tensor(origin, dtype=torch.float32)
        position = torch.tensor([1/z*np.cos(pitch)*np.sin(yaw),
                                 1/z*np.sin(pitch),
                                 1/z*np.cos(pitch)*np.cos(yaw)], dtype=torch.float32)
        up = torch.tensor([np.sin(roll), np.cos(roll), 0.], dtype=torch.float32)
        intrinsic_mat = torch.eye(3)
        intrinsic_mat[0, 2] = x
        intrinsic_mat[1, 2] = y
        #intrinsic_mat[2, 2] = 1/z

        camera = pyredner.Camera(position=position, look_at=look_at, up=up, resolution=resolution, fov=fov, intrinsic_mat=intrinsic_mat)

        cameras.append(camera)

    return cameras

def render(list_of_objects, cameras):
    """
        Renders each list of objects using camera and default lights.

        Parameters:
            list_of_objects (list[list[pyredner.Object]]): List of list of objects to render in each scene.
            cameras (list[pyredner.Camera]): List of cameras.

        Returns:
            list[torch.Tensor[N, C, H, W]]: List of renders with alpha channel where each render is the product of the
            objects and cameras.
    """
    renders = []

    for objects in list_of_objects:
        for camera in cameras:
            scene = pyredner.Scene(camera=camera, objects=objects)

            # TODO: Parameterize lights
            lights = [pyredner.SpotLight(position=scene.camera.position.to(pyredner.get_device()),
                                         spot_direction=torch.tensor((0.0, 0.0, 1.0), device=pyredner.get_device()),
                                         spot_exponent=torch.tensor(0.0, device=pyredner.get_device()),
                                         intensity=torch.tensor((4.0, 4.0, 4.0), device=pyredner.get_device()))]

            #render = pyredner.render_albedo(scene, alpha=True)
            render = pyredner.render_deferred(scene, lights=lights, alpha=True)

            del scene, lights

            # Convert render to channels first
            render = render.permute(2, 0, 1)

            renders.append(render)

    return renders

