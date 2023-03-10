#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from pathlib import Path
from typing import Optional

import cv2
import fire
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

GLFW_KEY_R = 82


def visualize(
    rgb: Path,
    depth: Path,
    output: Optional[Path] = None,
    width: int = 1280,
    height: int = 960,
    fps: float = 30.0,
    max_frames: int = 30 * 5,
    point_size: float = 2.0,
    radius: float = 0.5,
):
    # Delay import because Armory import is slow
    from armory.art_experimental.attacks.carla_obj_det_utils import rgb_depth_to_linear

    color_raw = read_image(rgb)
    depth_raw = read_image(depth)

    assert np.all(color_raw.get_max_bound() == depth_raw.get_max_bound())

    # Create camera from parameters
    camera_width, camera_height = color_raw.get_max_bound()
    aspect_ratio = camera_width / camera_height
    camera_fx = camera_width / 2
    camera_fy = camera_width / 2
    camera_cx = camera_width / 2 - 0.5
    camera_cy = camera_height / 2 - 0.5

    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(int(camera_width), int(camera_height), camera_fx, camera_fy, camera_cx, camera_cy)
    cam.extrinsic = np.eye(4)

    # Convert RGB depth to linear depth
    depth_linear = np.asarray(depth_raw).astype(np.float32) / 255
    depth_linear = np.split(depth_linear, 3, axis=-1)
    depth_linear = rgb_depth_to_linear(*depth_linear).astype(np.float32)
    depth_linear = o3d.geometry.Image(depth_linear)

    # Create point cloud from RGBD and cleanup point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_linear, depth_scale=1.0, depth_trunc=1000.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam.intrinsic)
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)

    # Create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=width, height=height)
    vis.add_geometry(pcd)

    # Make background color black and render points small
    opt = vis.get_render_option()
    if opt is None:
        raise Exception(
            "Could not create visualization, probably because no DISPLAY is present.\n"
            "You can also try using the Open3D with headless support:\n"
            "http://www.open3d.org/docs/latest/tutorial/Advanced/headless_rendering.html"
        )

    opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.point_size = point_size

    # Set initial view and register callback for R key to reset view
    ctr = vis.get_view_control()

    def reset_view_point(vis):
        ctr.convert_from_pinhole_camera_parameters(cam)

    reset_view_point(vis)

    vis.register_key_callback(GLFW_KEY_R, reset_view_point)

    # If no output specified just display the visualization
    if not output:
        return vis.run()

    # Move camera and save each frame to a video
    video = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for theta in tqdm(np.linspace(0.0, 2 * np.pi, max_frames)):
        # Rotate camera in a circle biased by aspect ratio
        ctr.set_front([radius * np.sin(theta), radius / aspect_ratio * (1 - np.cos(theta)), -1])

        # Capture frame from visualization
        frame = vis.capture_screen_float_buffer(do_render=True)

        # Convert to 8-bit BGR
        frame = (np.array(frame) * 255).round().astype(np.uint8)
        frame = frame[:, :, ::-1]  # RGB -> BGR

        video.write(frame)


def read_image(path: Path):
    image = Image.open(path)
    image = image.convert("RGB")
    image = np.array(image)
    image = o3d.geometry.Image(image)

    return image


if __name__ == "__main__":
    fire.Fire(visualize)
