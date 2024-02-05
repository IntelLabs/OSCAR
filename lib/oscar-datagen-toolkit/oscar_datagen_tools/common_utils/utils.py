#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Tuple

import coloredlogs
import hydra
import numpy as np
from numpy import random
from omegaconf import DictConfig
from scipy import ndimage

__all__ = [
    "format_patch_name",
    "format_sensor_name",
    "generate_image_filename",
    "get_image_id",
    "get_sensor_type",
    "verify_image_filename",
    "connect_to_carla_server",
    "filename_decode",
    "load_sensors_static_metadata",
    "load_dynamic_metadata",
    "build_projection_matrix",
    "get_image_point",
    "get_world_point",
    "get_patch_in_camera",
]

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

# regex patterns
filename_pattern = re.compile(
    r"route[0-9]+_[0-9]*\.[0-9]+z_[0-9]*\.[0-9]+deg\.[0-9]+\.png", re.IGNORECASE
)  # route<CAMERA_ID>_<Z_LOCATION>z_<PITCH>deg.0000000n.png

# camera constants
CAMERA_ID = -1

# unformatted sensor directory name: camera.sensor.<TYPE>.<ID>
SENSOR_TYPE = -2
SENSOR_ID = -1


def format_patch_name(patch_id: int) -> str:
    """Adjust the format of the patch name, where the expected format is: static.prop.<ID>

    Parameters
    ----------
    patch_id : id
        Patch's unique identifier.
    Returns
    -------
    patch_name: str
        Patch name.
    """
    return f"static.prop.{patch_id}"


def format_sensor_name(name: str) -> str:
    """Adjust the format of the sensor name, where the expected format is: <TYPE>

    Parameters
    ----------
    name : id
        Sensor's name
    Returns
    -------
    sensor_name: str
        Sensor's formatted name.
    """
    sensor_type = name.split(".")[SENSOR_TYPE]
    return sensor_type


def generate_image_filename(sensor_id: int, z_location: float, pitch: float, index: int) -> str:
    """Creates an image filename with the format: 0000000n.png.

    Parameters
    ----------
    sensor_id : int
        Camera ID.
    z_location: float
        Camera's elevation
    pitch: float
        Camera's pitch angle
    index: int
        Frame index
    Returns
    -------
    image_name: str
        Image's formatted name.
    """
    return f"route{sensor_id}_{z_location:.2f}z_{pitch:.2f}deg.{index:08}.png"


def get_image_id(path: Path) -> int:
    """Gets the image ID from it's filename.

    Parameters
    ----------
    path : Path
        Path to image's file.
    Returns
    -------
    image_id: str
        Image's identifier.
    """

    # The expected filename follows the format "route<CAMERA_ID>_<Z_LOCATION>z_<PITCH>deg.0000000n.png"
    return path.stem


def get_sensor_type(path: Path) -> str:
    """Gets the sensor' type name.

    Parameters
    ----------
    path : Path
        Path to sensor's directory.
    Returns
    -------
    sensor_type: str
        Sensor's type name.
    """
    return path.name


def verify_image_filename(filename: str) -> bool:
    """Verify if the image name follows the
    format: route<CAMERA_ID>_<Z_LOCATION>z_<PITCH>deg.0000000n.png.

    Parameters
    ----------
    name : str
        Image's name.
    Returns
    -------
    result: bool
        True if the name is correct, False otherwise.
    """
    if not filename_pattern.match(filename):
        logger.warning(
            f"Filename {filename} does not follow the format: route<CAMERA_ID>_<Z_LOCATION>z_<PITCH>deg.0000000n.png"
        )
        return False

    return True


def connect_to_carla_server(cfg: DictConfig):
    """Connect to CARLA server.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration structure.
    Returns
    -------
    result: bool
        True if the simulation starts successfully, False otherwise.
    """
    # TODO Find a way to avoid this delayed import. This is needed in this
    # moment due to a circular import error.
    from oscar_datagen_tools.simulation import CollectorController

    # First instantiation of the Context Singleton object
    context = hydra.utils.instantiate(cfg.context, reinit=True)

    # set random seed before instantiate the spawn actors
    logger.info(f"Random seed: {context.client_params.seed}")
    random.seed(context.client_params.seed)

    # Create controller so we connect to the CARLA server and have a context
    output_dir = Path(cfg.paths.data_dir) if cfg.paths.data_dir else None
    controller = CollectorController(
        max_frames=cfg.max_frames,
        output_dir=output_dir,
    )

    if not controller.setup_simulation():
        logger.error("Error while setting up the simulator.")
        return None

    # Convert all to primitive containers when instantiating the actors, specially
    # for the Blueprint's Sensor data type where the Camera's init method verify
    # this instance type.
    actors = []
    for actor_config in cfg.spawn_actors:
        # FIXME: Pass context here instead of using singleton?
        actor = hydra.utils.instantiate(actor_config, _convert_="all")
        if not isinstance(actor, Iterable):
            actor = [actor]
        actors.extend(actor)

    controller.actors = actors

    return controller


def filename_decode(name: str) -> Tuple[str, str]:
    """Decode the image filename to extract the camera ID and the frame number. The expected
    filename format is the following: route<CAMERA_ID>_<ELEVATION>z_<ANGLE>deg.0000000n.
    Parameters
    ----------
    name : str
        Image filename.
    Returns
    -------
    (sensor_id, frame_num) : Tuple[str, str]
    """
    name_substring = name.split("_")

    # get the camera id
    assert "route" in name_substring[0]
    sensor_id = name_substring[0]
    sensor_id = sensor_id.replace("route", "")

    # get frame number
    frame = name_substring[-1].split(".")
    frame = frame[-2]

    return sensor_id, frame


def load_sensors_static_metadata(path: Path, sensor_id: str) -> dict:
    """Load the static metadata for an specific camera and sensor type.
    Parameters
    ----------
    path : Path
        Path to the static metadata.
    sensor_id:
        Sensor identifier.
    Returns
    -------
    sensor_metadata : dict
    """
    sensor = {}
    with open(path) as metadata_file:
        data = json.load(metadata_file)

        if "sensors" in data:
            sensor = data["sensors"][sensor_id]

    return sensor


def load_dynamic_metadata(path: Path, sensor_id: str) -> dict:
    """Load the dynamic metadata for an specific camera and sensor type.
    Parameters
    ----------
    path : Path
        Path to the static metadata.
    sensor_id:
        Sensor identifier.
    Returns
    -------
    metadata : dict
    """
    metadata = {}

    with open(path) as metadata_file:
        data = json.load(metadata_file)
        if "sensors" in data:
            metadata["sensor"] = data["sensors"][sensor_id]

        if "patches" in data:
            metadata["patches"] = data["patches"]

    return metadata


def build_projection_matrix(w, h, fov, dim=3):
    """Code taken from: https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/"""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(dim)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    """Code taken from: https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/"""
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc[0], loc[1], loc[2], 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


def get_world_point(image_point, camera_transform, camera_width, camera_height, camera_fov):
    # add homogeneous coordinates
    image_point = np.pad(image_point, ((0, 0), (0, 2)), "constant", constant_values=1)

    # intrinsic matrix
    K = build_projection_matrix(camera_width, camera_height, camera_fov, dim=4)
    K_inv = np.linalg.inv(K)

    # permutation matrix (x, y, z, 1) -> (y, -z, x, 1)
    P = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    P_inv = np.linalg.inv(P)

    # extrinsic matrix
    C2W = np.array(camera_transform.get_matrix(), dtype=np.float32)[:3, :]
    origin = np.array(
        [[camera_transform.location.x, camera_transform.location.y, camera_transform.location.z]],
        dtype=np.float32,
    )

    # project 2d coordinates to uncentered 3d world coordinates
    p3d = K_inv @ image_point.T  # project to camera
    p3d = P_inv @ p3d  # convert to ue4 coordinates
    p3d = C2W @ p3d  # project to world

    # center and normalize
    vec = p3d - origin.T
    vec = vec * p3d[2:3] / vec[2:3]

    return (p3d - vec).T[:, :2]  # ignore z


def get_patch_in_camera(sensor_static_metadata: dict, dynamic_metadata: dict):
    # calculate camera matrix
    image_width = sensor_static_metadata["image_size_x"]
    image_height = sensor_static_metadata["image_size_y"]
    fov = sensor_static_metadata["fov"]
    k = build_projection_matrix(image_width, image_height, fov)

    # map from 2D to 3D
    w2c = np.array(dynamic_metadata["sensor"]["actorsnapshot_transform_get_inverse_matrix"])
    corners = []
    patch_found = None

    assert (
        len(dynamic_metadata["patches"]) <= 1
    ), "This implementation does not handle multiple patches"
    for patch_id in dynamic_metadata["patches"]:
        patch_found = patch_id
        patch_corners = dynamic_metadata["patches"][patch_id]["corners"]
        for corner_key in patch_corners.keys():
            location = patch_corners[corner_key]["actor_location"]
            # point -> [x, y]
            point = get_image_point(location, k, w2c)
            corner = [point[0], point[1]]
            corners.append(corner)

    return (patch_found, corners)
