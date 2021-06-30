#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import pytest
import numpy as np
from PIL import Image
from skimage.transform import rescale
from skimage.data import stereo_motorcycle
from oscar.defences.preprocessor.detectron2 import Detectron2PreprocessorPyTorch
import torch

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", params=[True, False])
def detectron2_model(request):
    logger.warning("Creating Detectron2 Model (you should only see this message once!)")
    dt2 = Detectron2PreprocessorPyTorch(config_path="detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
                                        weights_path="detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl",
                                        score_thresh=0.5,
                                        resize=request.param)

    if torch.cuda.is_available():
        dt2 = dt2.cuda()

    return dt2


@pytest.fixture
def mask_color():
    return np.array([115/255, 108/255, 99/255], dtype=np.float32)


@pytest.fixture
def real_images_112x112(mask_color):
    img1, img2, _ = stereo_motorcycle()

    # Resize from (500, 741, 3) to (112, 166, 3)
    size = 112
    transform_func = lambda x: rescale(x, size / 500, multichannel=True, anti_aliasing=True)
    img1, img2 = transform_func(img1), transform_func(img2)

    # Crop 112x112 from 166x112.
    img1 = img1[:size, :size, :]
    img2 = img2[-size:, -size:, :]

    # Add image with no content
    img3 = np.zeros_like(img2) + 0.25

    imgs = np.stack([img1, img2, img3])
    imgs = np.expand_dims(imgs, 0)
    imgs = imgs.astype(np.float32)

    # Make sure no pixels in imgs match mask_color
    imgs[imgs == mask_color] += 1/255
    imgs = np.clip(imgs, 0, 1)

    return imgs

@pytest.fixture
def real_images_112x112_torch(real_images_112x112):
    x = torch.from_numpy(real_images_112x112)
    if torch.cuda.is_available():
        x = x.cuda()
    return x
