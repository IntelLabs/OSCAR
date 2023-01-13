#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
import torchvision
import math

from typing import Optional, List, Dict, Tuple

from torch import Tensor
from torch.jit.annotations import List
import torch.nn.functional as F
from torchvision.models.detection.image_list import ImageList

from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.transform import resize_boxes, resize_keypoints
from torchvision.models.detection.roi_heads import paste_masks_in_image

__all__ = ["ReverseTargetTransform", "Transform"]


class ReverseTargetTransform(torch.nn.Module):
    """
    Reverse the transformation on targets (e.g. box ratio).

    Adapted from torchvision.models.detection.transform.GeneralizedRCNNTransform::postprocess()
        https://github.com/pytorch/vision/blob/v0.11.3/torchvision/models/detection/transform.py#L223
    """

    def forward(
            self,
            result,  # type: List[Dict[str, Tensor]]
            image_shapes,  # type: List[Tuple[int, int]]
            original_image_sizes  # type: List[Tuple[int, int]]
    ):

        # type: (...) -> List[Dict[str, Tensor]]
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks

            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result


class Transform(GeneralizedRCNNTransform):
    """ Make a differentiable batching. """

    def __init__(self, *args, interpolation='bilinear', **kwargs):
        super().__init__(*args, **kwargs)

        self.interpolation = interpolation

    # Use parameterized _resize_image_and_masks
    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        h, w = image.shape[-2:]
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])
        if torchvision._is_tracing():
            raise NotImplementedError
        else:
            image, target = _resize_image_and_masks(image,
                                                    size,
                                                    float(self.max_size),
                                                    target,
                                                    interpolation=self.interpolation)

        if target is None:
            return image, target

        bbox = target["boxes"]
        if len(bbox) > 0:
            bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "keypoints" in target:
            keypoints = target["keypoints"]
            if len(keypoints) > 0:
                keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image, target

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)

        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # batch_shape = [len(images)] + max_size
        # batched_imgs = images[0].new_full(batch_shape, 0)
        # for img, pad_img in zip(images, batched_imgs):
        #     pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        # Replace with differentiable ops.
        # TODO: Fix this bug in the official torchvision.
        images_padded = []
        for img in images:
            img_size = img.shape
            pad = (0, max_size[-1] - img_size[-1], 0, max_size[-2] - img_size[-2])
            img_padded = torch.nn.functional.pad(img, pad, mode='constant', value=0.0)
            images_padded.append(img_padded)
        batched_imgs = torch.stack(images_padded)

        return batched_imgs

    def forward(
            self,
            images: List[Tensor],
            targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        image_list, targets = super().forward(images, targets)

        # Return Dict instead of Tuple
        ret = {"image_list": image_list, "original_image_sizes": original_image_sizes, "targets": targets}
        return ret


# Parameterize interpolation because depth images need to use nearest neighbor mode
def _resize_image_and_masks(image, self_min_size, self_max_size, target, interpolation='bilinear'):
    # type: (Tensor, float, float, Optional[Dict[str, Tensor]], str) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))
    max_size = float(torch.max(im_shape))
    scale_factor = self_min_size / min_size
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size
    interp_kwargs = {}
    if interpolation != "nearest":
        interp_kwargs["align_corners"] = False
    image = torch.nn.functional.interpolate(image[None],
                                            scale_factor=scale_factor,
                                            mode=interpolation,
                                            recompute_scale_factor=True,
                                            **interp_kwargs)[0]

    if target is None:
        return image, target

    if "masks" in target:
        mask = target["masks"]
        if len(mask) > 0:
            mask = F.interpolate(mask[:, None].float(), scale_factor=scale_factor)[:, 0].byte()
        target["masks"] = mask
    return image, target
