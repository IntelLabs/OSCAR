#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Tuple, List, Dict

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.roi_heads import fastrcnn_loss

__all__ = ["BoxHead", "BoxPredictor", "BoxLoss"]


class BoxHead(torch.nn.Module):
    """The model architecture of the RoI-Box head, which outputs logits.

    Adapted from torchvision.models.detection.roi_heads.RoIHeads
        https://github.com/pytorch/vision/blob/v0.11.3/torchvision/models/detection/roi_heads.py#L485
        * Removed select_training_samples() as a separate rpn.ProposalSampler.
        * Removed Loss as a separate BoxLoss.
        * Removed Prediction as a separate BoxPredictor.
    """

    def __init__(self, box_roi_pool, box_head, box_predictor):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

    def forward(self, features, proposals, image_shapes):
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        ret = {}
        ret['class_logits'] = class_logits
        ret['box_regression'] = box_regression
        return ret


class BoxPredictor(torch.nn.Module):
    """Making Box predictions from logits.

    Adapted from torchvision.models.detection.roi_heads.RoIHeads
        https://github.com/pytorch/vision/blob/v0.11.3/torchvision/models/detection/roi_heads.py#L485
    """

    def __init__(self, box_coder, score_thresh, nms_thresh, detections_per_img, min_size=1e-2, area_threshold=0.):
        super().__init__()

        self.box_coder = box_coder
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.min_size = min_size
        self.area_threshold = area_threshold

    @torch.no_grad()
    def forward(self, class_logits, box_regression, proposals, image_shapes):
        """We add the no_grad decorator because we usually don't need gradients in prediction.
        """

        result: List[Dict[str, torch.Tensor]] = []
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append({
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            })

        return result

    def postprocess_detections(
            self,
            class_logits,  # type: Tensor
            box_regression,  # type: Tensor
            proposals,  # type: List[Tensor]
            image_shapes  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=self.min_size)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # keep boxes with area greater than threshold
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            inds = torch.where(areas >= self.area_threshold)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels


class BoxLoss(torch.nn.Module):
    """Computing Box losses from logits.

    Adapted from torchvision.models.detection.roi_heads.RoIHeads
        https://github.com/pytorch/vision/blob/v0.11.3/torchvision/models/detection/roi_heads.py#L485
    """

    def __init__(self):
        super().__init__()

    def forward(self, class_logits, box_regression, labels, regression_targets):
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)

        losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        return losses
