#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Optional, List, Dict, Tuple

import torch, torchvision
from torch import Tensor

from torchvision.models.detection.roi_heads import RoIHeads as RoIHeads_
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor as FastRCNNPredictor_
from torchvision.models.detection.roi_heads import fastrcnn_loss, maskrcnn_loss, keypointrcnn_loss
from torchvision.models.detection.roi_heads import maskrcnn_inference, keypointrcnn_inference

from oscar.models.detection.rcnn.rpn import ProposalSampler

__all__ = ["FastRCNNPredictor", "RoIHeads"]

# Prior-aware loss function whose additional goal is to remove samples that have their true
# label masked; otherwise they would lead to -infty cross-entropy loss
def fastrcnn_prior_loss(class_logits, priors, box_regression, labels, regression_targets):
    # If the label logit is masked, remove from training batch
    priors = torch.reshape(priors, (len(labels), -1))
    class_logits = torch.reshape(class_logits, (len(labels), -1, class_logits.shape[-1]))
    box_regression = torch.reshape(box_regression, (len(labels), -1, box_regression.shape[-1]))
    train_logits = [[] for idx in range(len(labels))]
    train_box  = [[] for idx in range(len(labels))]
    regression_targets = list(regression_targets)
    for sample_idx in range(len(labels)):
        gt_unmasked_idx = torch.where(priors[sample_idx] != labels[sample_idx])[0]

        # Remove from predictions
        train_logits[sample_idx] = class_logits[sample_idx][gt_unmasked_idx]
        train_box[sample_idx] = box_regression[sample_idx][gt_unmasked_idx]

        # Remove from GT
        labels[sample_idx] = labels[sample_idx][gt_unmasked_idx]
        regression_targets[sample_idx] = regression_targets[sample_idx][gt_unmasked_idx]
    # Shape back
    train_logits = torch.cat(train_logits, dim=0)
    train_box    = torch.cat(train_box, dim=0)

    # Legacy (prior unaware) loss
    classification_loss, box_loss = fastrcnn_loss(train_logits, train_box, labels, regression_targets)

    return classification_loss, box_loss

# Class and box predictor that supports anchor priors
class FastRCNNPredictor(FastRCNNPredictor_):
    def forward(self, x, priors=None):
        # Prior unaware logits and box coordinates
        scores, bbox_deltas = super().forward(x)

        if priors is not None:
            # Mask logits
            masked_idx = torch.where(priors >= 0)[0]
            if max(priors) == -1:
                assert len(masked_idx) == 0, \
                    'Logit masking takes place when it shouldn''t!'

            # Replace masked logits with -infinity
            scores[masked_idx, priors[masked_idx]] = -torch.inf

        return scores, bbox_deltas

# RoI head that supports anchor priors
class RoIHeads(RoIHeads_):
    def __init__(self, *args, **kwargs):
        # Populate fields
        super().__init__(*args, **kwargs)

        # Instantiate proposal sampler; defaults to evaluation
        self.proposal_sampler = ProposalSampler(
            self.box_coder, self.fg_bg_sampler, self.proposal_matcher,
            False, False)

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        full_proposals, # type: [Tensor, Dict]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            full_proposals ([Tensor, Dict])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
### BEGIN CHANGE
        # Extract proposals and their priors
        if type(full_proposals) == dict:
            proposals = full_proposals['boxes']
            priors    = full_proposals['priors']
        elif type(full_proposals) == list:
            proposals = full_proposals
            priors    = None
        else:
            assert False, 'Invalid proposal datatype %s passed to RoIHeads!' % type(full_proposals)

        # Change flags of proposal sampler
        self.proposal_sampler.add_gt_to_proposals = self.training
        self.proposal_sampler.balanced_proposal_sampling = self.training
### END CHANGE
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError("target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
### BEGIN CHANGE
            # Sample proposals and unwrap
            ret = self.proposal_sampler(proposals, targets, priors)
            proposals, priors = ret['proposals'], ret['priors']
            labels, matched_idxs = ret['labels'], ret['matched_idxs']
            regression_targets = ret['regression_targets']
### END CHANGE
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
### BEGIN CHANGE
        if priors is not None:
            priors = torch.cat(priors, dim=0).type(torch.long) # Flatten
        class_logits, box_regression = self.box_predictor(box_features, priors)
### END CHANGE
        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
### BEGIN CHANGE
            if priors is not None:
                loss_classifier, loss_box_reg = fastrcnn_prior_loss(
                    class_logits, priors, box_regression, labels, regression_targets)
            else:
                loss_classifier, loss_box_reg = fastrcnn_loss(
                    class_logits, box_regression, labels, regression_targets)
### END CHANGE
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)

        return result, losses
