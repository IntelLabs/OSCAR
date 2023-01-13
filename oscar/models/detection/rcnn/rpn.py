#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Tuple, List, Dict, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

import torchvision
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.rpn import RegionProposalNetwork as RegionProposalNetwork_

__all__ = ["TorchvisionRPN", "RegionProposalNetwork", "RPNPredictor", "ProposalSampler", "RPNLoss"]

class TorchvisionRPN(RegionProposalNetwork_):
    """
    Assembling a torchvision compatible and prior-aware RPN using modular components
    """
    def __init__(self, *args, **kwargs):
        # Populate all needed fields
        super().__init__(*args, **kwargs)

        # Initialize components
        self.objectness_net = RegionProposalNetwork(self.head, self.anchor_generator)
        self.predictor = RPNPredictor(self.box_coder, self._pre_nms_top_n,
                                      self._post_nms_top_n, self.nms_thresh, self.score_thresh)
        self.rpn_loss = RPNLoss(self.box_coder, self.fg_bg_sampler,
                                self.proposal_matcher)
    def forward(self, images, features, targets = None):
        # Set auxiliaries
        self.add_gt_to_proposals        = self.training
        self.balanced_proposal_sampling = self.training

        # Get objectness and derivatives
        # objectness, pred_bbox_deltas, anchors, priors
        logits = self.objectness_net(images, features)

        # Support for legacy behaviour
        if not self.anchor_generator.use_priors:
            local_priors = None
        else:
            local_priors = logits['priors']

        # Get BBox predictions and derivatives
        # boxes, scores, priors
        predictions = self.predictor(logits['objectness'],
                                     logits['pred_bbox_deltas'],
                                     logits['anchors'],
                                     images.image_sizes,
                                     local_priors)

        # Training losses
        losses = {}
        if self.training:
            losses = self.rpn_loss(logits['objectness'],
                                   logits['pred_bbox_deltas'],
                                   logits['anchors'], targets)

        # Output dictionary if needed
        if not self.anchor_generator.use_priors:
            ret = predictions['boxes'] # Legacy torchvision behaviour
        else:
            ret = predictions

        return ret, losses


class RegionProposalNetwork(torch.nn.Module):
    """The model architecture of RPN, which outputs logits and anchors.
    The anchors will be consumed by both RPNLoss and RPNPredictor.

    Adapted from torchvision.models.detection.rpn.RegionProposalNetwork.
        https://github.com/pytorch/vision/blob/v0.11.3/torchvision/models/detection/rpn.py#L103
        * Removed Loss as a separate RPNLoss().
        * Removed Prediction as a separate RPNPredictor().
    """

    def __init__(
        self,
        head,
        anchor_generator,
    ):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head

    def forward(self, images, features):
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)

        # Anchors are required for calculating both loss and prediction,
        #   even though they are not part of the network.
        anchors = self.anchor_generator(images, features)
        if isinstance(anchors, tuple):
            anchors, priors = anchors
            # Move priors to right device
            # TODO: clean this up, for some reason AnchorGenerator doesn't place them on device
            priors = [priors[idx].to(anchors[0].device) for idx in range(len(priors))]
        else:
            priors = [None for _ in anchors]

        logits = {"objectness": objectness, "pred_bbox_deltas": pred_bbox_deltas,
                  "anchors": anchors, "priors": priors}
        return logits


class RPNPredictor(torch.nn.Module):
    """Making predictions from RPN logits.

    Adapted from torchvision.models.detection.rpn.RegionProposalNetwork.
        https://github.com/pytorch/vision/blob/v0.11.3/torchvision/models/detection/rpn.py#L103
    """

    def __init__(self, box_coder, pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh, min_size=1e-3):
        super().__init__()

        self.box_coder = box_coder

        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = min_size

    def forward(self, objectness, pred_bbox_deltas, anchors, image_sizes, priors=None):
        """_summary_

        Args:
            objectness (_type_): Score and filter proposal boxes.
            pred_bbox_deltas (_type_): Deltas from anchor boxes which yield proposal boxes.
            anchors (_type_): Anchors are required to decode proposal boxes from RPN logits of bbox_deltas.
            priors (_type_): Class priors for each anchor.
            image_sizes (_type_): Clip boxes to image.

        Returns:
            _type_: _description_
        """

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores, priors = \
            self.filter_proposals(proposals, objectness, priors, image_sizes, num_anchors_per_level)

        ret = {'boxes': boxes, 'scores': scores, 'priors': priors}
        return ret

    def filter_proposals(self, proposals, objectness, priors, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        if priors is not None:
            priors = [priors[idx][top_n_idx[idx]] for idx in range(num_images)]
        else:
            # Passthrough for modular functionality
            priors = [None for _ in range(len(proposals))]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        final_priors = []
        for boxes, scores, lvl, prior, img_shape in zip(proposals, objectness_prob, levels, priors, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            if prior is not None:
                prior = prior[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            if prior is not None:
                prior = prior[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            if prior is not None:
                prior = prior[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
            final_priors.append(prior)
        return final_boxes, final_scores, final_priors

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']


class ProposalSampler(torch.nn.Module):
    """
    In training mode, add gt boxes to proposals, then subsample a smaller number of proposals.

    Adapted from torchvision.models.detection.roi_heads.RoIHeads::select_training_samples()
        https://github.com/pytorch/vision/blob/v0.11.3/torchvision/models/detection/roi_heads.py#L746
    """

    def __init__(self, box_coder, fg_bg_sampler, proposal_matcher, add_gt_to_proposals, balanced_proposal_sampling):
        """_summary_

        Args:
            box_coder (_type_): _description_
            fg_bg_sampler (_type_): _description_
            proposal_matcher (_type_): _description_
            add_gt_to_proposals (bool, optional): Add groundtruth boxes into proposals. Used in training. Defaults to False.
            balanced_proposal_sampling (bool, optional): Sample positive and negative proposals. Used in training. Defaults to False.
        """
        super().__init__()

        self.box_coder = box_coder
        self.fg_bg_sampler = fg_bg_sampler
        self.proposal_matcher = proposal_matcher
        self.add_gt_to_proposals = add_gt_to_proposals
        self.balanced_proposal_sampling = balanced_proposal_sampling

    def forward(self, proposals, targets, priors=None):
        """
        Produce box regression targets for the downstream RoI Heads.
            Optionally, add groundtruth boxes into proposals
            Optionally, perform balanced sampling of proposals.

        Args:
            proposals (list): Initial proposals.
            targets (list): Groundtruth bounding boxes of objects.

        Returns:
            _type_: _description_
        """
        self.check_targets(targets)
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        if self.add_gt_to_proposals:
            # append ground-truth bboxes to proposals
            proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        if self.balanced_proposal_sampling:
            # sample a fixed proportion of positive-negative proposals
            sampled_inds = self.subsample(labels)

        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            if self.balanced_proposal_sampling:
                img_sampled_inds = sampled_inds[img_id]
                proposals[img_id] = proposals[img_id][img_sampled_inds]
                labels[img_id] = labels[img_id][img_sampled_inds]
                matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
                # Add unrestricted priors for ground-truth and sample
                if priors is not None:
                    priors[img_id] = torch.cat(
                        (priors[img_id], -torch.ones(len(gt_boxes[img_id]),
                                                    device=priors[img_id].device)))
                    priors[img_id] = priors[img_id][img_sampled_inds]

            # Calculate targets for roi_heads loss.
            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        ret = {"regression_targets": regression_targets, "labels": labels,
               "proposals": proposals, "priors": priors,
               "matched_idxs": matched_idxs}
        return ret

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])
        # if self.has_mask():
        #     assert all(["masks" in t for t in targets])

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros((proposals_in_image.shape[0],),
                                                            dtype=torch.int64,
                                                            device=device)
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds


class RPNLoss(torch.nn.Module):
    """Compute RPN losses from logits and targets.

    Adapted from torchvision.models.detection.rpn.RegionProposalNetwork.
        https://github.com/pytorch/vision/blob/v0.11.3/torchvision/models/detection/rpn.py#L103
    """

    def __init__(self, box_coder, fg_bg_sampler, proposal_matcher):
        super().__init__()
        self.box_coder = box_coder
        self.fg_bg_sampler = fg_bg_sampler

        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = proposal_matcher

    def forward(self, objectness, pred_bbox_deltas, anchors, targets):
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)

        losses = {}
        assert targets is not None
        labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return losses

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction='sum',
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes
