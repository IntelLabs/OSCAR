#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch

from torchvision.models.detection.rpn import RegionProposalNetwork as RegionProposalNetwork_


class RegionProposalNetwork(RegionProposalNetwork_):
    """
        Adds ability to filter region proposals by score threshold.

            score_thresh (float): only propose regions with objectness score >= score_thresh (default: 0.0)
    """
    def __init__(self, *args, score_thresh=0.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.score_thresh = score_thresh

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        boxes, scores = super().filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level)

        # Keep boxes with score > score_thresh
        top_boxes = []
        top_scores = []
        for b, s in zip(boxes, scores):
            keep = torch.sigmoid(s) >= self.score_thresh
            top_boxes.append(b[keep])
            top_scores.append(s[keep])

        return top_boxes, top_scores
