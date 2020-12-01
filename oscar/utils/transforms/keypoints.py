#
# Copyright (C) 2020 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

import math
from typing import *

import torch
import torch.nn as nn


class CocoToOpenposeKeypoints(nn.Module):
    def forward(self, coco_keypoints: torch.Tensor) -> torch.Tensor:
        # coco keypoints (k=17):
        #     ['Nose', Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
        #      'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']
        # openpose keypoints (k=18, with Neck):
        #     ['Nose', *'Neck'*, 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri','Rhip',
        #      'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

        T, M, K, V = coco_keypoints.size()
        assert K == 17
        assert V == 3

        ordered_indices = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 1, 2, 3, 4]

        openpose_keypoints = list()
        for t in range(T):
            instance_list = list()

            for m in range(M):
                # This will re-arrange coco keypoints to openpose ordering,
                # WITHOUT the neck keypoint
                reordered_keypoints = [coco_keypoints[t, m, i] for i in ordered_indices]

                l_shoulder_keypoint = coco_keypoints[t, m, 5]
                r_shoulder_keypoint = coco_keypoints[t, m, 6]

                neck_keypoint = (l_shoulder_keypoint + r_shoulder_keypoint) / 2
                neck_keypoint[:2] = torch.round(neck_keypoint[:2])

                reordered_keypoints.insert(1, neck_keypoint)
                instance_list.append(torch.stack(reordered_keypoints))

            openpose_keypoints.append(torch.stack(instance_list))

        return torch.stack(openpose_keypoints)  # (T, M, 18, 3)


class OpenposeToCocoKeypoints(nn.Module):
    def __init__(self, reconvert_from_stgcn_input: bool = True):
        super().__init__()
        self.reconvert_from_stgcn_input = reconvert_from_stgcn_input

    def forward(self, openpose_keypoints: torch.Tensor) -> torch.Tensor:
        # openpose keypoints (k=18, with Neck):
        #     ['Nose', *'Neck'*, 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri','Rhip',
        #      'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']
        # coco keypoints (k=17):
        #     ['Nose', Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
        #      'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']

        if self.reconvert_from_stgcn_input:
            openpose_keypoints = openpose_keypoints.permute(1, 3, 2, 0)

        T, M, K, V = openpose_keypoints.size()
        assert K == 18
        assert V == 3

        ordered_indices = [0, 13, 14, 15, 16, 4, 1, 5, 2, 6, 3, 10, 7, 11, 8, 12, 9]

        coco_keypoints = list()
        for t in range(T):
            instance_list = list()

            for m in range(M):
                keypoints = openpose_keypoints[t, m]

                # delete the neck keypoint
                keypoints = torch.cat([keypoints[:1], keypoints[2:]])

                reordered_keypoints = [keypoints[i] for i in ordered_indices]
                instance_list.append(torch.stack(reordered_keypoints))

            coco_keypoints.append(torch.stack(instance_list))

        return torch.stack(coco_keypoints)  # (T, M, 17, 3)


class KeypointsToSTGCNInput(nn.Module):
    def __init__(self, num_frames_enforced: Optional[int] = None):
        super().__init__()

        self.num_frames_enforced = num_frames_enforced

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        T, M, K, V = keypoints.size()
        assert K == 18
        assert V == 3

        if self.num_frames_enforced is not None:
            repeat = math.ceil(float(self.num_frames_enforced) / T)
            if repeat >= 2:
                keypoints = keypoints.repeat(repeat, 1, 1, 1)
            keypoints = keypoints[: int(self.num_frames_enforced)]

        return keypoints.permute(3, 0, 2, 1)  # (3, T', 18, M)


class BatchStandardizeSTGCNInputTimesteps(nn.Module):
    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        N, V, T, K, M = keypoints.size()
        assert K == 18
        assert V == 3
        assert M <= 2

        num_target_timesteps = 300

        repeat = math.ceil(float(num_target_timesteps) / T)
        if repeat >= 2:
            keypoints = keypoints.repeat(1, 1, repeat, 1, 1)
        keypoints = keypoints[:, :, :num_target_timesteps]

        return keypoints


def get_dt2_to_stgcn_default_keypoint_transforms():
    return [
        CocoToOpenposeKeypoints(),
        KeypointsToSTGCNInput(num_frames_enforced=300),
    ]
