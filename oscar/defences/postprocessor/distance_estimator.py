#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import List, Tuple, Dict
import numpy as np

class DistanceEstimator:
    def __init__(self,
                 intercept: List[float],
                 slope: List[float],
                 confidence: List[float] = [0.9, 0.9, 0.9],
                 penality_ratio: float = 0.9,
                 logarithmic: bool = True,
                 reciprocal: bool = True) -> None:
        # intercept and slope are list of parameters in order of [Pedestrian, Vehicle, TrafficLight]
        assert len(intercept) == len(slope) == len(confidence)
        self.intercept = intercept
        self.slope = slope
        self.confidence = confidence
        self.penality_ratio = penality_ratio
        self.logarithmic = logarithmic
        self.reciprocal = reciprocal

    def estimate_distance(self, label: int, bbox: np.ndarray) -> Tuple[float]:
        # bbox: (x1, y1, x2, y2)
        area = abs(bbox[0] - bbox[2]) * abs(bbox[1] - bbox[3])
        if self.logarithmic:
            area = np.log(area)

        idx = label - 1
        distance = self.intercept[idx] + self.slope[idx] * area
        delta = (1.0 - self.confidence[idx]) * distance
        return (distance - delta, distance + delta)

    def discount_score(self, depth: float, estimated_distance: Tuple[float], score: float) -> float:
        dist_low, dist_high = estimated_distance
        if dist_low <= depth <= dist_high:
            return score

        # If outside of confidence range, discount the score proportional to the deviation
        deviation = min(abs(depth - dist_low), abs(depth - dist_high)) / depth
        deviation = min(1, max(0, deviation))
        new_score = score * (1 - deviation * self.penality_ratio)
        return new_score

    def extract_median_depth(self, image: np.ndarray, bbox: np.ndarray) -> float:
        linear_depth = 1000 * np.exp(((image - 1) * 5.70378))
        x1, y1, x2, y2 = bbox.astype(int)
        distances = linear_depth[x1:x2, y1:y2]
        distance_median = np.median(distances)

        if self.logarithmic:
            distance_median = np.log(distance_median)

        if self.reciprocal:
            distance_median = np.reciprocal(distance_median)

        return distance_median

    def update_score(self, x: np.ndarray, preds: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        for img, pred in zip(x, preds):
            # Extract depth data: the last 3 channels are depth channels, which have same value
            assert img.shape[-1] == 6 or img.shape[-1] == 3
            depth = img[:, :, -1] # in range (0, 1)
            for i, label in enumerate(pred['labels']):
                distance = self.estimate_distance(label, pred['boxes'][i])
                depth_median = self.extract_median_depth(depth, pred['boxes'][i])
                pred['scores'][i] = self.discount_score(depth_median, distance, pred['scores'][i])

        return preds
