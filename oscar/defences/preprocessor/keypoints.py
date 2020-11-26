#
# Copyright (C) 2020 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Dict, Generator, List, Optional, Tuple, Union

from detectron2.layers import cat, interpolate
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.structures import Instances
import torch
import torch.nn as nn

from oscar.defences.preprocessor.detectron2_preprocessor import Detectron2Preprocessor
from oscar.utils.layers import TwoDeeArgmax
from oscar.utils.transforms.keypoints import (
    CocoToOpenposeKeypoints,
    KeypointsToSTGCNInput,
)
from oscar.utils.transforms.videos import (
    ChannelsFirst,
    RGB2BGR,
    UnNormalizeColorSpace,
)


def _make_input_transform(
    input_format: str,
    channels_first: bool,
    clip_values: Tuple[float, float],
) -> nn.Sequential:

    transforms: List[nn.Module] = list()

    if input_format == "RGB":
        transforms.append(RGB2BGR())
    else:
        assert input_format == "BGR"

    if not channels_first:
        transforms.append(ChannelsFirst())

    if clip_values == (0.0, 1.0):
        transforms.append(UnNormalizeColorSpace())
    else:
        assert clip_values == (0.0, 255.0)

    return nn.Sequential(*transforms)


def _make_output_transform(
    keypoint_output_format: str, convert_to_stgcn_input: bool
) -> nn.Sequential:

    transforms: List[nn.Module] = list()

    if keypoint_output_format == "openpose":
        transforms.append(CocoToOpenposeKeypoints())
    else:
        assert keypoint_output_format == "coco"

    if convert_to_stgcn_input:
        transforms.append(KeypointsToSTGCNInput(num_frames_enforced=300))

    return nn.Sequential(*transforms)


def _create_input_dict(
    image: torch.Tensor,
    gt_instances: Optional[Instances] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[torch.Tensor, Instances]]:

    C, H, W = image.size()
    assert C == 3

    if device is not None:
        image = image.to(device)

    input_dict = {"image": image.float(), "height": H, "width": W}

    if gt_instances is not None:
        input_dict.update(instances=gt_instances)

    return input_dict


def _estimate_heatmaps(
    dt2_model: GeneralizedRCNN,
    batched_inputs: List[dict],
    max_instances: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    # Reference: https://github.com/IntelLabs/GARD/blob/e4ed5b59f9736dcc0879c63d9eac243a135aab01/notebook/detectron2_keypoint_heatmaps.ipynb

    T = len(batched_inputs)

    output = list()
    num_empty_heatmaps = 0
    for t in range(T):
        frame_input = batched_inputs[t]
        H, W = frame_input["height"], frame_input["width"]

        images = dt2_model.preprocess_image([frame_input])
        features = dt2_model.backbone(images.tensor)
        proposals, _ = dt2_model.proposal_generator(images, features, None)
        pred_instances = dt2_model.roi_heads._forward_box(features, proposals)

        features = [features[f] for f in dt2_model.roi_heads.keypoint_in_features]
        pred_boxes = [x.pred_boxes for x in pred_instances]

        keypoint_features = dt2_model.roi_heads.keypoint_pooler(features, pred_boxes)

        maps = dt2_model.roi_heads.keypoint_head.layers(keypoint_features)
        rois = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

        widths_ceil = (rois[:, 2] - rois[:, 0]).clamp(min=1).ceil()
        heights_ceil = (rois[:, 3] - rois[:, 1]).clamp(min=1).ceil()

        num_instances, num_keypoints = maps.shape[:2]

        if num_instances == 0:
            num_empty_heatmaps += 1

        canvas = torch.zeros((max_instances, num_keypoints, H, W))
        if device is not None:
            canvas = canvas.to(device)

        for i in range(min(max_instances, num_instances)):
            outsize = (int(heights_ceil[i]), int(widths_ceil[i]))

            roi_map = interpolate(
                maps[i].unsqueeze(0), size=outsize, mode="bicubic", align_corners=False
            ).squeeze(0)

            max_score, _ = roi_map.view(num_keypoints, -1).max(1)
            max_score = max_score.view(num_keypoints, 1, 1)
            tmp_full_res = (roi_map - max_score).exp_()
            tmp_pool_res = (maps[i] - max_score).exp_()

            roi_map_scores = tmp_full_res / tmp_pool_res.sum((1, 2), keepdim=True)

            for k in range(roi_map_scores.shape[0]):
                heatmap = roi_map_scores[k, :, :]
                h, w = heatmap.size(0), heatmap.size(1)
                x1, y1, x2, y2 = rois[i]

                canvas[i, k, int(y1) : h + int(y1), int(x1) : w + int(x1)] += heatmap

        output.append(canvas)

    return torch.stack(output) if num_empty_heatmaps < T else None


def _heatmaps_to_keypoints(heatmaps: torch.Tensor, argmax: nn.Module) -> torch.Tensor:

    K, H, W = heatmaps.size()

    output = list()

    for k in range(K):
        heatmap = heatmaps[k]

        x_argmax, y_argmax = argmax(heatmap)
        score = heatmap[
            y_argmax.detach().round().long().clamp(0, H - 1),
            x_argmax.detach().round().long().clamp(0, W - 1),
        ]

        keypoint = [x_argmax.unsqueeze(0), y_argmax.unsqueeze(0), score.unsqueeze(0)]
        keypoint = torch.cat(keypoint, dim=0).float()

        output.append(keypoint)

    return torch.stack(output)


class VideoToKeypointsPreprocessor(Detectron2Preprocessor):
    def __init__(
        self,
        config_path: str,
        weights_path: str,
        score_thresh: float = 0.7,
        input_width: int = 320,
        input_height: int = 240,
        input_format: str = "RGB",
        channels_first: bool = False,
        clip_values: Tuple[float, float] = (0.0, 1.0),
        batch_frames_per_video: Optional[int] = 100,
        limit_frames_processed: Optional[int] = 300,
        limit_frames_estimated: Optional[int] = 50,
        keypoint_output_format: str = "openpose",
        convert_to_stgcn_input: bool = True,
        device_type: str = "gpu",
    ):

        super().__init__(
            config_path,
            weights_path,
            score_thresh=score_thresh,
            iou_thresh=None,
            device_type=device_type,
        )

        del self.aug

        self.batch_frames_per_video = batch_frames_per_video
        self.limit_frames_processed = limit_frames_processed
        self.limit_frames_estimated = limit_frames_estimated
        self.max_instances = 1

        self.input_video_transform = _make_input_transform(
            input_format, channels_first, clip_values
        ).to(self._device)

        self.output_keypoints_transform = _make_output_transform(
            keypoint_output_format, convert_to_stgcn_input
        ).to(self._device)

        self.argmax_layer = TwoDeeArgmax(temperature=100).to(self._device)

    def _iter_preprocessed_inputs(
        self, x: torch.Tensor, estimate=False
    ) -> Generator[List[dict], None, None]:

        for n in range(x.size(0)):
            video = x[n]

            if estimate and self.limit_frames_estimated is not None:
                # typically limit_frames_estimated < limit_frames_processed
                # since DT2 model could go out of memory
                # for large number of frames in backward pass
                video = video[: self.limit_frames_estimated]

            elif not estimate and self.limit_frames_processed is not None:
                video = video[: self.limit_frames_processed]

            video = self.input_video_transform(video)

            yield [
                _create_input_dict(video[t], device=self._device)
                for t in range(video.size(0))
            ]

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        self.model.eval()

        output = list()
        for video_inputs in self._iter_preprocessed_inputs(x):
            T = len(video_inputs)
            batch_size = int(self.batch_frames_per_video or T)

            video_keypoints = list()
            for t in range(0, T, batch_size):
                batch_inputs = video_inputs[t : min(t + batch_size, T)]
                outputs = self.model(batch_inputs)

                for t_prime in range(len(outputs)):
                    instances = outputs[t_prime]["instances"]
                    frame_keypoints = instances.pred_keypoints[: self.max_instances]

                    if frame_keypoints.size(0) == 0:
                        # no instances were detected, passing zeros as keypoints
                        size = (self.max_instances,) + frame_keypoints.size()[1:]
                        frame_keypoints = torch.zeros(size).to(self._device)

                    video_keypoints.append(frame_keypoints)

            video_keypoints = torch.stack(video_keypoints)
            video_keypoints = self.output_keypoints_transform(video_keypoints)

            output.append(video_keypoints)

        return torch.stack(output), y

    def estimate_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self.model.eval()
        self.argmax_layer.train()

        output = list()
        for video_inputs in self._iter_preprocessed_inputs(x, estimate=True):
            video_heatmaps = _estimate_heatmaps(
                dt2_model=self.model,
                batched_inputs=video_inputs,
                max_instances=self.max_instances,
                device=self._device,
            )

            video_keypoints = list()

            if video_heatmaps is not None:

                T, M, K, H, W = video_heatmaps.size()

                for t in range(T):
                    frame_keypoints = list()
                    for m in range(M):
                        hmaps = video_heatmaps[t, m]
                        frame_keypoints.append(
                            _heatmaps_to_keypoints(hmaps, self.argmax_layer)
                        )
                    frame_keypoints = torch.stack(frame_keypoints)
                    video_keypoints.append(frame_keypoints)

            else:
                # no instances were detected in any frame,
                # but armory has no mechanism to abstain

                T = len(video_inputs)
                for t in range(T):
                    frame = video_inputs[t]["image"]

                    # XXX: hack to pass dummy gradients
                    keypoints_shape = (self.max_instances, 17, 3)
                    dummy_keypoints = torch.zeros(keypoints_shape).to(self._device)
                    dummy_keypoints[:, :, 2] += frame.norm()
                    # XXX: have set the confidences as norm of the frame

                    video_keypoints.append(dummy_keypoints)

            video_keypoints = torch.stack(video_keypoints)
            video_keypoints = self.output_keypoints_transform(video_keypoints)

            output.append(video_keypoints)

        return torch.stack(output)
