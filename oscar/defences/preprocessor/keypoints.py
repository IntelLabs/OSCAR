#
# Copyright (C) 2020 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
from typing import Dict, Generator, List, Optional, Tuple, Union

from detectron2.layers import cat, interpolate
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.structures import Instances
import numpy as np
import torch
import torch.nn as nn

from oscar.defences.preprocessor.preprocessor_pytorch import PreprocessorPyTorch
from oscar.defences.preprocessor.detectron2 import Detectron2PreprocessorPyTorch
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
        transforms.append(KeypointsToSTGCNInput(num_frames_enforced=None))

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

        # Put canvas on same device as maps because code below assumes it anyways.
        canvas = torch.zeros((max_instances, num_keypoints, H, W), device=maps.device)

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


class VideoToKeypointsPreprocessor(PreprocessorPyTorch):
    def __init__(
        self,
        *args,
        device_type: str = "gpu",
        batch_dim = 0,
        batch_size = 1,
        **kwargs,
    ):
        torch_module = VideoToKeypointsPreprocessorPyTorch(*args, **kwargs)

        super().__init__(torch_module, device_type=device_type, batch_dim=batch_dim, batch_size=batch_size)

    # NOTE: VideoToKeypointsPreprocessorPyTorch implements its own estimate_gradient so we call that here
    #       rather than use PreprocessorPyTorch's nice batching implementation.
    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return self.module.estimate_gradient(x, grad, device=self.module_device)

class VideoToKeypointsPreprocessorPyTorch(Detectron2PreprocessorPyTorch):
    def __init__(
        self,
        config_path: str,
        weights_path: str,
        score_thresh: float = 0.7,
        input_format: str = "RGB",
        channels_first: bool = False,
        clip_values: Tuple[float, float] = (0.0, 1.0),
        batch_frames_per_video: Optional[int] = 1,
        limit_frames_processed: Optional[int] = 300,
        keypoint_output_format: str = "openpose",
        convert_to_stgcn_input: bool = True,
    ):
        super().__init__(
            config_path,
            weights_path,
            score_thresh=score_thresh,
            iou_thresh=None,
        )

        del self.aug

        self.batch_frames_per_video = batch_frames_per_video
        self.limit_frames_processed = limit_frames_processed
        self.max_instances = 1

        self.input_video_transform = _make_input_transform(
            input_format, channels_first, clip_values
        )

        self.output_keypoints_transform = _make_output_transform(
            keypoint_output_format, convert_to_stgcn_input
        )

        self.argmax_layer = TwoDeeArgmax(temperature=100)

    def _iter_preprocessed_inputs(
        self, x: torch.Tensor
    ) -> Generator[List[dict], None, None]:

        for n in range(x.size(0)):
            video = x[n]

            if self.limit_frames_processed is not None:
                video = video[: self.limit_frames_processed]

            video = self.input_video_transform(video)

            yield [
                _create_input_dict(video[t])
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
                        # no instances were detected
                        if len(video_keypoints) > 0:
                            # interpolating last detected keypoints
                            frame_keypoints = video_keypoints[-1]
                        else:
                            # no keypoints detected in first frame, passing zeros
                            size = (self.max_instances,) + frame_keypoints.size()[1:]
                            frame_keypoints = torch.zeros(size, device=frame_keypoints.device)

                    # no keypoints detected in the frame, passing zeros
                    else:
                        if len(video_keypoints) > 0:
                            prev_frame_keypoints = video_keypoints[-1]
                            x_disp = torch.pow((torch.mean(prev_frame_keypoints[0, :, 0]) - torch.mean(frame_keypoints[0, :, 0])), 2)
                            y_disp = torch.pow((torch.mean(prev_frame_keypoints[0, :, 1]) - torch.mean(frame_keypoints[0, :, 1])), 2)

                            thre = 50000
                            if x_disp + y_disp > thre:
                                frame_keypoints = video_keypoints[-1]

                    video_keypoints.append(frame_keypoints)

            video_keypoints = torch.stack(video_keypoints)
            video_keypoints = self.output_keypoints_transform(video_keypoints)

            output.append(video_keypoints)

        return torch.stack(output)

    def estimate_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self.model.eval()
        self.argmax_layer.train()

        output = list()
        for video_inputs in self._iter_preprocessed_inputs(x):
            video_heatmaps = _estimate_heatmaps(
                dt2_model=self.model,
                batched_inputs=video_inputs,
                max_instances=self.max_instances,
            )

            video_keypoints = list()

            if video_heatmaps is not None:

                T, M, K, H, W = video_heatmaps.size()

                for t in range(T):
                    frame_keypoints = list()
                    for m in range(M):
                        hmaps = video_heatmaps[t, m]

                        # keypoints were either detected or this is first frame
                        if int(hmaps.sum()) != 0 or len(video_keypoints) == 0:
                            # keypoints were either detected or this is first frame
                            keypts = _heatmaps_to_keypoints(hmaps, self.argmax_layer)
                            frame_keypoints.append(keypts)
                        else:
                            # interpolating last detected keypoints for instance m
                            frame_keypoints.append(video_keypoints[-1][m])

                    frame_keypoints = torch.stack(frame_keypoints)
                    if len(video_keypoints) != 0:
                        prev_frame_keypoints = video_keypoints[-1]
                        x_disp = torch.pow((torch.mean(prev_frame_keypoints[0, :, 0]) - torch.mean(frame_keypoints[0, :, 0])), 2)
                        y_disp = torch.pow((torch.mean(prev_frame_keypoints[0, :, 1]) - torch.mean(frame_keypoints[0, :, 1])), 2)

                        thre = 50000
                        if x_disp + y_disp > thre:
                            frame_keypoints = video_keypoints[-1]
                    video_keypoints.append(frame_keypoints)

            else:
                # no instances were detected in any frame,
                # but armory has no mechanism to abstain

                T = len(video_inputs)
                for t in range(T):
                    frame = video_inputs[t]["image"]

                    # XXX: hack to pass dummy gradients
                    keypoints_shape = (self.max_instances, 17, 3)
                    dummy_keypoints = torch.zeros(keypoints_shape, device=frame.device)
                    dummy_keypoints[:, :, 2] += frame.norm()
                    # XXX: have set the confidences as norm of the frame

                    video_keypoints.append(dummy_keypoints)

            video_keypoints = torch.stack(video_keypoints)
            video_keypoints = self.output_keypoints_transform(video_keypoints)

            output.append(video_keypoints)

        return torch.stack(output)

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray, device) -> np.ndarray:
        N, T = x.shape[:2]

        output = list()
        for n in range(N):
            video_grads_out = list()
            for t in range(T):
                if (
                    self.limit_frames_processed is None
                    or t < self.limit_frames_processed
                ):
                    self.model.zero_grad()
                    self.argmax_layer.zero_grad()

                    frame_tensor = torch.tensor(
                        x[n : n + 1, t : t + 1].copy(),
                        device=device,
                        requires_grad=True,
                    )
                    frame_grads_in_tensor = torch.tensor(
                        grad[n : n + 1, :, t : t + 1], device=device
                    )

                    frame_keypoints_tensor = self.estimate_forward(frame_tensor)
                    frame_keypoints_tensor.backward(frame_grads_in_tensor)
                    frame_grads_out = frame_tensor.grad.detach().cpu().numpy()

                    assert frame_grads_out.shape[:2] == (1, 1)
                    video_grads_out.append(frame_grads_out[0, 0])

                else:
                    video_grads_out.append(np.zeros_like(x[n, t]))

            output.append(np.stack(video_grads_out, axis=0))

        return np.stack(output, axis=0)
