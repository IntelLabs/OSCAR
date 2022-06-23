#
# Copyright (C) 2022 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
from typing import Optional
from typing import Tuple
from typing import Any
import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch


class StepSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return torch.sigmoid(input)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> torch.Tensor:
        input = ctx.saved_tensors[0]
        grad = torch.ones_like(input)

        GAMMA = 1e-2 / 1e4
        grad[input > 5] = GAMMA
        grad[input < -5] = GAMMA

        return grad


class DynamicBackgroundSubtraction(PreprocessorPyTorch):
    def __init__(self, **kwargs):
        super().__init__()
        assert "orb_good_match_percent" in kwargs.keys()
        assert "orb_levels" in kwargs.keys()
        assert "orb_scale_factor" in kwargs.keys()
        assert "bg_sub_thre" in kwargs.keys()

        self.orb_good_match_percent = kwargs['orb_good_match_percent']
        self.orb_levels = kwargs['orb_levels']
        self.orb_scale_factor = kwargs['orb_scale_factor']
        self.orb_gaussian_ksize = (
            kwargs["orb_gaussian_ksize"][0], kwargs["orb_gaussian_ksize"][1]
        ) if "orb_gaussian_ksize" in kwargs.keys() else None
        self.subtraction_gaussian_ksize = (
            kwargs["subtraction_gaussian_ksize"][0],
            kwargs["subtraction_gaussian_ksize"][1]
        ) if "subtraction_gaussian_ksize" in kwargs.keys() else None
        self.bg_sub_thre = kwargs["bg_sub_thre"]
        self.median_ksize = (kwargs["median_ksize"][0],
                             kwargs["median_ksize"][1]
                             ) if "median_ksize" in kwargs.keys() else None
        self.binary_mask = True if "binary_mask" in kwargs.keys(
        ) and kwargs["binary_mask"] == "true" else False

        self.orb_max_matches = 1e7
        self.orb_max_iters = 1e6

        self.ORB = cv2.ORB_create(nfeatures=int(self.orb_max_matches),
                                  nlevels=self.orb_levels,
                                  scaleFactor=self.orb_scale_factor)
        self.ORB_matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    def _preprocess_frame(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame before keypoint detection
        1. Blur frames
        2. Covert to grayscale
        """
        # blur frames
        if self.orb_gaussian_ksize is not None:
            img = cv2.GaussianBlur(img, self.orb_gaussian_ksize,
                                   cv2.BORDER_DEFAULT)

        # convert frames to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _postprocess_frame(self, img: torch.Tensor) -> torch.Tensor:
        """
        Postprocess the frame from background subtraction
        Apply median filter to remove the pepper and salt noise (dots aren't removed in background subtraction)
        """
        img = torch.permute(img, (0, 3, 1, 2))
        # blur it sequentially in case cuda out of memory
        img_blur = torch.zeros_like(img)
        for i in range(img.shape[0]):
            img_blur[i] = kornia.filters.median_blur(
                img[i].type(torch.float16).unsqueeze(0), kernel_size=self.median_ksize)[0]
        img_blur = torch.permute(img_blur, (0, 2, 3, 1))
        return img_blur

    def _detect_keypoints(self,
                          img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect keypoints of the given frame, and return the keypoint coordinates and descriptors
        img: (H, W, C)
        """
        # preprocess frame before keypoint detection
        img_pre = self._preprocess_frame(img)

        # orb keypoint detection
        kpts, descs = self.ORB.detectAndCompute(img_pre, None)
        return kpts, descs

    def _get_homography(self, kpts1: np.ndarray, descs1: np.ndarray,
                        kpts2: np.ndarray, descs2: np.ndarray) -> np.ndarray:
        """
        Find homography between 2 frames
        kpts: detected keypoints
        descs: computed descriptors
        """
        # match features and sort by score
        matches = self.ORB_matcher.match(descs1, descs2, None)
        matches = list(matches)
        matches.sort(key=lambda x: x.distance, reverse=False)

        # preserve good matches
        numGoodMatches = int(len(matches) * self.orb_good_match_percent)
        matches = matches[:numGoodMatches]

        # extract location of good matches
        pts1 = np.zeros((len(matches), 2), dtype=np.float32)
        pts2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            pts1[i, :] = kpts1[match.queryIdx].pt
            pts2[i, :] = kpts2[match.trainIdx].pt

        # find homography matrix H
        H, _ = cv2.findHomography(pts1,
                                  pts2,
                                  cv2.USAC_ACCURATE,
                                  maxIters=int(self.orb_max_iters))
        assert H.shape == (3, 3)

        return H

    def _warp_frame(
            self, frames: torch.Tensor,
            ref_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Warp frames with respect to the reference frame
        """
        K, N, H, W, _ = frames.shape
        # all frames should in range [0, 1]
        assert frames.max() <= 1.0, frames.max()
        assert K == 1  # batch size 1

        frames_np = frames.detach().cpu().squeeze().numpy()
        frames_np = (frames_np * 255.0).astype(np.uint8)

        # detect keypoint in all frames
        orb_list = list()
        for idx in range(N):
            kpts, descs = self._detect_keypoints(frames_np[idx])
            orb_list.append((kpts, descs))

        # compute and store all homography matrices
        homo_list = list()  # store all homography matrices
        for idx in range(N):
            # use reference frame as global coordinate
            h = self._get_homography(*orb_list[idx], *orb_list[ref_idx])
            homo_list.append(h)

        homo_list = torch.from_numpy(np.array(homo_list)).to(frames.device)

        # differentiable warp perspective
        frames_c = torch.permute(frames[0], (0, 3, 1, 2))
        frames_warped = kornia.geometry.transform.warp_perspective(
            frames_c, homo_list, (H, W))
        frames_warped = torch.permute(frames_warped,
                                      (0, 2, 3, 1)).to(frames.device)

        # create mask for masked array median computation
        masks = torch.ones((N, 1, H, W), dtype=torch.double).to(frames.device)
        masks = kornia.geometry.transform.warp_perspective(
            masks, homo_list, (H, W))
        masks = masks.squeeze()
        masks[masks != 0] = 1
        masks = 1 - masks

        return frames_warped, masks, homo_list

    def _compute_RGB_median(
            self, frames: torch.Tensor,
            masks: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the masked median
        Each channel is computed separately
        """
        N, H, W, _ = frames.shape  # warped frames
        assert masks.shape == (N, H, W)

        frame_c = frames.detach().cpu().numpy()
        # mask has the same size as frames
        masks = torch.repeat_interleave(masks.unsqueeze(-1), repeats=3, dim=3)
        masks = masks.detach().cpu().numpy()

        # use masked array for median computation
        frame_ma = np.ma.array(frame_c, mask=masks)
        median = np.ma.median(frame_ma, axis=0).data
        assert median.shape == (H, W, 3)

        return median, frame_ma, masks

    def _transform_frame(self, frame: np.ndarray,
                         median: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blur warped frames and median before subtraction
        """
        if self.subtraction_gaussian_ksize is not None:
            for idx in range(frame.shape[0]):
                frame[idx] = cv2.GaussianBlur(frame[idx],
                                              self.subtraction_gaussian_ksize,
                                              cv2.BORDER_DEFAULT)
            median = cv2.GaussianBlur(median, self.subtraction_gaussian_ksize,
                                      cv2.BORDER_DEFAULT)

        return frame, median

    def _background_subtraction(self, frame_ma: np.ndarray, median: np.ndarray,
                                threshold: float) -> torch.Tensor:
        """
        Denote [median - threshold, median + threshold] as background
        Subtract background from the warped frames
        """
        # preprocess warped frames and median
        frame_ma, median = self._transform_frame(frame_ma, median)

        ALPHA = 1e-10
        bg_sub_mask = np.abs(np.asarray(frame_ma - median))
        bg_sub_mask = torch.from_numpy(bg_sub_mask)
        bg_sub_mask = bg_sub_mask.mean(dim=-1).unsqueeze(
            dim=-1)  # avg over color dimension
        bg_sub_mask = bg_sub_mask - threshold
        bg_sub_mask = bg_sub_mask / (torch.abs(bg_sub_mask) + ALPHA)
        bg_sub_mask = F.relu(bg_sub_mask)
        assert not bg_sub_mask.isnan().any()

        return bg_sub_mask

    def _binarize_images(self, img: torch.Tensor) -> torch.Tensor:
        """
        Replace the vanishing gradient with a small const
        """
        BETA = 1e-3
        img_out = (img - BETA) / BETA * 100
        img_out = StepSigmoid.apply(img_out)

        return img_out

    def _restore_frame(self, frames: torch.Tensor,
                       homo_list: torch.Tensor) -> torch.Tensor:
        """
        Warp the frames back to the original camera view
        """
        _, H, W, _ = frames.shape

        frames_c = torch.permute(frames, (0, 3, 1, 2))
        inv_homo_list = torch.linalg.inv(homo_list)
        frames_res = kornia.geometry.transform.warp_perspective(
            frames_c, inv_homo_list, (H, W))
        frames_res = torch.permute(frames_res, (0, 2, 3, 1)).to(frames.device)

        return frames_res

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x.double()

        # 1. warp frames
        frames_warped, masks, homo_list = self._warp_frame(x, ref_idx=0)

        # 2. compute median
        median, frame_ma, masks = self._compute_RGB_median(
            frames_warped, masks)

        # 3. apply background subtraction
        bg_sub_mask = self._background_subtraction(frame_ma,
                                                   median,
                                                   threshold=self.bg_sub_thre)
        bg_sub_mask = bg_sub_mask.to(frames_warped.device)

        # 3.1 track rgb frames
        frames_warped_bgsub = frames_warped * bg_sub_mask

        # 3.2 track binary mask (optional)
        if self.binary_mask:
            frames_warped_bgsub = self._binarize_images(frames_warped_bgsub)

        # 3.3 apply median filter (optional, but shouldn't be used in combination with 3.2)
        if self.median_ksize is not None:
            frames_warped_bgsub = self._postprocess_frame(frames_warped_bgsub)

        # 4. transform back to original camera view
        frames_warped_bgsub_restored = self._restore_frame(
            frames_warped_bgsub, homo_list)

        # 5. unsqueeze to (1, N, H, W, C)
        frames_warped_bgsub_restored = frames_warped_bgsub_restored.unsqueeze(
            0).float().to(x.device)
        assert frames_warped_bgsub_restored.shape == x.shape

        return frames_warped_bgsub_restored, y


def main():
    import argparse
    import glob
    from pathlib import Path

    import numpy as np
    from cv2 import imread
    from cv2 import imwrite
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("pathname", type=str)
    parser.add_argument("--postfix", type=str, default="_bgmask")
    args = parser.parse_args()

    preprocessor = DynamicBackgroundSubtraction()

    # Hack for https://github.com/Trusted-AI/adversarial-robustness-toolbox/issues/1442
    preprocessor.params.append("_device")
    preprocessor.set_params(_device="cpu")

    images = list()
    print("Loading images assuming all frames are named in a sorted order...")
    filepaths = sorted(glob.glob(args.pathname))
    for filepath in tqdm(filepaths):
        images.append(imread(filepath))

    x = np.stack(images, axis=0) / 255.0
    x = np.expand_dims(x, axis=0)
    x = preprocessor(x)[0][0]
    x = np.uint8(x * 255.0)
    print("Finished preprocessing all frames.")

    print("Saving frames to disk...")
    for i, filepath in tqdm(enumerate(filepaths), total=len(filepaths)):
        filepath_ = Path(filepath)
        filepath_ = filepath_.parent / (filepath_.stem + args.postfix +
                                        filepath_.suffix)
        imwrite(str(filepath_), x[i])


if __name__ == "__main__":
    main()
