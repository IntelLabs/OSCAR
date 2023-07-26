#
# Copyright (C) 2023 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
from typing import Optional, Tuple, Any, List, Dict, Union
import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector
from oscar.defences.preprocessor.small_scale_filter import SmallScaleFilter

def _get_new_losses(
    self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]
) -> Tuple[Dict[str, "torch.Tensor"], List["torch.Tensor"], List["torch.Tensor"]]:
    """
    Get the loss tensor output of the model including all preprocessing.

    :param x: Samples of shape (nb_samples, height, width, nb_channels).
    :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The fields of the Dict are as
                follows:

                - boxes (FloatTensor[N, 4]): the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                                            0 <= y1 < y2 <= H.
                - labels (Int64Tensor[N]): the labels for each image
    :return: Loss gradients of the same shape as `x`.
    """
    import torch
    import torchvision

    self._model.train()

    # Apply preprocessing
    if self.all_framework_preprocessing:

        if y is not None and isinstance(y, list) and isinstance(y[0]["boxes"], np.ndarray):
            y_tensor = []
            for i, y_i in enumerate(y):
                y_t = {}
                y_t["boxes"] = torch.from_numpy(y_i["boxes"]).type(torch.float).to(self.device)
                y_t["labels"] = torch.from_numpy(y_i["labels"]).type(torch.int64).to(self.device)
                if "masks" in y_i:
                    y_t["masks"] = torch.from_numpy(y_i["masks"]).type(torch.int64).to(self.device)
                y_tensor.append(y_t)
        elif y is not None and isinstance(y, dict):
            y_tensor = []
            for i in range(y["boxes"].shape[0]):
                y_t = {}
                y_t["boxes"] = y["boxes"][i]
                y_t["labels"] = y["labels"][i]
                y_tensor.append(y_t)
        else:
            y_tensor = y  # type: ignore

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_tensor_list_grad = []
        y_preprocessed = []
        inputs_t = []

        if isinstance(x, np.ndarray):
            if self.clip_values is not None:
                x_grad = transform(x / self.clip_values[1]).to(self.device)
            else:
                x_grad = transform(x).to(self.device)
            x_grad.requires_grad = True
        else:
            x_grad = x.to(self.device)
            if x_grad.shape[3] < x_grad.shape[1] and x_grad.shape[3] < x_grad.shape[2]:
                x_grad = torch.permute(x_grad, (0, 3, 1, 2))

        image_tensor_list_grad.append(x_grad)
        x_grad_1 = x_grad
        x_preprocessed_i, y_preprocessed_i = self._apply_preprocessing(
            x_grad_1, y=y_tensor, fit=False, no_grad=False
        )
        for i_preprocessed in range(x_preprocessed_i.shape[0]):
            inputs_t.append(x_preprocessed_i[i_preprocessed])
            y_preprocessed.append(y_preprocessed_i[i_preprocessed])

    elif isinstance(x, np.ndarray):
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y, fit=False, no_grad=True)

        if y_preprocessed is not None and isinstance(y_preprocessed[0]["boxes"], np.ndarray):
            y_preprocessed_tensor = []
            for i, y_i in enumerate(y_preprocessed):
                y_preprocessed_t = {}
                y_preprocessed_t["boxes"] = torch.from_numpy(y_i["boxes"]).type(torch.float).to(self.device)
                y_preprocessed_t["labels"] = torch.from_numpy(y_i["labels"]).type(torch.int64).to(self.device)
                if "masks" in y_i:
                    y_preprocessed_t["masks"] = torch.from_numpy(y_i["masks"]).type(torch.uint8).to(self.device)
                y_preprocessed_tensor.append(y_preprocessed_t)
            y_preprocessed = y_preprocessed_tensor

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_tensor_list_grad = []

        for i in range(x_preprocessed.shape[0]):
            if self.clip_values is not None:
                x_grad = transform(x_preprocessed[i] / self.clip_values[1]).to(self.device)
            else:
                x_grad = transform(x_preprocessed[i]).to(self.device)
            x_grad.requires_grad = True
            image_tensor_list_grad.append(x_grad)

        inputs_t = image_tensor_list_grad

    else:
        raise NotImplementedError("Combination of inputs and preprocessing not supported.")

    if isinstance(y_preprocessed, np.ndarray):
        labels_t = torch.from_numpy(y_preprocessed).to(self.device)  # type: ignore
    else:
        labels_t = y_preprocessed  # type: ignore

    if inputs_t[0].shape[0] != 3:
        inputs_t = [ip.permute(2, 0, 1) for ip in inputs_t]
    output = self._model(inputs_t, labels_t)

    return output, inputs_t, image_tensor_list_grad

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

        #"""
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

        self.edge_filter_size = (kwargs["edge_filter_size"]) if "edge_filter_size" in kwargs.keys() else 5
        self.gaussian_filter_size = (kwargs["gaussian_filter_size"]) if "gaussian_filter_size" in kwargs.keys() else 11
        self.edge_sub_thres = (kwargs["edge_sub_thres"]) if "edge_sub_thres" in kwargs.keys() else 15.0/255.0

        self.prefilter_ss = True if "prefilter_ss" in kwargs.keys(
                        ) and kwargs["prefilter_ss"] == "true" else False
        self.postfilter_ss = True if "postfilter_ss" in kwargs.keys(
                        ) and kwargs["postfilter_ss"] == "true" else False


        self.orb_max_matches = 5e3 #1e7
        self.orb_max_iters = 1e3

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

        pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # find homography matrix H
        H, _ = cv2.findHomography(pts1, pts2, cv2.USAC_ACCURATE,
                                  maxIters=int(self.orb_max_iters))
        assert H.shape == (3, 3)

        return H

    def _warp_frame(
            self, frames: torch.Tensor,
            ref_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Warp frames with respect to the reference frame
        """
        if(frames.shape[-1] != 3):
            frames = torch.permute(frames, (0,2,3,1))
        frames = torch.unsqueeze(frames, dim = 0)

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
            frames_c, homo_list, (int(1.5*H), int(1.5*W)))
        frames_warped = torch.permute(frames_warped,
                                      (0, 2, 3, 1)).to(frames.device)

        # create mask for masked array median computation
        masks = torch.ones((N, 1, H, W), dtype=torch.double).to(frames.device)
        masks = kornia.geometry.transform.warp_perspective(
            masks, homo_list, (int(1.5*H), int(1.5*W)))
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

        # mask should have the same size as frames
        masks = torch.repeat_interleave(masks.unsqueeze(-1), repeats=3, dim=3)
        median = torch.quantile(masks*frames,0.5,dim=0,keepdim=False,interpolation='lower')

        assert median.shape == (H, W, 3)

        return median, frames, masks

    def _transform_frame(self, frame: np.ndarray,
                         median: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blur warped frames and median before subtraction
        """
        if self.subtraction_gaussian_ksize is not None:
            frames = torch.permute(frame, (0, 3, 1, 2))
            sigTmp = 0.3*(0.5*(self.subtraction_gaussian_ksize[0]-1)-1)+0.8
            frames = kornia.filters.gaussian_blur2d(frames,
                                self.subtraction_gaussian_ksize,(sigTmp,sigTmp))
            frames = torch.permute(frames,(0,2,3,1))
            median = median.unsqueeze(0)
            median = torch.permute(median,(0,3,1,2))
            median = kornia.filters.gaussian_blur2d(median,
                                self.subtraction_gaussian_ksize,(sigTmp,sigTmp))
            median = torch.permute(median,(0,2,3,1))
        else:
            frames = frame

        return frames, median

    def _background_subtraction(self, frame_ma: np.ndarray, median: np.ndarray,
                                threshold: float) -> torch.Tensor:
        """
        Denote [median - threshold, median + threshold] as background
        Subtract background from the warped frames
        """
        # preprocess warped frames and median
        frame_ma, median = self._transform_frame(frame_ma, median)

        ALPHA = 1e-10
        bg_sub_mask = torch.abs(frame_ma-median)
        bg_sub_mask = bg_sub_mask.mean(dim=-1).unsqueeze(dim=-1)  # avg over color dimension
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
            frames_c, inv_homo_list, (int(2*H/3), int(2*W/3)))
        frames_res = torch.permute(frames_res, (0, 2, 3, 1)).to(frames.device)

        return frames_res

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Monkey patch: https://github.com/Trusted-AI/adversarial-robustness-toolbox/issues/1943
        PyTorchObjectDetector._get_losses = _get_new_losses

        x = x.double()

        if self.prefilter_ss: #(optional)
            ssfilter_kwargs = {'kwargs': {"edge_filter_size": self.edge_filter_size,
                               "edge_sub_thres": self.edge_sub_thres,
                               "gaussian_filter_size": self.gaussian_filter_size}}
            self.prefilter = SmallScaleFilter(**ssfilter_kwargs)
            # Filter out small scale features/noise
            if(x.shape[-1] != 3):
                frames_ss = torch.permute(x,(0,2,3,1))
                frames_ss = self.prefilter._ss_filter(frames_ss)
            else:
                frames_ss = self.prefilter._ss_filter(x)

            if(x.shape[-1] != 3):
                frames_ss = torch.permute(frames_ss,(0,3,1,2))

            assert frames_ss.shape == x.shape

            # 1. warp pre-filtered frames
            frames_warped, masks, homo_list = self._warp_frame(frames_ss, ref_idx=0)
        else:
            # 1. warp frames
            frames_warped, masks, homo_list = self._warp_frame(x, ref_idx=0)

        # 2. compute median
        median, frame_ma, masks = self._compute_RGB_median(frames_warped, masks)

        # 3. apply background subtraction
        bg_sub_mask = self._background_subtraction(frames_warped, median, threshold=self.bg_sub_thre)

        # 3.1 track rgb frames
        frames_warped_bgsub = frames_warped * bg_sub_mask
        # 3.1.a Remove any remnants of adv patch left behind due to patch misalignment
        if self.postfilter_ss: #(optional)
            ssfilter_kwargs = {'kwargs': {"edge_filter_size": self.edge_filter_size,
                               "edge_sub_thres": self.edge_sub_thres,
                               "gaussian_filter_size": self.gaussian_filter_size}}
            self.postfilter = SmallScaleFilter(**ssfilter_kwargs)
            frames_warped_bgsub = self.postfilter._ss_filter(frames_warped_bgsub)

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

        frames_warped_bgsub_restored = torch.squeeze(frames_warped_bgsub_restored,dim=0)

        if(x.shape[-1] != 3):
            frames_warped_bgsub_restored = torch.permute(frames_warped_bgsub_restored,(0,3,1,2))

        assert frames_warped_bgsub_restored.shape == x.shape

        return frames_warped_bgsub_restored, y
