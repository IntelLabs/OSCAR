#
# Copyright (C) 2022 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch


def _compute_frame_indices_for_median_pixel(
    x: torch.Tensor, channel: int = 0
) -> torch.Tensor:
    """
    This is a deterministic method for computing median indices,
    it returns the first occurence when median is not unique.
    In contrast, the `torch.median` implementation
    has non-deterministic gradients when median is not unique.
    (see warning in https://pytorch.org/docs/stable/generated/torch.median.html)
    """
    K, H, W, _ = x.size()
    xc = x[:, :, :, channel].detach().cpu().numpy()
    vmedian = np.percentile(xc, 50, axis=0, interpolation="nearest")

    indices = np.empty((K, H, W), dtype=int)
    for k in range(K):
        indices[k, :, :] = k
    indices[np.where(xc != vmedian)] = K

    # get first occurence of median
    kmedian = indices.min(axis=0)
    return torch.LongTensor(kmedian).to(x.device)


def _get_bg_frame_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=0)


def _get_bg_frame_median(x: torch.Tensor) -> torch.Tensor:
    K, H, W, C = x.size()
    kmed = _compute_frame_indices_for_median_pixel(x)
    rows = torch.arange(H).repeat(W, 1).T
    cols = torch.arange(W).repeat(H, 1)
    return x[kmed, rows, cols]


def _compute_fg_mask(
    x: torch.Tensor, bg_frame: torch.Tensor, threshold: float = 10 / 255
) -> torch.Tensor:
    ALPHA = 1e-8
    K, H, W, C = x.size()
    assert bg_frame.size() == (H, W, C)

    mask = torch.abs(x - bg_frame)
    mask = mask.mean(dim=-1).unsqueeze(dim=-1)  # avg over color dimension
    mask = mask - threshold
    mask = mask / (torch.abs(mask) + ALPHA)
    mask = F.relu(mask)
    assert not mask.isnan().any()
    return mask


class BackgroundSubtraction(PreprocessorPyTorch):
    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bg_frame = _get_bg_frame_median(x)
        mask = _compute_fg_mask(x, bg_frame)
        x = mask * x
        return x, y


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

    preprocessor = BackgroundSubtraction()

    # Hack for https://github.com/Trusted-AI/adversarial-robustness-toolbox/issues/1442
    preprocessor.params.append("_device")
    preprocessor.set_params(_device="cpu")

    images = list()
    print("Loading images assuming all frames are named in a sorted order...")
    filepaths = sorted(glob.glob(args.pathname))
    for filepath in tqdm(filepaths):
        images.append(imread(filepath))

    x = np.stack(images, axis=0) / 255.0
    x = preprocessor(x)[0]
    x = np.uint8(x * 255.0)
    print("Finished preprocessing all frames.")

    print("Saving frames to disk...")
    for i, filepath in tqdm(enumerate(filepaths), total=len(filepaths)):
        filepath_ = Path(filepath)
        filepath_ = filepath_.parent / (
            filepath_.stem + args.postfix + filepath_.suffix
        )
        imwrite(str(filepath_), x[i])


if __name__ == "__main__":
    main()
