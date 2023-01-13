#
# Copyright (C) 2023 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Optional
from typing import Tuple
import kornia
import torch
import torch.nn.functional as F

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

class SmallScaleFilter(PreprocessorPyTorch):
    def __init__(self, **kwargs):
        super().__init__()

        self.edge_filter_size = (kwargs["edge_filter_size"]) if "edge_filter_size" in kwargs.keys() else 5
        self.gaussian_filter_size = (kwargs["gaussian_filter_size"]) if "gaussian_filter_size" in kwargs.keys() else 11
        self.edge_sub_thres = (kwargs["edge_sub_thres"]) if "edge_sub_thres" in kwargs.keys() else 15.0/255.0

    def _ss_filter(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Identifies regions with small/fine scale features using edge detection.
        The identified regions are masked off in the original image based on threshold.
        In effect, this forces the attacker to use larger scale coherent features
        for attack rather a fine scale whoite noise.
        """

        assert frames.max() <= 1.0, frames.max()

        ALPHA = 1e-8

        frames = torch.permute(frames,(0,3,1,2))

        # Sharpen the edge with unsharpen and detect the edge using laplace
        fsize = self.edge_filter_size
        frames_mask = kornia.filters.unsharp_mask(frames, (fsize,fsize), (2.0,2.0), border_type='reflect')
        frames_mask = kornia.filters.laplacian(frames_mask, fsize, border_type='reflect', normalized=True)
        # Clip the edge values resulting from the above convolutions to the proper range of 0 - 1
        frames_mask = torch.clamp(frames_mask,min=0.0,max=1.0)

        # Convert the edges from RGB to grayscale because we want to capture sharp gradients/edges within
        # each RGB channel and also between the channels.
        frames_mask = kornia.color.rgb_to_grayscale(frames_mask)

        # Identifiy regions with high density of edges -> regions with small scale noisy features.
        # To indentify regions and not just the edge Gaussian blurring is used.
        # In regions with high edge density, blurring with "light-up" these scmall scale regions
        # while the large scale features would still be unaffected (or only mildly affected - below threshaold)
        fsize = self.gaussian_filter_size
        sigTmp = 0.3*(0.5*(fsize-1)-1)+0.8
        frames_mask = kornia.filters.gaussian_blur2d(frames_mask,(fsize,fsize),(sigTmp,sigTmp))

        # Identify regions with edge desity above certain threshold convert into a 0-1 form to
        # indicate weather the pixel is small scale feature (1) or not a small scale feature (0)
        frames_mask = (- frames_mask + self.edge_sub_thres)
        frames_mask = frames_mask / (torch.abs(frames_mask) + ALPHA)
        frames_mask = F.relu(frames_mask)

        frames = frames*frames_mask
        frames = torch.permute(frames,(0,2,3,1))

        return frames

    def forward(self, x: torch.Tensor,
        y: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Convert float to double as Kornia expects tensors of type double
        x = x.double()

        # In Eval 6, the shape of input varies from
        # (N*H*W*C) in benign scenarios to
        # (1*C*H*W) in adversarial scaenario.
        # To handle this, we need these if and if-else statements

        if(x.shape[-1] != 3):
            frames = torch.permute(x,(0,2,3,1))
            frames = self._ss_filter(frames)
        else:
            frames = self._ss_filter(x)

        if(x.shape[-1] != 3):
            frames = torch.permute(frames,(0,3,1,2))

        # Cast the tensor back to float after Kornia operations are complete
        frames = frames.float().to(x.device)

        assert frames.shape == x.shape

        return frames, y

