#
# Copyright (C) 2020 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import torch
import numpy as np
from art.defences.preprocessor.preprocessor import Preprocessor
from oscar.utils.utils import create_model, image_to_tensor, create_inputs
from detectron2.model_zoo import get_config_file


class BackgroundAblator(Preprocessor):
    def __init__(self,
        invert_mask=False,
        perturb_whole_frame=False,
        detectron2_config_path="",
        detectron2_weights_path="",
        detectron2_score_thresh=None,
        detectron2_iou_thresh=None,
        mid_clip_only=False,
        means=None
    ):
        self.invert_mask = invert_mask
        self.perturb_whole_frame = perturb_whole_frame

        # We may only examine the middle clip of a video if on-the-fly ablation with Detectron2 is prohibitively expensie.
        self.mid_clip_only = mid_clip_only

        if torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        # Ablation on the fly using Detectron2
        if detectron2_config_path != "" and detectron2_weights_path != "":
            detectron2_config_path = get_config_file(detectron2_config_path)
            self.detectron2_model, self.detectron2_metadata = create_model(detectron2_config_path, \
                                                                           detectron2_weights_path, \
                                                                           device=self._device, \
                                                                           score_thresh=detectron2_score_thresh, \
                                                                           iou_thresh=detectron2_iou_thresh)
            self.means = torch.tensor(means, dtype=torch.float32, device=self._device).view(1,3,1,1,1)


    @property
    def apply_fit(self):
        return True


    @property
    def apply_predict(self):
        return True


    def fit(self, x, y=None, **kwargs):
        return None


    # FIXME: We should compute the grad directly without repeating pytorch_ops.
    def _estimate_gradient(self, x, grad):
        x = torch.tensor(x, device=self._device, requires_grad=True)
        # Disable enforce_binary_masks so that it is differentiable.
        x_prime = self.forward(x, enforce_binary_masks=False)
        grad = torch.tensor(grad, device=x_prime.device)
        x_prime.backward(grad)
        x_grad = x.grad.detach().cpu().numpy()
        if x_grad.shape != x.shape:
            raise Exception("The input shape is {} while the gradient shape is {}".format(x.shape, x_grad.shape))
        return x_grad


    # We only change the shape here.
    # We don't have to make adversary aware of the ablation if the mask is precomputed.
    #   The result will be the same.
    def _passthrough_gradient(self, x, grad):
        x_grad = np.zeros(x.shape, dtype=grad.dtype)
        x_grad[:,:3,:,:,:] = grad
        return x_grad


    def estimate_gradient(self, x, grad):
        if self.perturb_whole_frame:
            # For educational purpose only.
            return self._passthrough_gradient(x, grad)
        else:
            return self._estimate_gradient(x, grad)


    def _generate_masks(self, x):
        self.detectron2_model.eval()
        # Example x: torch.Size([4, 3, 16, 112, 112])
        x = (x + self.means) / 255.
        nb_clips, _nb_channels, group_size, _height, _weight = x.shape
        masks = torch.zeros((nb_clips, 1, group_size, _height, _weight), dtype=torch.float32, device=self._device)
        for i in range(nb_clips):
            for j in range(group_size):
                img = x[i,:,j,:,:]
                # Create inputs for Detectron2 model
                batched_inputs = create_inputs([img], input_format='BGR')
                # Run inference on examples
                batched_outputs = self.detectron2_model.inference(batched_inputs)

                pred_masks = batched_outputs[0]['instances'].pred_masks
                # Aggregate binary masks, resulting in {0,1} torch.int64.
                mask = torch.sum(pred_masks, dim=0)
                mask = mask.to(torch.float32)
                masks[i,0,j,:,:] = mask
        return masks


    def forward(self, x, enforce_binary_masks=True):
        if self.mid_clip_only:
            b_size = x.shape[0]
            mid_idx = int(b_size / 2)
            x = x[mid_idx:mid_idx+1,:,:,:,:]

        if x.shape[1] != 4:
            imgs = x
            masks = self._generate_masks(x)
        else:
            # ablation with pre-computed masks.
            imgs, masks = x[:,:3,:,:,:], x[:,3:,:,:,:]
            masks = masks / 255.

        if enforce_binary_masks:
            masks = masks.round()
        if self.invert_mask:
            masks = 1 - masks
        x = imgs * masks

        if self.mid_clip_only:
            # Restore the shape so that Art.Attack won't complain.
            x = x.repeat(b_size, 1, 1, 1, 1)

        return x


    def __call__(self, x, y=None):
        x = torch.tensor(x, device=self._device)
        x = self.forward(x)
        x = x.cpu().numpy()
        return x, y
