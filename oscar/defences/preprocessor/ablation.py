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
        ignore_mask_gradient=False,
        detectron2_config_path="",
        detectron2_weights_path="",
        detectron2_score_thresh=None,
        detectron2_iou_thresh=None,
        is_input_normalized=True,
        means=None
    ):
        self.invert_mask = invert_mask
        self.ignore_mask_gradient = ignore_mask_gradient

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

        self.is_input_normalized = is_input_normalized

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
        if self.ignore_mask_gradient:
            # We do this because _estimate_gradient() is not effective on Detectron2 yet.
            return self._passthrough_gradient(x, grad)
        else:
            return self._estimate_gradient(x, grad)


    def _generate_masks(self, x):
        self.detectron2_model.eval()
        # Example x: torch.Size([4, 3, 16, 112, 112]) [0,255]
        x = x / 255.

        if x.max() > 1 or x.min() < 0:
            raise Exception("Expect x in [0, 1], but it is in [%s, %s]" % (x.min(), x.max()))

        nb_clips, _nb_channels, group_size, _height, _weight = x.shape
        masks = torch.zeros((nb_clips, 1, group_size, _height, _weight), dtype=torch.float32, device=self._device)
        for i in range(nb_clips):
            # Swap clips and channels so we can feed these clips in all at once. Contribution from Cory.
            imgs = x[i,:,:,:,:].permute((1, 0, 2, 3))
            # Create inputs for Detectron2 model
            batched_inputs = create_inputs(imgs, input_format='BGR')
            # Run inference on examples
            batched_outputs = self.detectron2_model.inference(batched_inputs)

            batched_masks = []
            for outputs in batched_outputs:
                # No object detected, then no ablation.
                # FIXME: It might end up no detection after centrol cropping later.
                #         Should have an area threshold here.
                if len(outputs['instances']) == 0:
                    height, width = outputs['instances'].image_size
                    mask = torch.ones((height, width), dtype=torch.bool, device=self._device)
                else:
                    mask = outputs['instances'].pred_masks.any(dim=0)
                batched_masks.append(mask)

            batched_masks_stack = torch.stack(batched_masks)
            masks[i, 0, :, :, :] = batched_masks_stack
        return masks


    def forward(self, x, enforce_binary_masks=True):
        # Input x: (nb_clips, nb_channels, clip_size, height, width)
        #   Example torch.Size([1, 3, 16, 240, 320])

        # The MARS pipeline gives float normalized inputs;
        # while our lite pipeline gives uint8 unnormalized inputs.
        if self.is_input_normalized:
            x = x + self.means

        nb_channels = x.shape[1]
        if nb_channels != 3 and nb_channels !=4:
            raise Exception("Unusual input shape: ", x.shape)

        # No pre-computed masks from the data pipeline.
        # Run detectron2 on raw frames or 112x112 frames.
        if nb_channels != 4:
            # Detectron2 expects unnormalized input.
            masks = self._generate_masks(x)
            # Make the scale of masks and imgs consistent with that of precomputed.
            masks = masks * 255.
            imgs = x - self.means
            x = torch.cat((imgs, masks), dim=1)

        # Ablation with masks.
        # imgs are normalized, while masks are not.
        imgs, masks = x[:,:3,:,:,:], x[:,3:,:,:,:]
        masks = masks / 255.

        if enforce_binary_masks:
            masks = masks.round()
        if self.invert_mask:
            masks = 1 - masks
        # The ablated pixels become zero in the normalized reprentation, i.e. the mean color.
        x = imgs * masks

        # The output is not normalized if the input is not.
        if not self.is_input_normalized:
            x = x + self.means

        return x


    def __call__(self, x, y=None):
        x = torch.tensor(x, device=self._device)
        x = self.forward(x)
        x = x.cpu().numpy()
        return x, y
