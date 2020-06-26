#
# Copyright (C) 2020 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import numpy as np
import torch
from art.classifiers import PyTorchClassifier
from armory.baseline_models.pytorch.ucf101_mars import make_model, DEVICE
from MARS.dataset.preprocess_data import get_mean as get_ucf101_mean
from art.utils import check_and_transform_label_format
import torch.nn.functional as F


# This is a lite preprocessing function for the data pipeline.
# We group video frames into clips of 16 consequtive frames here.
#   so that art.Attack won't complain the shape of MARS' prediction.
# Further, we only take the middle clip of a video so that slow defenses
#   run in reasonable time.
# Return (nb_clips, nb_channels, clip_size, height, width)
# TODO: Pad to 16 frames if video is too short.
# FIXME: Armory's scenario cannot configure this function.
def preprocessing_fn(inputs, clip_size=16, mid_clip_only=False):
    for i in range(len(inputs)):
        x = inputs[i]

        # Remove at most (clip_size-1) frames
        nb_frames, height, width, nb_channels = x.shape
        nb_clips = int(nb_frames / clip_size)
        nb_frames_removed = nb_frames % clip_size
        if nb_frames > clip_size and nb_frames_removed > 0:
            x = x[:-nb_frames_removed]

        # We only take the middle clip of 16 consecutive frames from a video
        #  so that Detectron2 on-the-fly is not too slow.
        # We have to do this in the data pipeline, because the Armory scenario
        #   let art.attack get the shape of x directly.
        if mid_clip_only:
            mid_clip_idx = int(nb_clips / 2)
            starting_frame_idx = mid_clip_idx * clip_size
            x = x[starting_frame_idx:starting_frame_idx+clip_size]
            nb_clips = 1

        # Reshape as several clips of 16 consequtive frames.
        x_clips = x.reshape(nb_clips, clip_size, height, width, nb_channels)
        x_clips = np.transpose(x_clips, (0,4,1,2,3))
        # Keep uint8 to indicate non-normalization.
        inputs[i] = x_clips
    return inputs


# A MARS preprocessing layer which resizes and normalizes images.
class MarsPreprocess(torch.nn.Module):
    def __init__(self, frame_size, means):
        super().__init__()
        self.frame_size = frame_size
        self.register_buffer('means', torch.tensor(means, dtype=torch.float32).view(1,3,1,1,1))


    def forward(self, x):
        frame_size = self.frame_size
        # Input: (nb_clips, nb_channels, clip_size, height, width) [0, 255]
        # Output:(nb_clips, nb_channels, clip_size, frame_size, frame_size), normalized.
        nb_clips, nb_channels, clip_size, height, width = x.shape
        x_nschw = x.permute(0,2,1,3,4)

        # Resize the shorter edge to 112 while keeping the spatial ratio
        if width < height:
            width_out = frame_size
            height_out = int(frame_size * height / width)
        else:
            height_out = frame_size
            width_out = int(frame_size * width / height)

        # Degrouping as NCHW.
        x_nchw = x_nschw.reshape(nb_clips * clip_size, nb_channels, height, width)
        # Resizing
        x_nchw = F.interpolate(x_nchw, size=(height_out, width_out), mode="bilinear", align_corners=True)
        # Central crop to 112x112
        if width_out != frame_size and height_out == frame_size:
            width_start = (width_out - frame_size) // 2
            x_nchw = x_nchw[:,:,:,width_start:width_start+frame_size]
        elif height_out != frame_size and width_out == frame_size:
            height_start = (height_out - frame_size) // 2
            x_nchw = x_nchw[:,:,height_start:height_start+frame_size,:]
        else:
            raise Exception("Unusal situation that we have {:d}x{:d} image after resizeing".format(width_out, height_out))

        # Regrouping as 16-frame-clips.
        x_clips = x_nchw.view(nb_clips, clip_size, nb_channels, frame_size, frame_size)

        # Moved the channel dimension to the second.
        x_clips = x_clips.permute(0,2,1,3,4)

        # Normalization RGB with mean. Alpha with zero-mean.
        x_clips[:,:3,:,:,:] -= self.means
        return x_clips

    def extra_repr(self):
        return 'MARS preprocessing which resizes images to {}x{} and normalizes by means {}'.format(
            self.frame_size,
            self.frame_size,
            self.means.squeeze()
        )


# Run predict() in one batch and no_grad().
#  Note: The batch_size is ignored, so no need to change the scenario definition.
class EfficientPyTorchClassifier(PyTorchClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction in one batch and no_grad().
        x_pth = torch.from_numpy(x_preprocessed)
        with torch.no_grad():
            x_cuda = x_pth.to(self._device)
            results_cuda = self._model(x_cuda)
            results = results_cuda[0].detach().cpu().numpy()

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions


# Make clip_values works with RGBA input.
# FIXME: Make a switch between 3 and 4 channels so that the model
#        also works with the official data pipeline which gives 3-channel input.
def get_art_model(model_kwargs, wrapper_kwargs, weights_file):
    # FIXME: We may find a better place for clipping parameters outside this function.
    preprocess_input_channels = model_kwargs.pop('preprocess_input_channels', 3)

    activity_means = get_ucf101_mean('activitynet')
    if preprocess_input_channels == 4:
        activity_means += [0.]
    activity_means = np.array(activity_means, dtype=np.float32)

    # The input_shape of MARS is fixed, no matter where we perform pre-processing.
    # But the clip_values for attack should match the return of preprocessing_fn().
    # We broadcast the shape since v_PommelHorse_g05_c03 is 400x226 instead of 320x240.
    clip_values = (0., 255.)

    model, optimizer = make_model(weights_file=weights_file, **model_kwargs)

    # Insert MARS preprocessing into the head of the model.
    mars_preprocess_layer = MarsPreprocess(frame_size=112, means=get_ucf101_mean('activitynet'))
    model.module.conv1 = torch.nn.Sequential(mars_preprocess_layer, model.module.conv1)

    model.to(DEVICE)

    wrapped_model = EfficientPyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(3, 16, 112, 112),
        nb_classes=101,
        **wrapper_kwargs,
        clip_values=clip_values,
    )
    return wrapped_model
