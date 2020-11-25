"""
Model contributed by: MITRE Corporation
Adapted from: https://github.com/craston/MARS
Intel modified to add model_kwargs to opts
"""
import logging
from typing import Union, Optional, Tuple, Any

from art.classifiers import PyTorchClassifier
import numpy as np
import torch
from torch import optim

from MARS.opts import parse_opts
from MARS.models.model import generate_model

from armory.baseline_models.pytorch.ucf101_mars import DEVICE, preprocessing_fn, MEAN, STD

logger = logging.getLogger(__name__)

def preprocessing_fn_torch(
    batch: Union[torch.Tensor, np.ndarray],
    consecutive_frames: int = 16,
    scale_first: bool = True,
    align_corners: bool = False,
    mean: np.ndarray = MEAN,
    std: np.ndarray = STD,
):
    """
    inputs - batch of videos each with shape (frames, height, width, channel)
    outputs - batch of videos each with shape (n_stack, channel, stack_frames, new_height, new_width)
        frames = n_stack * stack_frames (after padding)
        new_height = new_width = 112
    consecutive_frames - number of consecutive frames (stack_frames)

    After resizing, a center crop is performed to make the image square

    This is a differentiable alternative to MARS' PIL-based preprocessing.
        There are some
    """
    if not isinstance(batch, torch.Tensor):
        logger.warning(f"batch {type(batch)} is not a torch.Tensor. Casting")
        batch = torch.from_numpy(batch).to(DEVICE)
        # raise ValueError(f"batch {type(batch)} is not a torch.Tensor")
    if batch.dtype != torch.float32:
        raise ValueError(f"batch {batch.dtype} should be torch.float32")
    if batch.shape[0] != 1:
        raise ValueError(f"Batch size {batch.shape[0]} != 1")
    video = batch[0]

    if video.ndim != 4:
        raise ValueError(
            f"video dims {video.ndim} != 4 (frames, height, width, channel)"
        )
    if video.shape[0] < 1:
        raise ValueError("video must have at least one frame")
    if tuple(video.shape[1:3]) == (240, 320):
        standard_shape = True
    elif tuple(video.shape[1:3]) == (226, 400):
        logger.warning("Expected odd example shape (226, 400, 3)")
        standard_shape = False
    else:
        raise ValueError(f"frame shape {tuple(video.shape[1:])} not recognized")
    if video.max() > 1.0 or video.min() < 0.0:
        raise ValueError("input should be float32 in [0, 1] range")
    if not isinstance(consecutive_frames, int):
        raise ValueError(f"consecutive_frames {consecutive_frames} must be an int")
    if consecutive_frames < 1:
        raise ValueError(f"consecutive_frames {consecutive_frames} must be positive")

    # Select a integer multiple of consecutive frames
    while len(video) < consecutive_frames:
        # cyclic pad if insufficient for a single stack
        video = torch.cat([video, video[: consecutive_frames - len(video)]])
    if len(video) % consecutive_frames != 0:
        # cut trailing frames
        video = video[: len(video) - (len(video) % consecutive_frames)]

    if scale_first:
        # Attempts to directly follow MARS approach
        # (frames, height, width, channel) to (frames, channel, height, width)
        video = video.permute(0, 3, 1, 2)
        if standard_shape:  # 240 x 320
            sample_height, sample_width = 112, 149
        else:  # 226 x 400
            video = video[:, :, 1:-1, :]  # crop top/bottom pixels, reduce by 2
            sample_height, sample_width = 112, 200

        video = torch.nn.functional.interpolate(
            video,
            size=(sample_height, sample_width),
            mode="bilinear",
            align_corners=align_corners,
        )

        if standard_shape:
            crop_left = 18  # round((149 - 112)/2.0)
        else:
            crop_left = 40
        video = video[:, :, :, crop_left : crop_left + sample_height]

    else:
        # More efficient, but not MARS approach
        # Center crop
        sample_size = 112
        if standard_shape:
            crop_width = 40
        else:
            video = video[:, 1:-1, :, :]
            crop_width = 88
        video = video[:, :, crop_width:-crop_width, :]

        # Downsample to (112, 112) frame size
        # (frames, height, width, channel) to (frames, channel, height, width)
        video = video.permute(0, 3, 1, 2)
        video = torch.nn.functional.interpolate(
            video,
            size=(sample_size, sample_size),
            mode="bilinear",
            align_corners=align_corners,
        )

    if video.max() > 1.0:
        raise ValueError("Video exceeded max after interpolation")
    if video.min() < 0.0:
        raise ValueError("Video under min after interpolation")

    # reshape into stacks of frames
    video = torch.reshape(video, (-1, consecutive_frames) + video.shape[1:])

    # transpose to (stacks, channel, stack_frames, height, width)
    video = video.permute(0, 2, 1, 3, 4)
    # video = torch.transpose(video, axes=(0, 4, 1, 2, 3))

    # normalize before changing channel position?
    video = torch.transpose(video, 1, 4)
    video = ((video * 255) - torch.from_numpy(mean).to(DEVICE)) / torch.from_numpy(
        std
    ).to(DEVICE)
    video = torch.transpose(video, 4, 1)

    return video


def make_model(
    model_status: str = "ucf101_trained", weights_path: Optional[str] = None, **model_kwargs
) -> Tuple[torch.nn.DataParallel, optim.SGD]:
    statuses = ("ucf101_trained", "kinetics_pretrained")
    if model_status not in statuses:
        raise ValueError(f"model_status {model_status} not in {statuses}")
    trained = model_status == "ucf101_trained"
    if not trained and weights_path is None:
        raise ValueError("weights_path cannot be None for 'kinetics_pretrained'")

    opt = parse_opts(arguments=[])
    opt.dataset = "UCF101"
    opt.only_RGB = True
    opt.log = 0
    opt.batch_size = 1
    opt.arch = f"{opt.model}-{opt.model_depth}"

    if trained:
        opt.n_classes = 101
    else:
        opt.n_classes = 400
        opt.n_finetune_classes = 101
        opt.batch_size = 32
        opt.ft_begin_index = 4

        opt.pretrain_path = weights_path

    # Merge model_kwargs from config into MARS's opt
    logger.info("Merging opts from config..")
    for key, value in model_kwargs.items():
        if not hasattr(opt, key):
            logger.info(f"  Skipping {key} = {value}")
        old_value = getattr(opt, key)
        logger.info(f"  opt.{key} = {value} (was {old_value})")
        setattr(opt, key, value)

    logger.info(f"Loading model... {opt.model} {opt.model_depth}")
    model, parameters = generate_model(opt)

    if trained and weights_path is not None:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

    # Initializing the optimizer
    if opt.pretrain_path:
        opt.weight_decay = 1e-5
        opt.learning_rate = 0.001
    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov,
    )

    return model, optimizer


class OuterModel(torch.nn.Module):
    def __init__(
        self,
        weights_path: Optional[str],
        max_frames: Optional[int] = 0,
        preprocessing_consecutive_frames: Optional[int] = 16,
        preprocessing_scale_first: Optional[bool] = True,
        preprocessing_align_corners: Optional[bool] = False,
        preprocessing_mean: Optional[Union[list, np.ndarray]] = MEAN,
        preprocessing_std: Optional[Union[list, np.ndarray]] = STD,
        **model_kwargs,
    ):
        """
        Max frames is the maximum number of input frames.
            If max_frames == 0, False, no clipping is done
            Else if max_frames > 0, frames are clipped to that number.
            This can be helpful for smaller memory cards.
        """
        super().__init__()
        max_frames = int(max_frames)
        if max_frames < 0:
            raise ValueError(f"max_frames {max_frames} cannot be negative")
        self.max_frames = max_frames
        self.model, self.optimizer = make_model(
            weights_path=weights_path, **model_kwargs
        )

        # Preprocessing-related options
        self.preprocessing_consecutive_frames = preprocessing_consecutive_frames
        self.preprocessing_scale_first = preprocessing_scale_first
        self.preprocessing_align_corners = preprocessing_align_corners
        self.preprocessing_mean = np.array(preprocessing_mean, dtype=np.float32)
        self.preprocessing_std = np.array(preprocessing_std, dtype=np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_frames:
            x = x[:, : self.max_frames]

        if self.training:
            # Use preprocessing_fn_numpy in dataset preprocessing
            return self.model(x)
        else:
            x = preprocessing_fn_torch(x, self.preprocessing_consecutive_frames,
                                          self.preprocessing_scale_first,
                                          self.preprocessing_align_corners,
                                          self.preprocessing_mean,
                                          self.preprocessing_std)
            stack_outputs = self.model(x)
            output = stack_outputs.mean(axis=0, keepdims=True)

        return output


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = OuterModel(weights_path=weights_path, **model_kwargs)
    model.to(DEVICE)

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=model.optimizer,
        input_shape=(None, 240, 320, 3),
        nb_classes=101,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
