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
from torch import nn

from MARS.opts import parse_opts
from MARS.models.model import generate_model
import math
import MARS.models.model

from armory.baseline_models.pytorch.ucf101_mars import DEVICE, preprocessing_fn, MEAN, STD

logger = logging.getLogger(__name__)

# XXX: Copied from MARS with slight modification
class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400,
                 input_channels=3,
                 output_layers=[]):
        logger.info("Loading modified ResNeXt model...")
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        # XXX: Begin Modification
        self.conv0 = None
        self.bn0 = None
        if input_channels > 3:
            self.conv0 = nn.Conv3d(
                input_channels,
                3,
                kernel_size=1,
                bias=False)
            self.bn0 = nn.BatchNorm3d(3)
            input_channels = 3
        # XXX: End Modification

        self.conv1 = nn.Conv3d(
            input_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        #layer to output on forward pass
        self.output_layers = output_layers

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        #pdb.set_trace()
        # XXX: Begin Modification
        if self.conv0 is not None:
            x = self.conv0(x)
            x = self.bn0(x)
            x = self.relu(x)
        # XXX: End Modification
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)

        x6 = x5.view(x5.size(0), -1)
        x7 = self.fc(x6)

        if len(self.output_layers) == 0:
            return x7
        else:
            out = []
            out.append(x7)
            for i in self.output_layers:
                if i == 'avgpool':
                    out.append(x6)
                if i == 'layer4':
                    out.append(x4)
                if i == 'layer3':
                    out.append(x3)

        return out

    def freeze_batch_norm(self):
        for name,m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d): # PHIL: i Think we can write just  "if isinstance(m, nn._BatchNorm)
                m.eval() # use mean/variance from the training
                m.weight.requires_grad = False
                m.bias.requires_grad = False

# XXX: Monkey patch MARS's implementation with our modified implementation ResNeXt
MARS.models.model.resnext.ResNeXt = ResNeXt

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
        # XXX: PyTorchClassifier always puts x onto the specified device. So we tell it to leave x on the CPU and
        #      manually convert it to the GPU. However, we need to make sure the model is on the GPU.
        self.model.to('cuda')

        assert len(x.shape) == 5
        assert x.shape[0] == 1

        batched_outputs = []

        for i in range(0, min(x.shape[1], self.max_frames), self.preprocessing_consecutive_frames):
            # XXX: Manually move batch onto GPU
            x_batch = x[:, i:i + self.preprocessing_consecutive_frames, :, :, :].to('cuda')

            x_pre = preprocessing_fn_torch(x_batch, self.preprocessing_consecutive_frames,
                                                    self.preprocessing_scale_first,
                                                    self.preprocessing_align_corners,
                                                    self.preprocessing_mean,
                                                    self.preprocessing_std)

            outputs = self.model(x_pre)

            batched_outputs.append(outputs)

        batched_outputs = torch.cat(batched_outputs)
        output = batched_outputs.mean(axis=0, keepdims=True)

        return output


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = OuterModel(weights_path=weights_path, **model_kwargs)
    model.to(DEVICE) # XXX: I think this is useless since PyTorchClassifier will put the model onto the specified device_type below

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=model.optimizer,
        input_shape=(None, 240, 320, 3),
        nb_classes=101,
        clip_values=(0.0, 1.0),
        device_type='cpu', # XXX: All inputs will live on CPU and we will move them to the GPU as necessary
        **wrapper_kwargs,
    )
    return wrapped_model
