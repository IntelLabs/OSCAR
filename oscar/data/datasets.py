#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
from typing import Callable

import numpy as np

from MARS.dataset.preprocess_data import get_mean
from oscar.classifiers.ucf101_mars_lightning import MARSDataModule
from oscar.classifiers.ucf101_mars_lightning import parse_args
from armory.data import datasets
from armory.data.datasets import VideoContext

logger = logging.getLogger(__name__)


# Probably not the best way to write this...
class ArmoryDataLoader(datasets.ArmoryDataGenerator):
    def __init__(self, dataloader, context=None):
        self.dataloader = dataloader
        self._batch_size = dataloader.batch_size
        self.context = context

    def get_batch(self):
        return next(iter(self.dataloader))

    def __iter__(self):
        return iter(self.dataloader)

    def __next__(self):
        return self.get_batch()

    def __len__(self):
        return len(self.dataloader)


def _create_collate_fn(opt):
    def collate_fn(batch):
        # Extract first clip and label
        assert (len(batch) == 1)
        clip, label = batch[0]

        # Convert to numpy for Armory
        inputs = clip.numpy()

        # Convert channels first (CNHW) -> channels last (NHWC) for Armory
        assert (len(inputs.shape) == 4)
        inputs = inputs.transpose((1, 2, 3, 0))

        # Remove MARS mean-normalization for Armory
        mean = get_mean('activitynet')
        if inputs.shape[3] == 4:
            mean = np.concatenate((mean, [0]))
        inputs += mean

        # Normalize to [0, 1] for Armory
        inputs /= 255
        inputs = np.clip(inputs, 0, 1)

        # Add batch dimension
        inputs = np.expand_dims(inputs, axis=0)
        labels = np.array([label], dtype=np.int32)

        if opt.shuffle_axes:
            rng = np.random.default_rng()
            for shuffle_axis in opt.shuffle_axes:
                rng.shuffle(inputs, axis=shuffle_axis)

        return inputs, labels

    return collate_fn


def ucf101(batch_size=1, **kwargs):
    # Parse opts
    opt = parse_args([])
    # FIXME: We should really only update __dict__ and not create new keys, or at least warn if we're creating new keys.
    opt.__dict__.update(**kwargs)

    # NOTE: split is either 'train' or 'test' in Armory's kwargs; but it means something different to MARS, so we rewrite it
    #       here.
    if opt.split == 'test':
        opt.split = '1'

    # FIXME: We should probably support loading training data
    assert opt.split != 'train'

    # Load data module with Armory-style preprocessing: NHWC in [0, 1] range
    datamodule = MARSDataModule("RGB",
                                opt.frame_dir,
                                opt.annotation_path,
                                fold=int(opt.split),
                                batch_size=opt.batch_size,
                                num_workers=opt.n_workers,
                                frame_size=opt.sample_size,
                                clip_length=opt.sample_duration,
                                clip_step=opt.step_between_clips,
                                mid_clip_only=opt.mid_clip_only,
                                random_resized_crop_scale=(opt.random_resized_crop_scale_min, opt.random_resized_crop_scale_max),
                                test_indices=opt.test_index,
                                test_size=opt.test_size,
                                random_seed=opt.random_seed,
                                collate_fn=_create_collate_fn(opt))
    datamodule.setup('test')
    dataloader = datamodule.test_dataloader()

    ucf101_context = VideoContext(x_shape=(None, None, None, 3), frame_rate=25)
    dataloader = ArmoryDataLoader(dataloader, context=ucf101_context)

    return dataloader

def ucf101_clean(batch_size=1, **kwargs):
    return ucf101(batch_size=batch_size, **kwargs)

# shuffle data processed by preprocessing_fn
def shuffle_fn(x, shuffle_axes=[]):
    # np.random.shuffle only shuffles along the first axis and is recommended to be replaced by default_rng generator
    rng = np.random.default_rng()
    for shuffle_axis in shuffle_axes:
        # shuffle_axis = 0: shuffle on clip dimension
        # shuffle_axis = 1: shuffle on frame dimension
        rng.shuffle(x, axis=shuffle_axis)

    return x


# shuffle ucf101 clean dataset
# To do: UCF101 scenario needs to load dataset with "shuffle_axes" parameters; otherwise, default value will be used and shuffling will be disabled.
def ucf101_clean_shuffle(
    split: str = "train",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    cache_dataset: bool = True,
    preprocessing_fn: Callable = datasets.ucf101_canonical_preprocessing,
    framework: str = "numpy",
    shuffle_files: bool = True,
    shuffle_axes: int = [0, 1],
) -> datasets.ArmoryDataGenerator:

    if shuffle_axes:
        orig_preprocessing_fn = preprocessing_fn
        if orig_preprocessing_fn is None:
            orig_preprocessing_fn = lambda x: x
        preprocessing_fn = lambda x: shuffle_fn(orig_preprocessing_fn(x), shuffle_axes=shuffle_axes)

    return datasets.ucf101_clean(
        split=split,
        epochs=epochs,
        batch_size=batch_size,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
    )
