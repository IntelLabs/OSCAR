#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#
import logging
from collections import Counter

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import transforms as T
from torchvision.transforms import functional as TF

import pytorch_lightning as pl

from sklearn.model_selection import StratifiedShuffleSplit

from oscar.data.ucf101 import UCF101Dataset
from oscar.data.video import ClipSampler, MiddleClipSampler
from oscar.data.transforms import ExCompose, Permute, Squeeze, Unsqueeze, ExSplitLambda

from MARS.dataset.preprocess_data import get_mean

logger = logging.getLogger(__name__)


class MARSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        modality,
        frames_root,
        annotation_dir,
        fold=1,
        batch_size=16,
        num_workers=1,
        frame_size=112,
        clip_length=16,
        clip_step=1,
        mid_clip_only=False,
        random_resized_crop_scale=(0.5, 1.0),
        test_indices=None,
        test_size=0,
        random_seed=0,
        collate_fn=None,
        frame_cache_dir=None,
        train_file_patterns=["{:05d}.jpg", "TVL1jpg_x_{:05d}.jpg", "TVL1jpg_y_{:05d}.jpg"],
        test_file_patterns=["{:05d}.jpg"],
    ):
        super().__init__()

        assert modality in ['RGB', 'RGB_Flow',
                            'RGBMasked_Flow', 'RGBMasked_FlowMasked',
                            'RGBSeg_Flow',
                            'RGBSegMC_Flow',
                            'RGBSegSC_Flow']

        self.modality = modality
        self.frames_root = frames_root
        self.annotation_dir = annotation_dir
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frame_size = frame_size
        self.clip_length = clip_length
        self.clip_step = clip_step
        self.mid_clip_only = mid_clip_only
        self.random_resized_crop_scale = random_resized_crop_scale
        self.test_indices = test_indices
        self.test_size = test_size
        self.random_seed = random_seed
        self.collate_fn = collate_fn
        self.frame_cache_dir = frame_cache_dir
        self.train_file_patterns = train_file_patterns
        self.test_file_patterns = test_file_patterns

        from detectron2.data import MetadataCatalog
        self.palette = MetadataCatalog.get('coco_2017_val').thing_colors

        if 'RGBSegMC_' in self.modality:
            self.input_channels = len(self.palette) + 2 # COCO-things + XY
        elif 'RGBSegSC_' in self.modality:
            self.input_channels = 1 + 2 # Mask + XY
        else:
            self.input_channels = 3 + 2 # RGB + XY

    @classmethod
    def add_argparse_args(cls, parser):
        group = parser.add_argument_group(cls.__name__)
        group.add_argument('--modality', default='RGB', type=str, choices=['RGB', 'RGB_Flow', 'RGBMasked_Flow', 'RGBMasked_FlowMasked', 'RGBSeg_Flow', 'RGBSegMC_Flow', 'RGBSegSC_Flow'])
        group.add_argument('--dataset', default='UCF101', type=str, choices=['UCF101'])
        group.add_argument('--only_RGB', default=False, action='store_true')
        group.add_argument('--batch_size', default=32, type=int)
        group.add_argument('--frame_dir', default=None, type=str)
        group.add_argument('--annotation_path', default=None, type=str)
        group.add_argument('--frame_mask_dir', default=None, type=str)
        group.add_argument('--n_workers', default=4, type=int)
        group.add_argument('--split', default=1, type=int, choices=[1, 2, 3])
        group.add_argument('--sample_size', default=112, type=int)
        group.add_argument('--sample_duration', default=16, type=int)
        group.add_argument('--step_between_clips', default=1, type=int)
        group.add_argument('--random_resized_crop_scale_min', default=0.5, type=float)
        group.add_argument('--random_resized_crop_scale_max', default=1.0, type=float)
        group.add_argument('--test_size', default=0, type=int)
        group.add_argument('--test_index', default=None, type=int, nargs='+')
        group.add_argument('--random_seed', default=1, type=bool, help='Manually set random seed of sampling validation clip')
        group.add_argument('--mid_clip_only', default=False, type=bool)
        group.add_argument('--shuffle_axes', default=None, type=int, nargs='+')

        return parser

    def prepare_data(self):
        UCF101Dataset(self.frames_root,
                      self.annotation_dir,
                      self.train_file_patterns,
                      fold=self.fold)

    def setup(self, stage=None):
        logger.info("Setting up data module for stage: %s", stage)

        channels_mean = torch.tensor([*get_mean('activitynet'), 127.5, 127.5])

        train_channels_mean = channels_mean
        test_channels_mean = channels_mean[0:3]

        # Create robust feature transform
        robust_extractor = None
        if 'RGBMasked_' in self.modality:
            from oscar.defences.preprocessor.detectron2 import CachedDetectron2Preprocessor
            from oscar.defences.preprocessor.ablator import AblatorPyTorch

            dt2 = CachedDetectron2Preprocessor(self.frame_cache_dir)
            robust_extractor = AblatorPyTorch(channels_mean / 255, detectron2=dt2)

        elif 'RGBSeg_' in self.modality:
            from oscar.defences.preprocessor.detectron2 import CachedDetectron2Preprocessor
            from oscar.defences.preprocessor.paletted_semantic_segmentor import PalettedSemanticSegmentorPyTorch

            dt2 = CachedDetectron2Preprocessor(self.frame_cache_dir)
            robust_extractor = PalettedSemanticSegmentorPyTorch(channels_mean[0:3] / 255, detectron2=dt2, palette=self.palette)

        elif 'RGBSegMC_' in self.modality:
            from oscar.defences.preprocessor.detectron2 import CachedDetectron2Preprocessor
            from oscar.defences.preprocessor.multichannel_semantic_segmentor import MultichannelSemanticSegmentorPyTorch

            dt2 = CachedDetectron2Preprocessor(self.frame_cache_dir)
            robust_extractor = MultichannelSemanticSegmentorPyTorch(detectron2=dt2, nb_channels=len(self.palette))

            train_channels_mean = 127.5
            test_channels_mean = 127.5

        elif 'RGBSegSC_' in self.modality:
            # TODO: Create another segmentor class that is faster and selects objects relevant to UCF101
            from oscar.defences.preprocessor.detectron2 import CachedDetectron2Preprocessor
            from oscar.defences.preprocessor.multichannel_semantic_segmentor import MultichannelSemanticSegmentorPyTorch

            dt2 = CachedDetectron2Preprocessor(self.frame_cache_dir)
            robust_extractor = MultichannelSemanticSegmentorPyTorch(detectron2=dt2, nb_channels=1) # 1 channel == person mask

            train_channels_mean = 127.5
            test_channels_mean = 127.5

        # Apply robust feature extractor to RGB channels only if not _FlowMasked
        if robust_extractor is not None and '_FlowMasked' not in self.modality:
            robust_extractor = ExSplitLambda(robust_extractor, 3, 0, dim=-1)

        robust_transform = ExCompose([
            T.Normalize(0, 255), # [0, 255] -> [0, 1]
            Permute(0, 2, 3, 1), # TCHW -> THWC
            Unsqueeze(0), # THWC -> NTHWC
            robust_extractor, # Apply robust feature extractor
            Squeeze(0), # NTHWC -> THWC
            Permute(0, 3, 1, 2), # THWC -> TCHW
            T.Normalize(0, 1/255), # [0, 1] -> [0, 255]
        ])

        # Train transform
        # FIXME: Don't load flow when modality does not specify _Flow!
        # FIXME: Is there a way to decouple rgb and flow datasets like we did above?
        #        The problem is they need to be synchronized somehow.
        train_transform = ExCompose([
            robust_transform,
            T.RandomResizedCrop(self.frame_size, scale=self.random_resized_crop_scale, ratio=(1., 1.)), # Crop then Resize
            T.RandomApply([TF.hflip, ExSplitLambda(T.Normalize(255, -1), 1, -2, dim=-1)]), # Horizontal flip and invert x-flow randomly
            T.Normalize(train_channels_mean, 1), # [0, 255] -> ~[-128, 128]
            Permute(1, 0, 2, 3), # TCHW -> CTHW
        ])
        train_sampler = ClipSampler(self.clip_length, self.clip_step)

        # Test transform
        test_transform = ExCompose([
            robust_transform,
            T.Resize(self.frame_size),
            T.CenterCrop(self.frame_size),
            T.Normalize(test_channels_mean, 1), # [0, 255] -> ~[-128, 128]
            Permute(1, 0, 2, 3), # TCHW -> CTHW
        ])

        test_sampler = range
        if self.mid_clip_only:
            test_sampler = MiddleClipSampler(self.clip_length, self.clip_step)

        if stage == 'fit' or stage is None:
            logger.info("Loading training data...")
            self.train_dataset = UCF101Dataset(self.frames_root,
                                               self.annotation_dir,
                                               self.train_file_patterns,
                                               train=True,
                                               fold=self.fold,
                                               transform=train_transform,
                                               sampler=train_sampler)
            logger.info("train data = %d", len(self.train_dataset))

            logger.info("Loading validation data...")
            self.val_dataset = UCF101Dataset(self.frames_root,
                                             self.annotation_dir,
                                             self.test_file_patterns,
                                             train=False,
                                             fold=self.fold,
                                             transform=test_transform,
                                             sampler=train_sampler)
            logger.info("val data = %d", len(self.val_dataset))

        if stage == 'test' or stage is None:
            logger.info("Loading test data...")
            test_dataset = UCF101Dataset(self.frames_root,
                                         self.annotation_dir,
                                         self.test_file_patterns,
                                         train=False,
                                         fold=self.fold,
                                         transform=test_transform,
                                         sampler=test_sampler)

            # Select test indices...
            if self.test_indices is not None:
                logger.info("Selecting data indices: %s", self.test_indices)
                test_dataset = torch.utils.data.Subset(test_dataset, self.test_indices)

            # ...or subsample test_dataset using a stratified split of test_size elements.
            elif self.test_size > 0:
                y = test_dataset.targets
                if test_dataset.target_transform is not None:
                    y_transform = [test_dataset.target_transform(y_) for y_ in y]

                sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_seed)
                _, indices = next(sss.split(y, y_transform))

                y_selected = [y[i] for i in indices]
                logger.info("Stratified subsampling test dataset to %d samples: %s", self.test_size, Counter(y_selected))

                test_dataset = torch.utils.data.Subset(test_dataset, indices)

            self.test_dataset = test_dataset
            logger.info("test data = %d", len(self.test_dataset))


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1, # Must be 1 because we can't batch whole videos
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=False,
                          collate_fn=self.collate_fn)
