#!/usr/bin/env python3
#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import argparse
import logging
import coloredlogs
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from tqdm import tqdm, trange
from oscar.defences.preprocessor.detectron2 import Detectron2PreprocessorPyTorch
from oscar.data.ucf101 import UCF101DataModule

logger = logging.getLogger(__name__)


class LitDetectron2(pl.LightningModule):
    def __init__(self, cache_dir, config, weights, batch_size=8):
        super().__init__()

        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.model = Detectron2PreprocessorPyTorch(config, weights)

        # Make parameterss require gradients otherwise we can't use DDP
        for param in self.model.parameters():
            param.requires_grad_()

    def forward(self, x, **kwargs):
        # NCHW -> NHWC; [0, 255] -> [0, 1]
        x = x.permute(0, 2, 3, 1) / 255

        y_preds = []

        for i in range(0, len(x), self.batch_size):
            x_batch = x[i:i+self.batch_size]

            y_batch = self.model(x_batch)

            for y in y_batch:
                y_preds.append(y.to('cpu'))

        return y_preds

    def test_step(self, batch, batch_idx):
        assert len(batch) == 1
        sample, y = batch[0]

        parent = sample['parent']
        indices = sample['indices']
        video = sample['x']

        predictions = self.forward(video)

        return {'parent': parent, 'indices': indices, 'predictions': predictions}

    def test_step_end(self, output_results):
        parent = output_results['parent']
        indices = output_results['indices']
        predictions = output_results['predictions']
        list_of_instances = list(map(instances_to_dict, predictions))

        assert len(indices) == len(predictions) == len(list_of_instances)

        sample_dir = self.cache_dir / parent.parent.name
        sample_dir.mkdir(parents=True, exist_ok=True)

        sample_path = sample_dir / (str(parent.name) + '.npz')
        np.savez_compressed(sample_path, instances=list_of_instances)

    @classmethod
    def add_argparse_args(cls, parser):
        group = parser.add_argument_group(cls.__name__)
        group.add_argument("--cache_dir", type=str, required=True)
        group.add_argument("--batch_size", type=int, default=8)

        return parser

def instances_to_dict(instances):
    d = {}
    d['image_size'] = instances.image_size
    d['fields'] = {}
    for k, v in instances.get_fields().items():
        d['fields'][k] = v.to('cpu')

    return d

def main(args):
    model = LitDetectron2(args.cache_dir, args.config, args.weights, args.batch_size)
    # FIXME: We should really use MARSDataModule
    datamodule = UCF101DataModule(args.frames_root,
                                  args.annotation_dir,
                                  train=args.train,
                                  num_workers=args.num_workers)

    trainer = pl.Trainer.from_argparse_args(args, benchmark=True)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    coloredlogs.install(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitDetectron2.add_argparse_args(parser)
    parser = UCF101DataModule.add_argparse_args(parser)
    parser.add_argument("--config", type=str, default="detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    parser.add_argument("--weights", type=str, default="detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl")

    args = parser.parse_args()

    logger.debug(args)

    main(args)
