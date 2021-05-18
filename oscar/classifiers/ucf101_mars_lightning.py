#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#
import os
import logging
import coloredlogs
import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from MARS.models.resnext import resnet101, get_fine_tuning_parameters

from oscar.data.mars import MARSDataModule

logger = logging.getLogger(__name__)

class Backbone(pl.LightningModule):
    def __init__(self, backbone=resnet101, checkpoint=None, **kwargs):
        super().__init__()

        self.clip_duration = kwargs['sample_duration']

        # "module" because pre-trained checkpoints use that name
        self.module = backbone(**kwargs)

        if checkpoint is not None:
            logger.info("Loading checkpoint: %s", checkpoint)

            state_dict = torch.load(checkpoint)['state_dict']

            # Ignore fully-connected layer parameters if they don't match shape.
            # This happens when the number of classes is different.
            if state_dict['module.fc.weight'].shape != self.module.fc.weight.shape:
                logger.warning("Ignoring module.fc.weight in: %s", checkpoint)
                state_dict['module.fc.weight'] = self.module.fc.weight

            if state_dict['module.fc.bias'].shape != self.module.fc.bias.shape:
                logger.warning("Ignoring module.fc.bias in: %s", checkpoint)
                state_dict['module.fc.bias'] = self.module.fc.bias

            self.load_state_dict(state_dict)

    def forward(self, x):
        return self.module(x)


class LitMARS(pl.LightningModule):
    def __init__(
        self,
        backbone_rgb,
        backbone_flow=None,
        lr=0.1, momentum=0.9, dampening=0.9, weight_decay=1e-3, nesterov=False,
        patience=10, cooldown=0, min_lr=0, factor=0.1,
        alpha=50,
        ft_begin_index=0
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone_rgb = Backbone(**backbone_rgb)
        self.criterion_rgb = nn.CrossEntropyLoss()

        self.backbone_flow = None
        if backbone_flow is not None:
            self.backbone_flow = Backbone(**backbone_flow)
            self.backbone_flow.freeze()
            self.criterion_flow = nn.MSELoss()

        self.acc_train = pl.metrics.Accuracy()
        self.acc_val = pl.metrics.Accuracy()
        self.acc_test = pl.metrics.Accuracy()
        self.acc_test_top5 = pl.metrics.Accuracy(top_k=5)

    @classmethod
    def add_argparse_args(cls, parser):
        group = parser.add_argument_group(cls.__name__)
        group.add_argument('--lr_patience', default=10, type=int)
        group.add_argument('--n_classes', default=101, type=int)
        group.add_argument('--n_finetune_classes', default=101, type=int)
        group.add_argument('--ft_begin_index', default=0, type=int)
        group.add_argument('--learning_rate', default=0.1, type=float)
        group.add_argument('--momentum', default=0.9, type=float)
        group.add_argument('--dampening', default=0.9, type=float)
        group.add_argument('--weight_decay', default=0.001, type=float)
        group.add_argument('--nesterov', default=False, action='store_true')
        group.add_argument('--optimizer', default='sgd', type=str, choices=['sgd'])
        group.add_argument('--cooldown', default=0, type=int)
        group.add_argument('--min_lr', default=0., type=float)
        group.add_argument('--lr_reduce_factor', default=0.1, type=float)
        group.add_argument('--MARS_alpha', default=50, type=float)
        group.add_argument('--output_layers', type=str, action='append')
        group.add_argument('--resume_path1', default=None, type=str)

        return parser

    def forward(self, x):
        logits, _ = self.backbone_rgb(x)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch # x = NCTHW
        x_rgb = x[:, :-2, :, :, :]
        x_flow = x[:, -2:, :, :, :]

        # Extract features from RGB backbone and compute loss
        logits, feats_rgb = self.backbone_rgb(x_rgb)
        scores = F.softmax(logits, dim=-1)
        loss_rgb = self.criterion_rgb(logits, y)

        # Extract features from Flow backbone and compute loss
        loss_flow = 0
        if self.backbone_flow is not None:
            _, feats_flow = self.backbone_flow(x_flow)

            loss_flow = self.criterion_flow(feats_rgb, feats_flow)

        # Compute weighted total loss
        loss = loss_rgb + loss_flow*self.hparams.alpha

        # Compute training accuracy
        self.acc_train(scores, y)

        # Log metrics
        self.log('train/epoch_loss_rgb', loss_rgb, on_step=False, on_epoch=True)
        self.log('train/epoch_loss_flow', loss_flow, on_step=False, on_epoch=True)
        self.log('train/epoch_acc', self.acc_train, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/epoch_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_rgb, y = batch

        # FIXME: We should compute all losses!
        logits, _ = self.backbone_rgb(x_rgb)
        scores = F.softmax(logits, dim=-1)
        loss = self.criterion_rgb(logits, y)

        self.acc_val(scores, y)

        self.log('val/epoch_acc', self.acc_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/epoch_loss', loss, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x_rgb, y = batch

        # Compute average logits of each 16-frame clip
        logits = []
        for clip in torch.split(x_rgb, self.backbone_rgb.clip_duration, dim=2):
            # FIXME: Everything appears to work even when you remove this if-statement...why?!
            if clip.shape[2] != self.backbone_rgb.clip_duration:
                continue
            clip_logits, _ = self.backbone_rgb(clip)
            logits.append(clip_logits)
        logits = torch.cat(logits)
        logits = torch.mean(logits, dim=0, keepdim=True)
        scores = F.softmax(logits, dim=-1)

        # Compute metrics
        loss = self.criterion_rgb(logits, y)

        self.acc_test(scores, y)
        self.acc_test_top5(scores, y)

        # Log metrics
        self.log('test/epoch_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/epoch_acc', self.acc_test, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/epoch_acc_top5', self.acc_test_top5, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        parameters = get_fine_tuning_parameters(self.backbone_rgb, self.hparams.ft_begin_index)
        optimizer = SGD(parameters,
                        lr=self.hparams.lr,
                        momentum=self.hparams.momentum,
                        dampening=self.hparams.dampening,
                        weight_decay=self.hparams.weight_decay,
                        nesterov=self.hparams.nesterov)

        scheduler = ReduceLROnPlateau(optimizer, 'max',
                                      patience=self.hparams.patience,
                                      cooldown=self.hparams.cooldown,
                                      min_lr=self.hparams.min_lr,
                                      factor=self.hparams.factor)

        lr_scheduler = { 'scheduler': scheduler,
                         'name': f"train/lr-{optimizer.__class__.__name__}",
                         'monitor': 'val/epoch_acc' }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


def main(args):
    pl.seed_everything(args.manual_seed)

    assert args.dataset in ['UCF101']

    # Load dataset
    datamodule = MARSDataModule(args.modality,
                                args.frame_dir,
                                args.annotation_path,
                                fold=int(args.split),
                                batch_size=args.batch_size,
                                num_workers=args.n_workers,
                                frame_size=args.sample_size,
                                clip_length=args.sample_duration,
                                clip_step=args.step_between_clips,
                                mid_clip_only=args.mid_clip_only,
                                random_resized_crop_scale=(args.random_resized_crop_scale_min, args.random_resized_crop_scale_max),
                                frame_cache_dir=args.frame_mask_dir)

    # Load model(s)
    backbone_rgb_kwargs = { 'checkpoint': None,
                            'input_channels': datamodule.input_channels - 2,
                            'num_classes': args.n_classes,
                            'sample_size': args.sample_size,
                            'sample_duration': args.sample_duration,
                            'output_layers': ['avgpool'], }

    backbone_flow_kwargs = None
    if '_Flow' in args.modality:
        backbone_flow_kwargs =  { 'checkpoint': args.resume_path1,
                                  'input_channels': 2,
                                  'num_classes': args.n_classes,
                                  'sample_size': args.sample_size,
                                  'sample_duration': args.sample_duration,
                                  'output_layers': ['avgpool'], }

    model = LitMARS(backbone_rgb_kwargs,
                    backbone_flow_kwargs,
                    lr=args.learning_rate,
                    momentum=args.momentum,
                    dampening=args.dampening,
                    weight_decay=args.weight_decay,
                    nesterov=args.nesterov,
                    patience=args.lr_patience,
                    cooldown=args.cooldown,
                    min_lr=args.min_lr,
                    factor=args.lr_reduce_factor,
                    alpha=args.MARS_alpha)

    # Setup trainer with logging, checkpoints, and early stopping
    tb_logger = TensorBoardLogger(args.result_path, name="", version=args.version, default_hp_metric=False)
    logger.info("Saving tensorboard to %s", os.path.join(tb_logger.save_dir, tb_logger.name, 'version_%s' % tb_logger.version))

    tb_logger.log_hyperparams(args.__dict__, {'epoch': model.current_epoch})

    checkpoint_callback = ModelCheckpoint(monitor='val/epoch_acc',
                                          mode='max',
                                          dirpath=tb_logger.log_dir,
                                          filename='model_{epoch:05d}',
                                          verbose=True,
                                          save_last=True)

    early_stop_callback = EarlyStopping(monitor='val/epoch_acc', mode='max', min_delta=0.0, patience=2*args.lr_patience+1, verbose=True)
    lr_monitor = LearningRateMonitor() # FIXME: Log LR on step?

    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=tb_logger,
                                            callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
                                            benchmark=True,
                                            sync_batchnorm=True)

    # Fit model to train data and then test model on test data
    trainer.fit(model, datamodule)
    result = trainer.test(model=model, datamodule=datamodule)

def parse_args(arguments=None):
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MARSDataModule.add_argparse_args(parser)
    parser = LitMARS.add_argparse_args(parser)

    # XXX: See if pytorch-lightning supports --version now
    parser.add_argument("--version", default=None, type=int, help="TensorBoard logger version number. Useful when using --resume_from_checkpoint to get results in same directory.")
    parser.add_argument('--result_path', default=None, type=str)
    parser.add_argument('--manual_seed', default=1337, type=int, help='Manually set random seed')

    # NOTE: Unsupported args from old code:
    #       --frame_mask_color [114.7748/255, 107.7354/255, 99.4750/255, 127.5/255, 127.5/255]
    #       --frame_mask_prob 1.
    #       --frame_mask_reverse False
    #       --mask_in_alpha False
    #       --frame_mask_no_ablate_flow False
    #       --frame_segment None [RGB, MC]
    #       --input_channels 3
    #       --residual_frames False
    #       --residual_frames_after_keyframe False
    #       --residual_frames_signed False
    #       --jitter False
    #       --gaussian_augmentation_std 0
    #       --gaussian_augmentation_truncated False
    #       --model resnext
    #       --model_depth 101
    #       --resnet_shortcut B
    #       --resnext_cardinality 32
    #       --flow_sample_size 112+
    #       --training True
    #       --n_epochs 400
    #       --begin_epoch 1
    #       --test_mid_frame_only False
    #       --ablate_before_attack False
    #       --ablate_after_attack False
    #       --MARS False
    #       --pretrain_path ""
    #       --MARS_pretrain_path ""
    #       --MARS_resume_path ""
    #       --resume_path2 ""
    #       --resume_path3 ""
    #       --log 1
    #       --checkpoint 2

    args = parser.parse_args(arguments)

    return args

def run_cli():
    coloredlogs.install(level=logging.INFO)

    args = parse_args()
    main(args)

if __name__ == '__main__':
    run_cli()
