#
# Copyright (C) 2020 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import pkg_resources
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
from art.estimators.classification.pytorch import PyTorchClassifier
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy

from mmskeleton.models.backbones.st_gcn_aaai18 import ST_GCN_18

from oscar.data.precomputed import PrecomputedLightningDataModule
from oscar.utils.argparse import NegateAction
from oscar.utils.transforms.keypoints import (
    BatchStandardizeSTGCNInputTimesteps,
    StandardizeSTGCNInputTimesteps,
)


logger = logging.getLogger(__name__)

MODEL_ZOO_DIR = Path(pkg_resources.resource_filename("oscar.model_zoo", ""))
STGCN_ZOO_DIR = MODEL_ZOO_DIR / "ST_GCN"
PRETRAINED_WEIGHTS_PATH = STGCN_ZOO_DIR / "st_gcn.kinetics-6fa43f73.pth"

DEFAULT_OPTIM_KWARGS = dict(
    lr=0.001,
    momentum=0.9,
    nesterov=True,
    weight_decay=0.00001,
)


class CustomSTGCN(ST_GCN_18):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        graph_layout: str = "openpose",
        graph_strategy: str = "spatial",
        edge_importance_weighting: bool = True,
    ):
        super().__init__(
            num_class=num_classes,
            in_channels=in_channels,
            edge_importance_weighting=edge_importance_weighting,
            graph_cfg={"layout": graph_layout, "strategy": graph_strategy},
        )

        self.input_transform = BatchStandardizeSTGCNInputTimesteps()

    def load_weights(self, weights_path: Union[Path, str], drop_fcn: bool = False):
        weights_path = Path(weights_path).resolve()

        state_dict = None
        try:
            state_dict = torch.load(weights_path)
        except RuntimeError:
            logger.info(
                "Couldn't load state dict to CPU, attempting to load onto GPU..."
            )
        if state_dict is None:
            state_dict = torch.load(
                weights_path, map_location=f"cuda:{torch.cuda.current_device()}"
            )
        assert state_dict is not None

        # if weights are a PyTorch Lightning checkpoint
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {
                k.replace("stgcn.", ""): v
                for k, v in state_dict.items()
                if k.startswith("stgcn")
            }

        # if loading pretrained model trained on 400 classes
        if drop_fcn:
            del state_dict["fcn.weight"]
            del state_dict["fcn.bias"]

        strict_mode = not drop_fcn

        self.load_state_dict(state_dict, strict=strict_mode)
        logger.info(f"Loaded STGCN weights from `{weights_path}`")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_transform(x)
        return super().forward(x)

    def extract_feature(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_transform(x)
        return super().extract_feature(x)


class LitSTGCN(LightningModule):
    def __init__(
        self,
        pretrained_weights_path=None,
        drop_fcn=True,
        fulltune_model=False,
        optim_kwargs=DEFAULT_OPTIM_KWARGS,
        **_,
    ):
        super().__init__()

        # instantiate model
        self.stgcn = CustomSTGCN(num_classes=101)

        # instantiate loss
        self.loss_fn = nn.CrossEntropyLoss()

        # instantiate accuracies
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

        # load pretrained weights
        if pretrained_weights_path is not None:
            self.stgcn.load_weights(pretrained_weights_path, drop_fcn=drop_fcn)

        self.optim_kwargs = optim_kwargs
        self.fulltune_model = fulltune_model
        self.pretrained_weights_path = pretrained_weights_path

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--pretrained_weights_path", type=str, default=str(PRETRAINED_WEIGHTS_PATH)
        )

        parser.add_argument("--fulltune_model", action="store_true")
        parser = NegateAction.add_to_parser(parser, "drop_fcn")

        for option in ["lr", "momentum", "weight_decay"]:
            arg = f"--optim_{option}"
            arg_default = DEFAULT_OPTIM_KWARGS[option]
            parser.add_argument(arg, type=float, default=arg_default)
        if DEFAULT_OPTIM_KWARGS["nesterov"] is True:
            # since default value is True,
            # we also want a --no_optim_nesterov flag
            parser = NegateAction.add_to_parser(parser, "optim_nesterov")
        else:
            parser.add_argument("--optim_nesterov", action="store_true")

        return parser

    def forward(self, x):
        return self.stgcn(x)

    def process_batch(self, batch):
        x, y = batch
        logits = self.stgcn(x)
        loss = self.loss_fn(logits, y)
        return logits, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.process_batch(batch)
        self.train_acc(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.process_batch(batch)
        self.valid_acc(logits, y)

        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", self.valid_acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.process_batch(batch)
        self.test_acc(logits, y)

        self.log("test_acc", self.test_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        logger.info(
            "Optimizer will **{}-tune** the model".format(
                "full" if self.fulltune_model else "fine"
            )
        )
        trainable_parameters = (
            self.stgcn.parameters()
            if self.fulltune_model
            else self.stgcn.fcn.parameters()
        )

        logger.info(f"Instantiating optimizer with arguments: {self.optim_kwargs}")
        return torch.optim.SGD(trainable_parameters, **self.optim_kwargs)


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: str
) -> PyTorchClassifier:

    if "num_classes" not in model_kwargs:
        model_kwargs.update(num_classes=101)

    stgcn = CustomSTGCN(**model_kwargs).eval()
    stgcn.load_weights(weights_path)

    wrapped_model = PyTorchClassifier(
        model=stgcn,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 300, 18, 1),
        nb_classes=101,
    )

    return wrapped_model


def setup_argparse_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--seed", type=int, default=42)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitSTGCN.add_model_specific_args(parser)

    parser.add_argument("--precomputed_train_dataset", type=str, required=True)
    parser.add_argument("--precomputed_test_dataset", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=128)

    parser.set_defaults(max_epochs=100, log_every_n_steps=5)

    return parser


def cli_main(args):
    pl.seed_everything(args.seed)

    tb_logger = pl.loggers.TensorBoardLogger(
        name=STGCN_ZOO_DIR.name, save_dir=str(MODEL_ZOO_DIR)
    )
    tb_logger.log_hyperparams(args)

    args_dict = vars(args)
    args_dict.update(
        optim_kwargs={
            k.replace("optim_", ""): v
            for k, v in args_dict.items()
            if k.startswith("optim_")
        }
    )

    lit_model = LitSTGCN(**args_dict)

    data_module = PrecomputedLightningDataModule(
        train_dataset_path=args.precomputed_train_dataset,
        test_dataset_path=args.precomputed_test_dataset,
        transforms_dict=dict(x=StandardizeSTGCNInputTimesteps()),
        val_split_percent=20,
        dataloader_batch_size=args.batch_size,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid_acc",
        mode="max",
        save_top_k=-1,
    )

    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, callbacks=[checkpoint_callback]
    )

    trainer.fit(lit_model, data_module)

    checkpoint_callback.to_yaml()

    best_model_path = Path(checkpoint_callback.best_model_path)
    trainer.test(ckpt_path=str(best_model_path))
    shutil.copy2(best_model_path, best_model_path.parent / "best_model.pth")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = setup_argparse_args(parser)
    args = parser.parse_args()

    cli_main(args)
