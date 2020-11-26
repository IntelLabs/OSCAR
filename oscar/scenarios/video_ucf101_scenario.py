#
# Copyright (C) 2020 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#
import logging

from armory.scenarios import video_ucf101_scenario
from importlib import import_module
from armory.data.datasets import ArmoryDataGenerator, EvalGenerator

logger = logging.getLogger(__name__)

# Patch load_dataset@armory/utils/config_loading.py because we want to pass kwargs
# See: https://github.com/twosixlabs/armory/issues/592
def _load_dataset(dataset_config, *args, num_batches=None, **kwargs):
    """
    Loads a dataset from configuration file

    If num_batches is None, this function will return a generator that iterates
    over the entire dataset.
    """
    dataset_module = import_module(dataset_config["module"])
    dataset_fn = getattr(dataset_module, dataset_config["name"])
    batch_size = dataset_config["batch_size"]
    framework = dataset_config.get("framework", "numpy")

    # XXX: BEGIN PATCH
    kwargs.update(dataset_config['kwargs'])
    # XXX: END PATCH

    dataset = dataset_fn(batch_size=batch_size, framework=framework, *args, **kwargs)
    if not isinstance(dataset, ArmoryDataGenerator):
        raise ValueError(f"{dataset} is not an instance of {ArmoryDataGenerator}")
    if dataset_config.get("check_run"):
        return EvalGenerator(dataset, num_eval_batches=1)
    if num_batches:
        return EvalGenerator(dataset, num_eval_batches=num_batches)

    return dataset

class Ucf101(video_ucf101_scenario.Ucf101):
    def __init__(self):
        logger.info("Monkey patching armory.scenarios.video_ucf101_scenario.load_dataset to accept dataset kwargs")
        video_ucf101_scenario.load_dataset = _load_dataset
