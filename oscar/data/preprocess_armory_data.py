#
# Copyright (C) 2020 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

from argparse import ArgumentParser
import json

from armory.paths import set_mode
from tqdm import tqdm

from oscar.data.armory2torch import Armory2TorchDataset
from oscar.data.precomputed import PrecomputedDataset


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("scenario_config_path", type=str)
    parser.add_argument("output_dataset_path", type=str)

    return parser.parse_args()


def precompute_data(
    input_dataset: Armory2TorchDataset, output_dataset: PrecomputedDataset
):
    n = len(input_dataset)

    for x, y in tqdm(input_dataset, total=n):
        output_dataset.add_item_no_name(x=x, y=y)


def main():
    set_mode("host")
    args = parse_args()

    with open(args.scenario_config_path, "r") as f:
        config = json.load(f)

    input_dataset = Armory2TorchDataset.from_scenario_config(
        config=config, epochs=1, shuffle_files=False
    )

    output_dataset = PrecomputedDataset(
        dataset_path=args.output_dataset_path,
        item_keys=("x", "y"),
        transforms_dict=None,
        mode="w",
    )

    precompute_data(input_dataset, output_dataset)


if __name__ == "__main__":
    main()
