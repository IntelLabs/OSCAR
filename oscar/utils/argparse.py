#
# Copyright (C) 2020 Georgia Institute of Technology. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

from argparse import ArgumentParser, Action, Namespace
from typing import List


class NegateAction(Action):
    # adapted from https://stackoverflow.com/a/34736291

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: List[str],
        option: str,
    ):
        setattr(namespace, self.dest, option[2:4] != "no")

    @classmethod
    def add_to_parser(cls, parser: ArgumentParser, dest: str) -> ArgumentParser:
        parser.add_argument(
            f"--{dest}", f"--no_{dest}", dest=dest, action=cls, default=True, nargs=0
        )

        return parser
