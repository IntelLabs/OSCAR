#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
logger = logging.getLogger(__name__)

class MonkeyPatch:
    """ Temporarily replace a module's object value, i.e., its functionality """
    def __init__(self, obj, name, value, verbose=False):
        self.obj = obj
        self.name = name
        self.value = value
        self.verbose = verbose

    def __enter__(self):
        self.orig_value = getattr(self.obj, self.name)

        if self.orig_value == self.value:
            return

        if self.verbose:
            logger.info("Monkey patching %s to %s", self.orig_value, self.value)

        setattr(self.obj, self.name, self.value)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.orig_value == self.value:
            return

        if self.verbose:
            logger.info("Reverting monkey patch on %s", self.orig_value)

        setattr(self.obj, self.name, self.orig_value)
