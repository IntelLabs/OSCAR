#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from detectron2.data import MetadataCatalog

for d in ['train', 'val']:
    MetadataCatalog.get(f"mnist_{d}").set(thing_classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

