#
# Copyright (C) 2020 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from detectron2.data import MetadataCatalog

for d in ['train', 'val']:
    MetadataCatalog.get(f"mnist_{d}").set(thing_classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

