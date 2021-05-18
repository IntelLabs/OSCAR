#!/usr/bin/env python3
#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import sys
import torch


def _convert_state_dict(state_dict, src_prefix, dst_prefix):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        new_key = key.replace(src_prefix, dst_prefix)
        state_dict[new_key] = state_dict[key]

        if key != new_key:
            del state_dict[key]
    return state_dict


if __name__ == "__main__":
    src_ckpt, tgt_pth = sys.argv[1:]

    ckpt = torch.load(src_ckpt)
    state_dict = ckpt['state_dict']

    # Delete unwanted keys.
    for key in [key for key in state_dict.keys()]:
        if not key.startswith("rgb_backbone.") and \
           not key.startswith("backbone_rgb."):
            del state_dict[key]

    # Change prefix of keys.
    state_dict = _convert_state_dict(state_dict, "backbone_rgb.module.", "module.")
    state_dict = _convert_state_dict(state_dict, "rgb_backbone.", "module.")

    ckpt['state_dict'] = state_dict
    torch.save(ckpt, tgt_pth)
