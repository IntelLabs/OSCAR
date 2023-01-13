#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch

def log_to_linear(log):
    return torch.exp((log - 1.) * 5.70378)

def linear_to_log(linear):
    return (1. + torch.log(linear) / 5.70378)
