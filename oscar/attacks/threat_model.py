#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from mart.attack.threat_model import ThreatModel

class MaskAdditive(ThreatModel):
    """We assume an adversary adds masked perturbation to the input."""

    def forward(self, input, target, perturbation, **kwargs):
        mask = target["perturbable_mask"]
        perturbation = perturbation * mask

        return input + perturbation
