#! /bin/bash

#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

# ==============================================================================
# -- Run annotation process ----------------------------------------------------
# ==============================================================================

USER=$(id -u):$(id -g)
subdirs=( $(find . -name "*.hydra" -exec dirname {} \;) )

# execute annotators
for subdir in "${subdirs[@]}"; do
    set -x
    docker compose run --rm --user $USER oscar_data_annotator $@ --dataset_parent_dir=$subdir
    set +x
done

# ==============================================================================
# -- Clean up ------------------------------------------------------------------
# ==============================================================================

docker compose down
