#! /bin/bash

#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

USER=$(id -u):$(id -g)

# run notebook in container
set -x
docker compose run --rm --user $USER --publish 8888:8888 --entrypoint "jupyter notebook --ip 0.0.0.0" oscar_data_saver
set +x

# cleanup
docker compose down
