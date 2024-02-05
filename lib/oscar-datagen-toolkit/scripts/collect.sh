#! /bin/bash

#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

# ==============================================================================
# -- Parse arguments -----------------------------------------------------------
# ==============================================================================

DOC_STRING="Collect new data from a CARLA sim server."

USAGE_STRING="Usage: $0 [--config CONFIGURATION-FILE] [HYDRA OPTIONS]"

OPTS=`getopt -o h --long help,config: -n 'parse-options' -- "$@"`

eval set -- "$OPTS"

CONFIG=""
HELP=false
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help )
      HELP=true;
      shift ;;
    --config )
      CONFIG=$2
      shift 2 ;;
    --)
      shift ;;
    * )
      # collecting remaining arguments
      ARGS+=("$1")
      shift ;;
  esac
done

if [ "$HELP" = true ]; then
    echo "$DOC_STRING"
    echo "$USAGE_STRING"
    docker compose run --rm --no-deps oscar_data_saver
    exit 0
fi

# ==============================================================================
# -- Prepare collection --------------------------------------------------------
# ==============================================================================

# prepare config file
if [ -n "$CONFIG" ]; then
    if [[ ! -f $CONFIG ]]; then
        echo "$CONFIG does not exist."
        exit 1
    fi

    cp -f $CONFIG .tmp_conf.yaml

    ARGS+=("--config-dir=/workspace")
    ARGS+=("--config-name=.tmp_conf")
fi

# ==============================================================================
# -- Run collection process ----------------------------------------------------
# ==============================================================================

USER=$(id -u):$(id -g)

set -x
docker compose run --rm --user $USER oscar_data_saver ${ARGS[@]}
exit_code=$?
set +x

# ==============================================================================
# -- Clean up ------------------------------------------------------------------
# ==============================================================================

if [ -n "$CONFIG" ]; then
    rm .tmp_conf.yaml
fi

docker compose down

exit $exit_code
