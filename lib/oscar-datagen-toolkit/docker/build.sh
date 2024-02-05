#! /bin/bash -e

#
# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

# ==============================================================================
# -- Parse arguments -----------------------------------------------------------
# ==============================================================================

DOC_STRING="Build the CARLA PythonAPI client."

USAGE_STRING="Usage: $0 [--carla-version=VERSION] [--python-version=VERSION]"

OPTS=`getopt -o h --long help,carla-version:,python-version: -n 'parse-options' -- "$@"`

eval set -- "$OPTS"

CARLA_VERSION=0.9.13
PYTHON_VERSION=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --carla-version )
      CARLA_VERSION="$2";
      shift ;;
    --python-version )
      PYTHON_VERSION="$2";
      shift ;;
    -h | --help )
      echo "$DOC_STRING"
      echo "$USAGE_STRING"
      exit 1
      ;;
    * )
      shift ;;
  esac
done

# ==============================================================================
# -- Build CARLA PythonAPI in Docker image -------------------------------------
# ==============================================================================

DOCKER_TAG="carla-client:$CARLA_VERSION-py$PYTHON_VERSION"
echo "Docker tag $DOCKER_TAG"

# FIXME: This will only work from the build directory because of how COPY works
docker build \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --force-rm \
    --tag $DOCKER_TAG \
    --target build \
    --file Dockerfile-oscar-datagen ..

# ==============================================================================
# -- Copy Carla PythonAPI wheel from Docker image ------------------------------
# ==============================================================================

docker create --name carla-build $DOCKER_TAG
docker cp carla-build:"/workspace/PythonAPI/carla/dist/" .
docker rm carla-build
