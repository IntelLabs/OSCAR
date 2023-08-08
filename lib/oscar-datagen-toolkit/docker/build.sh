#! /bin/bash

# ==============================================================================
# -- Parse arguments -----------------------------------------------------------
# ==============================================================================

DOC_STRING="Build the CARLA PythonAPI client."

USAGE_STRING="Usage: $0 [--python-version=VERSION]"

OPTS=`getopt -o h --long help,python-version: -n 'parse-options' -- "$@"`

eval set -- "$OPTS"

PY_VERSION=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-version )
      PY_VERSION="$2";
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
# -- Build Docker image --------------------------------------------------------
# ==============================================================================

DOCKER_TAG="carla-build:py$PY_VERSION"
echo "Docker tag $DOCKER_TAG"

if [[ "$(docker images -q $DOCKER_TAG 2> /dev/null)" == "" ]]; then
    echo "Docker image $DOCKER_TAG does not exist!"

    docker build \
        --no-cache \
        --build-arg PYTHONVERSION=$PY_VERSION \
        --force-rm \
        --tag $DOCKER_TAG \
        --file Dockerfile-base .
fi

# ==============================================================================
# -- Set up environment --------------------------------------------------------
# ==============================================================================

BUILD_FOLDER="build"

if [ ! -d "$BUILD_FOLDER" ]; then
    mkdir -p $BUILD_FOLDER
fi

pushd ${BUILD_FOLDER} >/dev/null

# ==============================================================================
# -- Build CARLA PythonAPI -----------------------------------------------------
# ==============================================================================

CARLA_SIM_VERSION=0.9.13
CARLA_BASENAME="carla"
CARLA_REPO="$BUILD_FOLDER/$CARLA_BASENAME"

if [ ! -d "$CARLA_BASENAME" ]; then
    git clone -b ${CARLA_SIM_VERSION} --depth 1 https://github.com/carla-simulator/carla

    cp -r ../patches/* $CARLA_BASENAME

    pushd ${CARLA_BASENAME} >/dev/null

    git apply *.patch

    popd >/dev/null
fi

popd >/dev/null

docker run \
    --rm \
    --net=host \
    --volume $(pwd)/$CARLA_REPO:/opt/$CARLA_BASENAME \
    ${DOCKER_TAG} make PythonAPI ARGS="--python-version=$PY_VERSION"
