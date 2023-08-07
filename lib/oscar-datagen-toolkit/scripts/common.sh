#! /bin/bash

source .env

# ==============================================================================
# -- Verify where the script was executed --------------------------------------
# ==============================================================================

SCRIPT_PATH=$(dirname "$SCRIPT")
ROOT_DIR_PATH=`echo $SCRIPT_PATH | rev | cut -d'/' -f2- | rev`

if [ "$ROOT_DIR_PATH" != "$PWD" ]; then
    echo "Ensure this script is ran from the root of the OSCAR datagen tools repo."
    exit 1
fi

# ==============================================================================
# -- Verify docker images ------------------------------------------------------
# ==============================================================================

export U_ID="$(id -u)"
export G_ID="$(id -g)"

BASE_NAME="oscar"
BASE_IMAGE_TAG="$BASE_NAME/base:$DATAGEN_TOOLS_VERSION"
DATAGEN_IMAGE_TAG="$BASE_NAME/datagen:$DATAGEN_TOOLS_VERSION"
CARLA_IMAGE_TAG="$BASE_NAME/carla:$CARLA_VERSION"

# Verify if Docker images are built
if [[ "$(docker images -q $BASE_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    echo "Docker image $BASE_IMAGE_TAG does not exist!"

    docker compose build base
fi

if [[ "$(docker images -q $DATAGEN_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    echo "Docker image $DATAGEN_IMAGE_TAG does not exist!"

    docker compose build annotator
fi

if [[ "$(docker images -q $CARLA_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    echo "Docker image $CARLA_IMAGE_TAG does not exist!"

    docker compose build carla
fi
