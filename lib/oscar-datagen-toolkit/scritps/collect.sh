#! /bin/bash

SCRIPT=$(realpath "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
source "$SCRIPT_PATH/common.sh"

# ==============================================================================
# -- Parse arguments -----------------------------------------------------------
# ==============================================================================

DOC_STRING="Collect new data from a CARLA sim server."

USAGE_STRING="Usage: $0 [OPTIONS] [CONFIGURATION-FILE]"

OPTS=`getopt -o h --long help,scale: -n 'parse-options' -- "$@"`

eval set -- "$OPTS"

config=""
scale=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help )
      echo "$DOC_STRING"
      echo "$USAGE_STRING"
      exit 1
      ;;
    --scale )
      scale=$2;
      shift ;;
    --)
      shift ;;
    * )
      config=$1
      shift ;;
  esac
done

if [ -z "$config" ]; then
    echo "$DOC_STRING"
    echo "$USAGE_STRING"
    exit 1
fi

# ==============================================================================
# -- Prepare config file -------------------------------------------------------
# ==============================================================================

if [[ ! -f $config ]]; then
    echo "$config does not exist."
    exit 1
fi

ln $config ./tmp_conf.yaml

# ==============================================================================
# -- Run collection process ----------------------------------------------------
# ==============================================================================

PROJECT_NAME="oscar"

build_command() {
    local tmp_uuid=$1
    local host=$2
    local command="oscar_data_saver \
    --config-dir=/workspace \
    --config-name=tmp_conf \
    context.client_params.host="$host" \
    context.client_params.port=$CARLA_PORT \
    context.simulation_params.traffic_manager_port=$CARLA_TM_PORT \
    hydra.run.dir=\"/opt/datagen-repo/data/$tmp_uuid\""
    echo "$command"
}

options=""
if [ "$scale" -gt 1 ]; then
    # enable the detach mode for scales greater than 1
    options="-d"
fi

docker compose --project-name $PROJECT_NAME up -d --scale collector=$scale --scale carla=$scale
echo "Run $scale data collection processes"

for i in $(seq $scale); do
    UUID=$(uuidgen)
    SERVER_ADDRESS="$PROJECT_NAME-carla-$i"
    COLLECT_COMMAND=$(build_command $UUID $SERVER_ADDRESS)

    echo "Collection command: $COLLECT_COMMAND"
    docker compose --project-name $PROJECT_NAME exec --index $i $options collector /bin/bash -c "$COLLECT_COMMAND"
    echo "Stored data at $HOST_MOUNT_POINT/$UUID"
done


# ==============================================================================
# -- Clean up ------------------------------------------------------------------
# ==============================================================================

if [[ "$scale" -gt 1 ]]; then
    echo "Waiting for data collections to complete..."

    exec_count=0
    while IFS= read -r line; do
        # check is not an empty line
        if [ "$line" != "" ]; then
            echo "Got service event: $line"
        fi

        # check it is an exec_die event
        if [[ "$line" == *"exec_die"* ]]; then
            exec_count=$((exec_count + 1))
        fi

        if [ "$exec_count" -eq "$scale" ]; then
            break
        fi
    done < <(docker compose --project-name $PROJECT_NAME events collector)
fi

rm tmp_conf.yaml
docker compose --project-name $PROJECT_NAME down
