#! /bin/bash

SCRIPT=$(realpath "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
source "$SCRIPT_PATH/common.sh"

# ==============================================================================
# -- Parse arguments -----------------------------------------------------------
# ==============================================================================

DOC_STRING="Collect new data from a CARLA sim server."

USAGE_STRING="Usage: $0 [--scale=NUM] [CONFIGURATION-FILE]"

OPTS=`getopt -o h --long help,scale: -n 'parse-options' -- "$@"`

eval set -- "$OPTS"

config=""
scale=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help )
      echo "$DOC_STRING"
      echo "$USAGE_STRING"
      docker compose run --no-deps --entrypoint="oscar_data_saver --help" collector
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
# -- Prepare collection --------------------------------------------------------
# ==============================================================================

intance_counter=".instance_counter"
if [[ ! -f $intance_counter ]]; then
    touch "$intance_counter"
    chmod 777 "$intance_counter"
fi

# reset counter
echo "0" > "$intance_counter"

if [[ ! -f $config ]]; then
    echo "$config does not exist."
    exit 1
fi

ln -f $config ./.tmp_conf.yaml

# ==============================================================================
# -- Run collection process ----------------------------------------------------
# ==============================================================================

PROJECT_NAME="oscar"

build_command() {
    local host=$1
    local command="oscar_data_saver \
    --config-dir=/workspace \
    --config-name=.tmp_conf \
    context.client_params.host="$host" \
    context.client_params.port=$CARLA_PORT \
    context.simulation_params.traffic_manager_port=$CARLA_TM_PORT"
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
    SERVER_ADDRESS="$PROJECT_NAME-carla-$i"
    COLLECT_COMMAND=$(build_command $SERVER_ADDRESS)

    echo "Collection command: $COLLECT_COMMAND"
    docker compose --project-name $PROJECT_NAME exec --index $i $options collector /bin/bash -c "$COLLECT_COMMAND"
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

rm .tmp_conf.yaml
docker compose --project-name $PROJECT_NAME down
