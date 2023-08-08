#! /bin/bash

# Default values
carlaPort=5000
qualityLevel="Epic"
options=""

# Function to display script usage
usage() {
  echo "Usage: $0 [-r] [-p <carla-port>] [-q <quality-level>]"
  echo "-r, --RenderOffScreen   Enable rendering off-screen"
  echo "-p, --carla-port       Specify the CARLA port (default: 5000)"
  echo "-q, --quality-level    Specify the quality level (default: Epic)"
  exit 1
}

# Parse command-line options
while getopts ":rp:q:" opt; do
  case $opt in
    r)
      options="$options -RenderOffScreen"
      ;;
    p)
      carlaPort=$OPTARG
      ;;
    q)
      qualityLevel=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      usage
      ;;
    :)
      echo "Option -$OPTARG requires an argument."
      usage
      ;;
  esac
done

# GPU varialbes
gpu_id="0"

# NOTE: a text file is used to keep track of the GPU
# usage among the scaled CARLA services, created with
# docker compose
intance_counter="/home/carla/workspace/.instance_counter"
if [[ -e $intance_counter ]]; then

  # get GPU information
  gpu_list=$(nvidia-smi -L)
  gpu_count=$(grep -c "GPU [0-9]:" <<< "$gpu_list")

  # determine GPU id
  counter=$(<"$intance_counter")
  gpu_id="$counter"

  # increase counter if there are enough GPUs
  counter=$((counter + 1))
  if (( counter >= gpu_count )); then
    counter="0"
  fi

  echo "$counter" > "$intance_counter"
fi

# launch CARLA server
options="$options -carla-port=$carlaPort -quality-level=$qualityLevel -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=$gpu_id"
command="/home/carla/CarlaUE4.sh $options"
echo "Server command: $command"
eval $command
