# OSCAR Datagen Toolkit

This repository contains a set of tools for generating synthetic datasets using the CARLA Simulator. [CARLA](https://carla.org/) is an open-source simulator for autonomous driving research that provides realistic urban environments, dynamic traffic, and realistic sensors.

## Requirements

- [Docker](https://docs.docker.com/engine/install/#server)

If you want to install the tools outside of the Docker container, build and install the CARLA Python package before install the tools:

```
cd docker

# build the CARLA client
# NOTE: Specify the Python3 version you want to use
./build --python-version=3.9

# install CARLA
pip install build/carla/PythonAPI/carla/dist/carla-0.9.13-cp39-cp39-linux_x86_64.whl
```

> **Note**: It is recommended to create a Python virtual environment.
>
> ```
> # [OPTIONAL] create conda environment
> conda create -n myenv python=3.9
> conda activate myenv
>
> # [OPTIONAL] or create virtualenv environment
> python3.9 -m venv .venv
> source .venv/bin/activate
> ```

## Getting started

- With Docker:

1. Clone this repository to your local machine.
2. Build the required Docker images:

```
# Build oscar/base image
docker compose build base

# Build oscar/datagen image
docker compose build annotator

# Build CARLA image
docker compose pull carla
```

> **Note**: The `collect.sh` and `annotate.sh` scripts automatically verify the exitances of these Docker images and build them if necessary. You can use the above command if you want to rebuild the Docker images.

- Manual installation

1. Run the following command:

```
pip install .
```

> **Note**: In this case, make sure that you have a running CARLA server to connect with.

> \[Optional\] Install the pre-commit hook:

```
pip install pre-commit==2.20.0

# To run the hook on every commit
pre-commit install

# To run it manually
pre-commit run -a
```

## Tools

The following tools are included in this repository:

### oscar_data_saver.py

This script sets up a CARLA simulation under certain parameters specified in one of the YAML's configuration files located at **configs/simulation**, and then proceed with collection of the data from the sensors included in that simulation. To use this script run:

```
./scripts/collect.sh <CONFIG-FILE>
```

where `CONFIG-FILE` is replaced with the path of the simulation configuration file.

To run this tool outside the Docker container use the following command"

```
oscar_data_saver --config-dir=</path/to/config/directory> --config-name=<CONFIG_NAME> hydra.run.dir=</path/to/output>
```

### oscar_data_annotator.py

This script takes the data previously generated by the **oscar_data_saver.py** tool and creates COCO- or MOTS-compatible datasets. The following is an example of how to use this tool:

```
./scripts/annotate.sh <FORMAT> --dataset_parent_dir="/opt/datagen-repo/data/UUID"
```

where:

- UUID: unique identifier for the data collection run.
- ANNOTATION: type of annotation that is going to be generated (*kwcoco*, *mots_txt* or *motx_png*).

> **Note**: The data directory is mapped to `/opt/datagen-repo/data` inside the Docker container.

To run this tool outside the Docker container use the following command:

```
oscar_data_annotator <FORMAT> --dataset_parent_dir="/opt/datagen-repo/data/UUID"
```