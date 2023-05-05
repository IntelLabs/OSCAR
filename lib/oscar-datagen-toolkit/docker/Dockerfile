# Copyright (c) 2022 Intel Corporation.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
FROM carlasim/carla:0.9.13

USER root
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential \
    libjpeg-dev libtiff-dev \
    python3.7 python3.7-dev python3-distutils \
    python3-pip python3.7-venv \
    git \
    xdg-user-dirs xdg-utils \
    fontconfig \
    zip
RUN apt-get clean

WORKDIR /opt
RUN mkdir -p /opt/carla-datagen-toolkit
COPY . /opt/carla-datagen-toolkit

# Create virtual environment
WORKDIR /opt/carla-datagen-toolkit
RUN python3.7 -m venv .carla_env

# Install CARLA datagen tools
RUN . .carla_env/bin/activate && \
    pip install --upgrade pip && \
    pip install .

# Setup environment
ENV PYTHONPATH "${PYTHONPATH}:/home/carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg"
ENTRYPOINT ["/opt/carla-datagen-toolkit/docker/scripts/entrypoint.sh"]

USER carla
WORKDIR /home/carla
