DATASETS ?= /datasets/data_collector
PORT := $(shell seq 49152 65535 | shuf | head -n1)
TM_PORT := $(shell echo "$(PORT) + 2" | bc)
CARLA_QUALITY_LEVEL ?= Epic
CONFIG_FILE ?= config_tracking
NV_GPU ?= "device=0"
DEFAULT_UUID := $(shell uuidgen)
UUID ?= $(DEFAULT_UUID)
PYTHON = python3.7

SIMULATION ?= sensors_hero
ANNOTATION ?= kwcoco
CATEGORIES ?= [Pedestrian]

CVAT_HOME = $(HOME)/cvat
CVAT_SHARE_VOLUME = cvat_share

DOCKER_TAG ?= carla-client:0.9.13
DOCKER_OPTIONS += --rm
DOCKER_OPTIONS += --net=host
DOCKER_OPTIONS += --volume $(shell pwd)\:/home/carla/PythonAPI/util/data_collector\:ro
DOCKER_OPTIONS += --volume $(DATASETS)\:$(DATASETS)\:rw
DOCKER_OPTIONS += --gpus $(NV_GPU)
DOCKER_OPTIONS += --env HYDRA_FULL_ERROR=1
DOCKER_RUN = docker run $(DOCKER_OPTIONS) $(DOCKER_TAG)
DOCKER_RUND = docker run -d $(DOCKER_OPTIONS) --name $(UUID) $(DOCKER_TAG)
DOCKER_RUN_IT = docker run $(DOCKER_OPTIONS) -it $(DOCKER_TAG)

# Taken from https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

# Taken from https://suva.sh/posts/well-documented-makefiles/
.PHONY: help
help:  ## Display this help
> @awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[1-9a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-36s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

#
# General Targets
#

.PHONY: carla_client
carla_client:  ## Run CARLA client
> @echo "$(GREEN)Starting CARLA client"
> $(DOCKER_RUN_IT) /bin/bash


.PHONY: docker_image
docker_image: docker/Dockerfile ## Build docker image
> docker build --no-cache -f $< -t $(DOCKER_TAG) .

.PHONY: launch_cvat
launch_cvat: # Launch CVAT UI
> @echo "Starting CVAT UI and mounting directories"
> docker volume rm cvat_${CVAT_SHARE_VOLUME}
> cd ${CVAT_HOME}; MOUNT_PATH=${DATASETS} docker-compose up -d

.PHONY: terminate_cvat
terminate_cvat: # Terminate CVAT UI
> @echo "Terminate CVAT UI"
> cd ${CVAT_HOME}; docker-compose down

.PHONY: kill_carla_servers
kill_carla_servers: ## Kill all CARLA server docker instances
> docker ps --filter="ancestor=carla-client:0.9.13" -q | xargs docker kill

.PHONY: clean_annotations
clean_annotations: ## Clean existing annotations
> @echo "Clearing annotations "
> $(DOCKER_RUN) rm -rf $(DATASETS)/*/*/kwcoco_annotations.json
> $(DOCKER_RUN) rm -rf $(DATASETS)/*/*/instances*

.PHONY: clean
clean: ## Clean all directories
> @echo "Clearing all directories"
> $(DOCKER_RUN) rm -rf $(DATASETS)/*

.PHONY: simulation
simulation:
> @echo "Running CARLA data collector with $(SIMULATION) simulation"
> $(DOCKER_RUND) /bin/bash ./CarlaUE4.sh -RenderOffScreen -carla-port=$(PORT) -quality-level=$(CARLA_QUALITY_LEVEL)
> while ! nc -z localhost $(PORT); do sleep 1; done
> $(DOCKER_RUN) oscar_data_saver simulation=$(SIMULATION) context.client_params.port=$(PORT) context.simulation_params.traffic_manager_port=$(TM_PORT) hydra.run.dir="/datasets/data_collector/config_$(SIMULATION)/$(UUID)"
> docker ps --quiet --filter="name=$(UUID)" -q | xargs docker kill

.PRECIOUS: $(DATASETS)/$(CONFIG_FILE)/%/collection_done
$(DATASETS)/$(CONFIG_FILE)/%/collection_done: ## Run data collection with config_tracking configuration
> $(DOCKER_RUND) /bin/bash ./CarlaUE4.sh -RenderOffScreen -carla-port=$(PORT) -quality-level=$(CARLA_QUALITY_LEVEL)
> while ! nc -z localhost $(PORT); do sleep 1; done
> $(DOCKER_RUN) oscar_data_saver simulation=tracking carla.port=$(PORT) carla.traffic_manager_port=$(TM_PORT) hydra.run.dir="$(@D)"
> docker ps --quiet --filter="name=$(UUID)" -q | xargs docker kill
> $(DOCKER_RUN) /bin/bash -c "echo $(UUID) > $@"

.PHONY: annotation
annotation:
> @echo "Running CARLA $(ANNOTATION) annotator"
> $(DOCKER_RUN) oscar_data_annotator $(ANNOTATION) --categories $(CATEGORIES) --dataset_parent_dir="/datasets/data_collector/config_$(SIMULATION)/$(UUID)"
