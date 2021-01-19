# Change these as necessary
DATASETS ?= /raid/datasets
DOCKER ?= nvidia-docker
ARMORY_CONFIG ?= $(HOME)/.armory/config.json

# Don't change these
MODEL_ZOO = oscar/model_zoo
POETRY = $(HOME)/.poetry/bin/poetry
ARMORY = $(shell which armory)
DOCKER_IMAGE_TAG = intellabs/oscar:0.12.3
JQ = jq --indent 4 -r
GIT_SUBMODULES = lib/armory/.git lib/mmskeleton/mmskeleton/.git
MAKEFILE_DIR = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

BLACK := $(shell tput -Txterm setaf 0)
RED := $(shell tput -Txterm setaf 1)
GREEN := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
PURPLE := $(shell tput -Txterm setaf 5)
BLUE := $(shell tput -Txterm setaf 6)
WHITE := $(shell tput -Txterm setaf 7)
RESET := $(shell tput -Txterm sgr0)

ifneq (,$(wildcard $(ARMORY_CONFIG)))
ARMORY_DATASET_DIR = $(shell $(JQ) .dataset_dir $(ARMORY_CONFIG))
ARMORY_OUTPUT_DIR = $(shell $(JQ) .output_dir $(ARMORY_CONFIG))
ARMORY_SAVED_MODEL_DIR = $(shell $(JQ) .saved_model_dir $(ARMORY_CONFIG))
ARMORY_TMP_DIR = $(shell $(JQ) .tmp_dir $(ARMORY_CONFIG))
endif

ifneq ($(ARMORY_SAVED_MODEL_DIR), $(MAKEFILE_DIR)$(MODEL_ZOO))
    $(warning $(YELLOW)WARNING: Please configure armory to use $(WHITE)saved_model_dir$(YELLOW) as $(WHITE)$(MAKEFILE_DIR)$(MODEL_ZOO)$(YELLOW); current value is $(ARMORY_SAVED_MODEL_DIR)$(RESET).)
endif

ifneq ($(ARMORY_OUTPUT_DIR), $(MAKEFILE_DIR)results)
    $(warning $(YELLOW)WARNING: Please configure armory to use $(WHITE)output_dir$(YELLOW) as $(WHITE)$(MAKEFILE_DIR)results$(YELLOW); current value is $(ARMORY_OUTPUT_DIR)$(RESET).)
endif

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
.PHONY: clean
clean: clean_scenarios
> rm -f oscar/model_zoo/MiDaS/model.pt
> git submodule deinit -f .
> rm -rf .venv

.PHONY: clean_scenarios
clean_scenarios:
> rm -f scenario_configs/**/*.json
> rm -f scenario_configs/*.json

.PHONY: ubuntu_deps
ubuntu_deps: ## Install Ubuntu dependencies
> apt install python3.7 python3.7-dev python3.7-venv jq

$(DATASETS):
> $(error You need to specify your datasets directory using DATASETS=/path/to/datasets when calling make.)

#
# Python Targets
#
$(POETRY):
> curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_VERSION="1.0.5" python
> sed -i "s/python/python3/g" $(POETRY)

.PHONY: poetry
poetry: $(POETRY) ## Launch poetry with ARGS
> $(POETRY) $(ARGS)

lib/%/.git:
> git submodule update --init `dirname $@`

.venv: $(POETRY) $(GIT_SUBMODULES) pyproject.toml
> $(POETRY) run pip install pip==20.2.4
> $(POETRY) install
> touch $@
> @echo "$(YELLOW)Make sure to configure armory if you haven't already:$(RESET)"
> @echo "    output_dir: $(GREEN)$(MAKEFILE_DIR)results$(RESET)"
> @echo "    saved_model_dir: $(GREEN)$(MAKEFILE_DIR)$(MODEL_ZOO)$(RESET)"

.PHONY: python_deps
python_deps: .venv ## Install python dependencies into virtual environment using poetry

.PHONY: test
test: .venv
> $(POETRY) run pytest test

#
# Docker Targets
#
.PHONY: docker_image
docker_image: docker/Dockerfile ## Creates OSCAR docker image for use in armory
> cd docker && $(DOCKER) build --file Dockerfile -t $(DOCKER_IMAGE_TAG) .

#
# Submission Targets
#
submission/:
> mkdir -p $@

submission/INTL_Dockerfile: docker/Dockerfile submission/
> cp $< $@

.PHONY: submission
submission: submission/INTL_Dockerfile \
            ucf101_submission
> $(info All submission files should be in the $@/ folder now.)

.PHONY: clean_submission
clean_submission:
> rm -f submission/*

#
# Armory Targets
#
.PRECIOUS: lib/armory/scenario_configs/%.json
lib/armory/scenario_configs/%.json: lib/armory/.git
> touch $@

scenario_configs/:
> mkdir -p $@

scenario_configs/%.json: lib/armory/scenario_configs/%.json scenario_configs/
> cat $< | $(JQ) '.sysconfig.docker_image = "$(DOCKER_IMAGE_TAG)"' > $@

results/%.json.armory_run: results/%.json | .venv
> $(JQ) ".sysconfig.output_dir = \"$(*D)/armory_run\"" $< | $(POETRY) run armory run --no-docker - $(ARGS)

results/%.json.armory_check: results/%.json | .venv
> $(JQ) ".sysconfig.output_dir = \"$(*D)/armory_check\"" $< | $(POETRY) run armory run --no-docker --check - $(ARGS)

results/%.json.armory_docker_run: results/%.json | .venv
> $(JQ) ".sysconfig.output_dir = \"$(*D)/armory_docker_run\"" $< | $(POETRY) run armory run - $(ARGS)

results/%.json.armory_docker_check: results/%.json | .venv
> $(JQ) ".sysconfig.output_dir = \"$(*D)/armory_docker_check\"" $< | $(POETRY) run armory run --check - $(ARGS)

scenario_configs/oscar/:
> mkdir -p $@

# Witness the magic of .SECONDEXPANSION! $$(*D) is the directory of the matched target exclude prefixes,
# and $$(@F) is the filename of the matched target. We used the | (order-only) to separate
# the base scenario JSON file and model weights .pth file so we can reference them using $| and $<,
# respectively. See Automatic Variables in Makefile documentation.
.SECONDEXPANSION:
results/%.json: $(MODEL_ZOO)/$$(*D)/*.pth | scenario_configs/oscar/$$(@F)
> $(if $(word 1, $(shell if [ -f $< ]; then echo $<; else echo; fi)),,$(error There are no pth files in $(MODEL_ZOO)/$(*D)/))
> $(if $(word 2, $^),$(error There are $(words $^) pth files in $(MODEL_ZOO)/$(*D)/. There should only be one to use: $^),)
> mkdir -p $(@D)
> cat $| | $(JQ) ".model.weights_file = \"$(*D)/$(shell basename $<)\"" \
         | $(JQ) ".sysconfig.output_filename = \"$(@F)\"" > $@

$(MODEL_ZOO)/%.pth:
> $(error No model exists in "$(@D)". You either need to train the model or ask someone for it.)

include *.mk
