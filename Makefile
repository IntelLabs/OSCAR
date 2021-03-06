# Change these as necessary
DATASETS ?= /raid/datasets
DOCKER ?= nvidia-docker
ARMORY_CONFIG ?= $(HOME)/.armory/config.json

# Don't change these
MODEL_ZOO = oscar/model_zoo
POETRY = $(HOME)/.poetry/bin/poetry
ARMORY = $(shell which armory)
DOCKER_IMAGE_TAG = intellabs/oscar:0.13.3
JQ = jq --indent 4 -r
GIT_SUBMODULES = lib/armory/.git lib/MARS/MARS/.git lib/mmskeleton/mmskeleton/.git
MAKEFILE_DIR = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ARMORY_SCENARIOS = scenario_configs
SCENARIOS = $(ARMORY_SCENARIOS)/oscar
RESULTS = results

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

ifneq ($(ARMORY_OUTPUT_DIR), $(MAKEFILE_DIR)$(RESULTS))
    $(warning $(YELLOW)WARNING: Please configure armory to use $(WHITE)output_dir$(YELLOW) as $(WHITE)$(MAKEFILE_DIR)$(RESULTS)$(YELLOW); current value is $(ARMORY_OUTPUT_DIR)$(RESET).)
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
clean: clean_scenarios clean_precomputed
> rm -f oscar/model_zoo/MiDaS/model.pt
> git submodule deinit -f .
> rm -rf .venv

.PHONY: clean_scenarios
clean_scenarios:
> rm -f $(ARMORY_SCENARIOS)/**/*.json
> rm -f $(ARMORY_SCENARIOS)/*.json

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
> sed -i "s/python$$/python3/g" $(POETRY)

.PHONY: poetry
poetry: $(POETRY) ## Launch poetry with ARGS
> $(POETRY) $(ARGS)

lib/%/.git:
> git submodule update --init `dirname $@`

.venv: $(POETRY) $(GIT_SUBMODULES) pyproject.toml
> $(POETRY) run pip install pip==20.2.4
> $(POETRY) install
> $(POETRY) run pip install https://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp37-cp37m-linux_x86_64.whl \
                            https://download.pytorch.org/whl/cu110/torchvision-0.8.1%2Bcu110-cp37-cp37m-linux_x86_64.whl
> $(POETRY) run pip uninstall -y tensorflow
> $(POETRY) run pip install --use-feature=2020-resolver tensorflow-gpu==1.15.0
> touch $@
> @echo "$(YELLOW)Make sure to configure armory if you haven't already:$(RESET)"
> @echo "    output_dir: $(GREEN)$(MAKEFILE_DIR)$(RESULTS)$(RESET)"
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
            .venv \
            ucf101_submission \
            dapricot_submission
> $(info $(GREEN)All submission files should be in the $@/ folder now.$(RESET))

.PHONY: clean_submission
clean_submission:
> rm -f submission/*

#
# Armory Targets
#
.PRECIOUS: lib/armory/scenario_configs/%.json
lib/armory/scenario_configs/%.json: lib/armory/.git
> touch $@

$(ARMORY_SCENARIOS)/:
> mkdir -p $@

$(ARMORY_SCENARIOS)/%.json: lib/armory/scenario_configs/%.json $(ARMORY_SCENARIOS)/
> @test -s $< || { echo "$(RED)Armory scenario $*.json does not exist!$(RESET)"; exit 1; }
> cat $< | $(JQ) '.sysconfig.docker_image = "$(DOCKER_IMAGE_TAG)"' > $@

$(RESULTS)/%.json.armory_run: $(RESULTS)/%.json | .venv
> $(JQ) ".sysconfig.output_dir = \"$(*D)/armory_run\"" $< | $(POETRY) run armory run --no-docker - $(ARGS)

$(RESULTS)/%.json.armory_check: $(RESULTS)/%.json | .venv
> $(JQ) ".sysconfig.output_dir = \"$(*D)/armory_check\"" $< | $(POETRY) run armory run --no-docker --check - $(ARGS)

$(RESULTS)/%.json.armory_docker_run: $(RESULTS)/%.json | .venv
> $(JQ) ".sysconfig.output_dir = \"$(*D)/armory_docker_run\"" $< | $(POETRY) run armory run - $(ARGS)

$(RESULTS)/%.json.armory_docker_check: $(RESULTS)/%.json | .venv
> $(JQ) ".sysconfig.output_dir = \"$(*D)/armory_docker_check\"" $< | $(POETRY) run armory run --check - $(ARGS)

$(SCENARIOS)/%.json.armory_run: $(SCENARIOS)/%.json | .venv
> $(JQ) ".sysconfig.output_dir = \"$(*D)/armory_run\"" $< | $(POETRY) run armory run --no-docker - $(ARGS)

$(SCENARIOS)/%.json.armory_check: $(SCENARIOS)/%.json | .venv
> $(JQ) ".sysconfig.output_dir = \"$(*D)/armory_check\"" $< | $(POETRY) run armory run --no-docker --check - $(ARGS)

$(SCENARIOS)/%.json.armory_docker_run: $(SCENARIOS)/%.json | .venv
> $(JQ) ".sysconfig.output_dir = \"$(*D)/armory_docker_run\"" $< | $(POETRY) run armory run - $(ARGS)

$(SCENARIOS)/%.json.armory_docker_check: $(SCENARIOS)/%.json | .venv
> $(JQ) ".sysconfig.output_dir = \"$(*D)/armory_docker_check\"" $< | $(POETRY) run armory run --check - $(ARGS)



$(SCENARIOS)/:
> mkdir -p $@

# Witness the magic of .SECONDEXPANSION! $$(*D) is the directory of the matched target exclude prefixes,
# and $$(@F) is the filename of the matched target. We used the | (order-only) to separate
# the base scenario JSON file and model weights .pth file so we can reference them using $| and $<,
# respectively. See Automatic Variables in Makefile documentation.
.SECONDEXPANSION:
$(RESULTS)/%.json: $(MODEL_ZOO)/$$(*D)/*.pth | $(SCENARIOS)/$$(@F)
> $(if $(word 1, $(shell if [ -f $< ]; then echo $<; else echo; fi)),,$(error There are no pth files in $(MODEL_ZOO)/$(*D)/))
> $(if $(word 2, $^),$(error There are $(words $^) pth files in $(MODEL_ZOO)/$(*D)/. There should only be one to use: $^),)
> mkdir -p $(@D)
> cat $| | $(JQ) ".model.weights_file = \"$(*D)/$(shell basename $<)\"" \
         | $(JQ) ".sysconfig.output_filename = \"$(@F)\"" > $@

$(MODEL_ZOO)/%.pth:
> $(error No model exists in "$(@D)". You either need to train the model or ask someone for it.)

#
# Precompute preprocessed data for training
#
PRECOMPUTED_DATA_DIR = $(ARMORY_DATASET_DIR)/precomputed

$(PRECOMPUTED_DATA_DIR):
> mkdir -p $@

$(PRECOMPUTED_DATA_DIR)/preprocessed.%.train.scenario.json: $(SCENARIOS)/%.json | $(PRECOMPUTED_DATA_DIR)
> cat $< | $(JQ) '.dataset.train_split = "train"' \
         | $(JQ) '.dataset.framework = "numpy"' \
         | $(JQ) 'del(.dataset.eval_split)' > $@

$(PRECOMPUTED_DATA_DIR)/preprocessed.%.test.scenario.json: $(SCENARIOS)/%.json | $(PRECOMPUTED_DATA_DIR)
> cat $< | $(JQ) '.dataset.eval_split = "test"' \
         | $(JQ) '.dataset.framework = "numpy"' \
         | $(JQ) 'del(.dataset.train_split)' > $@

.DELETE_ON_ERROR: $(PRECOMPUTED_DATA_DIR)/preprocessed.%.h5
$(PRECOMPUTED_DATA_DIR)/preprocessed.%.h5: $(PRECOMPUTED_DATA_DIR)/preprocessed.%.scenario.json
> $(POETRY) run python -m oscar.data.preprocess_armory_data $< $@

.PHONY: preprocessed.phony
preprocessed.%.train preprocessed.%.test: preprocessed.phony # phony by proxy
> $(MAKE) $(PRECOMPUTED_DATA_DIR)/$@.h5
> @echo "Preprocessed data located at $(PRECOMPUTED_DATA_DIR)/$@.h5"

.PHONY: clean_precomputed
clean_precomputed:
> rm -rf $(PRECOMPUTED_DATA_DIR)


include *.mk
