FROM twosixarmory/pytorch:0.12.3

RUN apt-get update

# Install system-level dependencies that cv2 depends upon
RUN apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx

# Install system-level dependencies that mmskeleton depends upon
RUN apt-get install -y --no-install-recommends cuda-compiler-$CUDA_PKG_VERSION cuda-command-line-tools-$CUDA_PKG_VERSION

# Remove Pillow dependency because it is version 7.1.2 instead of 7.0.0
#RUN /opt/conda/bin/pip uninstall -y Pillow
# Remove Pillow a second time because that leaves 7.0.0 installed in a weird state
#RUN /opt/conda/bin/pip uninstall -y Pillow

# Install poetry
#RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_VERSION="1.0.5" python
#RUN $HOME/.poetry/bin/poetry config virtualenvs.create false

# Install dependencies using poetry
#COPY oscar /workspace/oscar
#COPY lib /workspace/lib
#COPY poetry.lock pyproject.toml /workspace/
#RUN $HOME/.poetry/bin/poetry install

# Armory submissions process does not support COPY in Dockerfile, so we
# manually install the dependencies here. Hopefully they don't get out of sync!
# Install Cython first because pycocotools will complain otherwise
RUN /opt/conda/bin/pip install Cython==0.29.21
RUN /opt/conda/bin/pip install kornia==0.4.1 \
                               opencv-python==4.4.0.46 \
                               pycocotools==2.0.2 \
                               pytorch_lightning==1.0.6

# Install mmskeleton and its dependencies
# Clone mmskeleton and add missing __init__.py files for `pip install` to discover the `mmskeleton.deprecated.datasets` package
# The deprecated package is not used in mmskeleton code but still imported in `mmskeleton.datasets.skeleton.skeleton_process`
# (see: https://github.com/open-mmlab/mmskeleton/blob/b4c076baa9e02e69b5876c49fa7c509866d902c7/mmskeleton/datasets/skeleton/skeleton_process.py#L3)
RUN git clone https://github.com/open-mmlab/mmskeleton /tmp/mmskeleton/ \
    && cd /tmp/mmskeleton \
    && git reset --hard b4c076baa9e02e69b5876c49fa7c509866d902c7 \
    && touch mmskeleton/deprecated/__init__.py \
    && touch mmskeleton/deprecated/datasets/__init__.py
RUN printf "\e[1;33m%s\e[0m\n" "WARNING: The 'mmskeleton' package requires GPU during setup, which is typycally not available in '(nvidia-)docker build'." \
    && printf "\e[1;33m%s\e[0m\n" "Please make sure you have 'default-runtime' for the docker daemon set to 'nvidia'." \
    && printf "\e[1;33m%s\e[0m\n" "This can be changed in your '/etc/docker/daemon.json' and restarting docker." \
    && printf "\e[1;33m%s\e[0m\n" "For more details, please see these links:" \
    && printf "\e[1;33m%s\e[0m\n" "1. https://github.com/NVIDIA/nvidia-docker/wiki/Advanced-topics#default-runtime" \
    && printf "\e[1;33m%s\e[0m\n" "2. https://github.com/NVIDIA/nvidia-docker/issues/595#issuecomment-519714769" \
    && /opt/conda/bin/pip install /tmp/mmskeleton/

# Install detectron2 using pip
RUN /opt/conda/bin/pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html

# Set cache directory for Detectron2 models
ENV FVCORE_CACHE="/tmp"

# Set torch directory to cache models
ENV TORCH_HOME="/workspace"

# Loop forever
CMD tail -f /dev/null

