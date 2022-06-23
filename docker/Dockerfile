FROM twosixarmory/pytorch:0.15.3

# Remove this when Armory updates their base images!
RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update

# Install system-level dependencies that cv2 depends upon
RUN apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6

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
RUN /opt/conda/bin/pip install Cython==0.29.28
RUN /opt/conda/bin/pip install kornia==0.6.5 \
                               pycocotools==2.0.4 \
                               pytorch_lightning==1.6.3

# Install detectron2 using pip
RUN /opt/conda/bin/pip install https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/detectron2-0.6%2Bcu113-cp38-cp38-linux_x86_64.whl

# Set cache directory for Detectron2 models
ENV FVCORE_CACHE="/tmp"

# Set torch directory to cache models
ENV TORCH_HOME="/workspace"

# Loop forever
CMD tail -f /dev/null

