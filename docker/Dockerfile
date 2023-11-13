FROM twosixarmory/armory:0.19.1

# Armory submissions process does not support COPY in Dockerfile, so we
# manually install the dependencies here. Hopefully they don't get out of sync!
RUN pip install git+https://github.com/IntelLabs/MART@v0.5.3 \
                kornia==0.6.12 \
                git+https://github.com/mariusarvinte/score_sde_pytorch \
                diffusers==0.20.0 \
                accelerate==0.21.0

# [Optional] Install pytorch targeting the CUDA version in the image. (torch 2.1.0 requires cu118+)
# It looks like the nvidia docker runtime is fine with running cu120 programs on a cu116 image, as long as the host driver supports cu120.
# (ARMORY image is based on nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04)
# RUN pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# NOTE: We do this because of errors in the code on using numba
RUN pip install numba==0.56.2

# Set torch directory to cache models
ENV TORCH_HOME="/workspace"

WORKDIR /workspace
