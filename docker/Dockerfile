FROM twosixarmory/armory:0.18.0

# Armory submissions process does not support COPY in Dockerfile, so we
# manually install the dependencies here. Hopefully they don't get out of sync!
# NOTE: We install ART 1.14.1 with a bugfix that enables us to use PyTorch 2.0.
#       We spent a lot of time trying to use ART 1.15 with PyTorch 2.0, but we
#       failed to make this compatible.
RUN pip install git+https://github.com/IntelLabs/MART@v0.5.2 \
                kornia==0.6.12 \
                --force-reinstall git+https://github.com/Trusted-AI/adversarial-robustness-toolbox@5f039c0255212a1357b7ede63860d6dae4b4925f \
                git+https://github.com/mariusarvinte/score_sde_pytorch

# NOTE: We do this because of incompatibilities with the conda version
RUN pip install --force-reinstall torchaudio

# NOTE: MART causes us to upgrade protobuf but this causes all kinds of errors
#       because docker never forgets.
RUN pip install protobuf==3.19.6

# NOTE: We do this because of errors in the code on using numba
RUN pip install numba==0.56.2

# Set torch directory to cache models
ENV TORCH_HOME="/workspace"

WORKDIR /workspace
