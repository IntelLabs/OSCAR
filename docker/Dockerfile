FROM twosixarmory/armory:0.16.2

# Armory submissions process does not support COPY in Dockerfile, so we
# manually install the dependencies here. Hopefully they don't get out of sync!
# Install Cython first because pycocotools will complain otherwise
RUN pip install git+https://github.com/IntelLabs/MART@v0.2.1 \
                kornia==0.6.5

# Set torch directory to cache models
ENV TORCH_HOME="/workspace"

WORKDIR /workspace
