FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    freeglut3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./get_custom_scikit_image.sh /get_custom_scikit_image.sh
RUN chmod +x /get_custom_scikit_image.sh
RUN /get_custom_scikit_image.sh

RUN pip3 install matplotlib tqdm tensorboardX timm mmcv open3d xformers

RUN git clone --recurse-submodules https://github.com/tauzn-clock/NDDepth 
WORKDIR /NDDepth

