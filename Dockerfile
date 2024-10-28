FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# DUST3R
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/tauzn-clock/NDDepth 
WORKDIR /NDDepth

RUN pip3 install matplotlib tqdm tensorboardX timm mmcv open3d
RUN apt-get install freeglut3-dev -y