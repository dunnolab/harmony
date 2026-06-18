FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venvs/flexdock
ENV CONDA_DIR=/opt/conda
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${VENV_PATH}/bin:${CONDA_DIR}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

ARG TORCH_VERSION=2.5.0
ENV CUDA_TAG=cu121

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    gfortran \
    git \
    libblas-dev \
    liblapack-dev \
    libxc-dev \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VENV_PATH}" && \
    "${VENV_PATH}/bin/pip" install --upgrade pip wheel setuptools

COPY requirements.txt /tmp/requirements.txt

RUN "${VENV_PATH}/bin/pip" install --no-cache-dir \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    "torch==${TORCH_VERSION}"

RUN "${VENV_PATH}/bin/pip" install --no-cache-dir -r /tmp/requirements.txt

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniforge.sh && \
    conda config --add channels conda-forge && \
    conda install -y xtb && \
    conda clean -afy

RUN useradd -m -s /bin/bash user && \
    mkdir -p /workspace && \
    chown -R user:user /workspace "${VENV_PATH}"

WORKDIR /workspace
USER user

