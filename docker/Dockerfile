FROM python:3.10-slim

LABEL description="Single-cell SAIGE pipeline: persampleprocessing + concatandwrite"

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libhdf5-dev \
        pkg-config \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        scanpy==1.9.8 \
        anndata==0.10.3 \
        pandas==2.1.0 \
        numpy==1.26.0 \
        scipy==1.11.0 \
        h5py==3.10.0

RUN mkdir -p /pipeline

COPY scripts/persampleprocessing.py /pipeline/persampleprocessing.py
COPY scripts/concatandwrite.py      /pipeline/concatandwrite.py

RUN chmod +x /pipeline/persampleprocessing.py \
             /pipeline/concatandwrite.py

WORKDIR /pipeline
